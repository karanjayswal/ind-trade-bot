"""Local dashboard API — FastAPI app that serves `web/` and streams the snapshot.

CONTRACTS.md section M. Three things in this file are load-bearing for safety and
are deliberately hard to switch off:

* **Loopback only.** `create_app` refuses to build an app whose configured bind host
  is not a loopback address, and :class:`LoopbackOnlyMiddleware` refuses every
  connection whose client (or whose own accepting socket) is a routable address.
  The middleware is inside the app, so a caller that bypasses :func:`run_server` and
  hands the app to uvicorn with `--host 0.0.0.0` still cannot be reached from the
  network. This process can place real orders and has no authentication; an
  unauthenticated trading API on a shared network is the single worst failure
  available here, so it fails closed in two independent places.
* **Arming is a pass-through.** `POST /api/arm` hands `confirm` straight to
  `engine.arm`. The server never compares the phrase itself, so exactly one place in
  the codebase decides what arms this bot.
* **`armed` is not a setting.** `POST /api/config` rejects any patch containing the
  key, and points at `/api/arm` / `/api/disarm`.

Websocket framing: `/ws` sends the *same object* as `GET /api/state`, one JSON
object per frame, so a client can use one render function for both. A frame that is
not a snapshot is an error frame and carries a top-level `"error"` key, which a
snapshot never has (a snapshot carries `"errors"`, plural).
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import logging
import math
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import clock
from .config import ARM_PHRASE, Config
from .errors import (
    ApiError,
    AuthError,
    ConfigError,
    DataError,
    IndbotError,
    OrderError,
    RateLimitError,
    RiskBlocked,
)
from .journal import Journal

if TYPE_CHECKING:  # pragma: no cover - import cycle / module may not exist yet
    from .engine import Engine

log = logging.getLogger(__name__)

#: Where the single-page UI lives. Only files under here are ever served.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: Explicit media types. A file whose suffix is not in this map is NOT served, so a
#: stray config.json, .db or .env inside web/ cannot be handed out by accident.
MEDIA_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
}

#: Seconds without an engine publish before /ws sends a snapshot anyway. The UI's
#: connection dot stays honest and a missed publish self-heals within one beat.
WS_HEARTBEAT_S = 10.0

#: A send that takes longer than this means a stalled client. Give up on it and close
#: rather than hold the socket open forever.
WS_SEND_TIMEOUT_S = 10.0

#: Completed backtest jobs kept in memory before the oldest finished one is evicted.
MAX_BACKTEST_JOBS = 20

#: Most specific first — RateLimitError subclasses ApiError, so order matters.
_ERROR_STATUS: tuple[tuple[type, int], ...] = (
    (RateLimitError, 429),
    (AuthError, 401),
    (ConfigError, 400),
    (RiskBlocked, 409),
    (OrderError, 502),
    (DataError, 503),
    (ApiError, 502),
    (IndbotError, 500),
)


# ---------------------------------------------------------------------------
# JSON encoding
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Serialise the objects the snapshot and the journal hand us.

    `as_dict()` wins over `dataclasses.asdict` because indbot's dataclasses add
    derived fields (a Position's P&L, a BucketVerdict's note) that the UI prints.
    """
    to_dict = getattr(obj, "as_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001 - a broken __repr__ must not kill a frame
            log.debug("as_dict() failed on %s", type(obj).__name__, exc_info=True)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseException):
        return {"error": type(obj).__name__, "detail": str(obj)}
    return str(obj)


def _scrub_nonfinite(obj: Any) -> Any:
    """Replace NaN/Infinity with None, recursively.

    JSON has no NaN. `json.dumps` would emit a bare `NaN` that `JSON.parse` rejects,
    which would blank the whole dashboard over one bad float — so a single unusable
    number becomes a null and the rest of the frame still renders.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nonfinite(v) for v in obj]
    return obj


def dumps(payload: Any) -> str:
    """JSON for the wire: strict (no NaN), with indbot's types understood."""
    try:
        return json.dumps(payload, default=_json_default, allow_nan=False)
    except ValueError:
        # allow_nan=False raised: something in there is NaN or Infinity. Scrub and
        # retry rather than drop the frame.
        log.warning("non-finite number in a response payload; sending nulls for it")
        plain = json.loads(json.dumps(payload, default=_json_default))
        return json.dumps(_scrub_nonfinite(plain), allow_nan=False)


def to_jsonable(payload: Any) -> Any:
    """Round-trip `payload` through :func:`dumps` into plain JSON types."""
    return json.loads(dumps(payload))


class SafeJSONResponse(JSONResponse):
    """JSONResponse that understands indbot's dataclasses and never emits NaN."""

    def render(self, content: Any) -> bytes:
        return dumps(content).encode("utf-8")


# ---------------------------------------------------------------------------
# the loopback rule
# ---------------------------------------------------------------------------


def _parse_host(host: str) -> ipaddress._BaseAddress | None:
    """Parse a host string as an IP address, or None when it is a name."""
    cleaned = (host or "").strip().strip("[]")
    if not cleaned:
        return None
    if "%" in cleaned:  # link-local scope id, e.g. fe80::1%en0
        cleaned = cleaned.split("%", 1)[0]
    try:
        return ipaddress.ip_address(cleaned)
    except ValueError:
        return None


def is_loopback(host: str) -> bool:
    """True when `host` is 127.0.0.0/8, ::1 or the literal name "localhost"."""
    cleaned = (host or "").strip().strip("[]").lower()
    if cleaned in ("localhost", "localhost.localdomain"):
        return True
    addr = _parse_host(cleaned)
    return bool(addr is not None and addr.is_loopback)


def assert_loopback(host: str, *, what: str = "bind host") -> str:
    """Return `host` unchanged if it is loopback, else raise ConfigError naming it.

    Called from `create_app` (so no app can exist for a routable host) and again
    from `run_server`. `0.0.0.0` on a cafe wifi turns this process into an
    unauthenticated trading API for everyone on the network — refuse, loudly.
    """
    cleaned = (host or "").strip()
    if not cleaned:
        raise ConfigError(
            f"no {what} configured; indbot binds to 127.0.0.1 only",
            key="host",
            allowed="127.0.0.1, ::1 or localhost",
        )
    if is_loopback(cleaned):
        return cleaned
    raise ConfigError(
        f"refusing to start: {what} {cleaned!r} is not a loopback address. "
        "indbot serves an unauthenticated API that can place real orders, so it "
        "listens on 127.0.0.1 only. If you need it from another machine, put it "
        "behind an SSH tunnel: ssh -L 8765:127.0.0.1:8765 <host>",
        key="host",
        allowed="127.0.0.1, ::1 or localhost",
    )


class LoopbackOnlyMiddleware:
    """ASGI guard: drop any connection that is not local to this machine.

    Pure ASGI rather than BaseHTTPMiddleware so it also covers the websocket scope.
    It checks both ends: the peer (`scope["client"]`) and the socket this connection
    was accepted on (`scope["server"]`, which uvicorn fills with the connection's own
    local address — a non-loopback value there means the process is bound to a
    routable interface despite `create_app`'s check).

    It also rejects a foreign `Origin`. CONTRACTS.md section M says loopback and no
    auth, and loopback alone is not enough: `POST /api/panic`, `/api/stop`,
    `/api/start` and `/api/disarm` take no body, so any web page the user happens to
    have open could fire one at http://127.0.0.1:<port> and flatten a live position
    or stop the bot mid-trade. A browser always labels such a request with its own
    Origin, and the dashboard's own Origin is loopback, so this blocks the drive-by
    without adding a password the UI would have to carry.

    A host string that is not an IP at all (Starlette's TestClient says
    "testclient", and some ASGI servers omit the field) is allowed through: a real
    remote peer always presents a real address, so this keeps tests and unix-socket
    deployments working without weakening the rule that matters.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") in ("http", "websocket"):
            bad = self._offending_address(scope)
            if bad is not None:
                where, host = bad
                log.error(
                    "REFUSED a %s connection: %s address %s is not loopback (path=%s)",
                    scope.get("type"), where, host, scope.get("path"),
                )
                await self._reject(scope, send, where, host)
                return
            origin = self._foreign_origin(scope)
            if origin is not None:
                log.error(
                    "REFUSED a %s request from origin %s (path=%s)",
                    scope.get("type"), origin, scope.get("path"),
                )
                await self._reject(scope, send, "Origin", origin)
                return
        await self.app(scope, receive, send)

    @staticmethod
    def _offending_address(scope: dict) -> tuple[str, str] | None:
        """The first non-loopback IP in the scope, as (which end, host), or None."""
        for key, where in (("client", "peer"), ("server", "local")):
            pair = scope.get(key)
            host = pair[0] if isinstance(pair, (tuple, list)) and pair else None
            if not isinstance(host, str):
                continue
            addr = _parse_host(host)
            if addr is None:
                continue  # a name, not an address: see the class docstring
            if not addr.is_loopback:
                return where, host
        return None

    @staticmethod
    def _foreign_origin(scope: dict) -> str | None:
        """The request's Origin when it is not a loopback one, else None.

        No Origin header means a non-browser client (curl, the test client) — allowed,
        because the loopback checks above already bound who can connect at all.
        """
        for name, value in scope.get("headers") or []:
            if name.lower() != b"origin":
                continue
            raw = value.decode("latin-1").strip()
            if not raw or raw == "null":
                return raw or None
            host = raw.split("://", 1)[-1]
            host = host.split("/", 1)[0]
            if host.startswith("["):                      # [::1]:8765
                host = host[1:].split("]", 1)[0]
            else:
                host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
            return None if is_loopback(host) else raw
        return None

    @staticmethod
    async def _reject(scope: dict, send: Callable, where: str, host: str) -> None:
        detail = (
            f"indbot accepts local connections only; this connection's {where} "
            f"is {host}."
        )
        if scope.get("type") == "websocket":
            # Close before accept: the handshake never completes.
            await send({"type": "websocket.close", "code": 1008, "reason": "not loopback"})
            return
        body = dumps({"error": "Forbidden", "detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


class ModeRequest(BaseModel):
    """`{"mode": "backtest"|"paper"|"live"}`."""

    mode: str = Field(..., min_length=1, description='"backtest", "paper" or "live"')


class ArmRequest(BaseModel):
    """`{"confirm": "..."}` — the phrase is compared by the engine, never here."""

    confirm: str = Field(
        ...,
        min_length=1,
        description="The exact arming phrase. Passed straight to engine.arm().",
    )


class ConfigPatchRequest(BaseModel):
    """A settings patch, flat (`{"min_score": 0.6}`) or wrapped (`{"patch": {...}}`).

    `armed` is rejected whichever form is used: arming is an act, not a setting.
    """

    model_config = ConfigDict(extra="allow")

    patch: dict[str, Any] | None = None

    def flatten(self) -> dict[str, Any]:
        """The patch itself, however the client chose to wrap it."""
        if isinstance(self.patch, dict):
            return dict(self.patch)
        return dict(self.__pydantic_extra__ or {})


class BacktestRequest(BaseModel):
    """`{"underlying": "NIFTY", "start": "2026-09-01", "end": "2026-10-20"}`."""

    underlying: str = Field(..., min_length=1)
    start: date
    end: date


# ---------------------------------------------------------------------------
# backtest jobs
# ---------------------------------------------------------------------------


@dataclass
class BacktestJob:
    """One queued/running/finished backtest, readable by id while it runs."""

    job_id: str
    underlying: str
    start: date
    end: date
    status: str = "queued"  # queued | running | done | error | cancelled
    cancel_requested: bool = False
    created_ts: int = 0
    started_ts: int | None = None
    finished_ts: int | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    updates: int = 0
    result: Any = None
    error: dict[str, Any] | None = None
    future: Future | None = field(default=None, repr=False, compare=False)

    def is_cancelled(self) -> bool:
        """Polled by the backtester between days so a long run can be stopped."""
        return self.cancel_requested

    def summary(self) -> dict[str, Any]:
        """Everything but the (potentially large) result block."""
        return {
            "job_id": self.job_id,
            "underlying": self.underlying,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "status": self.status,
            "cancel_requested": self.cancel_requested,
            "created_ts": self.created_ts,
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "progress": self.progress,
            "updates": self.updates,
            "error": self.error,
        }

    def as_dict(self) -> dict[str, Any]:
        """The summary plus the result, once there is one."""
        out = self.summary()
        out["result"] = self.result
        return out


class BacktestRunner:
    """Runs one backtest at a time and keeps each job's progress readable by id.

    `max_workers=1` on purpose: a backtest hammers the 5/s data rate limit that the
    live engine also needs, and two at once would starve the trading loop.
    """

    def __init__(self, make_backtester: Callable[[], Any]) -> None:
        self._make = make_backtester
        self._jobs: dict[str, BacktestJob] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indbot-backtest")

    def submit(self, underlying: str, start: date, end: date) -> BacktestJob:
        """Queue a backtest and return its job record (status "queued")."""
        if end < start:
            raise ConfigError(
                f"backtest end {end.isoformat()} is before start {start.isoformat()}"
            )
        job = BacktestJob(
            job_id=str(uuid.uuid4()),
            underlying=str(underlying).strip().upper(),
            start=start,
            end=end,
            created_ts=clock.epoch_s(clock.now()),
        )
        with self._lock:
            self._evict_locked()
            self._jobs[job.job_id] = job
        job.future = self._pool.submit(self._run, job)
        log.info(
            "backtest %s queued: %s %s..%s", job.job_id, job.underlying,
            start.isoformat(), end.isoformat(),
        )
        return job

    def get(self, job_id: str) -> BacktestJob | None:
        """One job by id, or None."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> BacktestJob:
        """Ask a job to stop. A year of NIFTY spends the same 5/s data budget the
        live engine needs, so the user must be able to call it off mid-run."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"no backtest job {job_id!r}")
            if job.status in ("done", "error", "cancelled"):
                return job
            job.cancel_requested = True
            queued = job.status == "queued"
        if queued and job.future is not None and job.future.cancel():
            # Never started: nothing will set its terminal state, so do it here.
            with self._lock:
                job.status = "cancelled"
                job.finished_ts = clock.epoch_s(clock.now())
        log.info("backtest %s cancellation requested", job_id)
        return job

    def list(self) -> list[dict[str, Any]]:
        """Every known job, newest first, without result payloads."""
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_ts, reverse=True)
        return [j.summary() for j in jobs]

    def shutdown(self) -> None:
        """Stop accepting work and drop anything still queued."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _evict_locked(self) -> None:
        """Drop the oldest finished jobs once the table is full."""
        if len(self._jobs) < MAX_BACKTEST_JOBS:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.status in ("done", "error", "cancelled")),
            key=lambda j: j.finished_ts or j.created_ts,
        )
        for job in finished[: max(1, len(self._jobs) - MAX_BACKTEST_JOBS + 1)]:
            self._jobs.pop(job.job_id, None)

    def _note(self, job: BacktestJob) -> Callable[[dict], None]:
        """The progress callback handed to Backtester.run."""

        def progress(update: Any) -> None:
            with self._lock:
                job.updates += 1
                if isinstance(update, dict):
                    job.progress = to_jsonable(update)
                else:  # a backtester that reports a string still shows something
                    job.progress = {"message": str(update)}

        return progress

    def _run(self, job: BacktestJob) -> None:
        """Worker body. Records the outcome on the job; never raises into the pool."""
        with self._lock:
            job.status = "running"
            job.started_ts = clock.epoch_s(clock.now())
        try:
            backtester = self._make()
            kwargs: dict[str, Any] = {"progress": self._note(job)}
            # Only pass `cancel` when this backtester accepts it: the engine may hand
            # us its own object, and an unexpected keyword must not fail the run.
            try:
                accepted = inspect.signature(backtester.run).parameters
            except (TypeError, ValueError):  # pragma: no cover - exotic callables
                accepted = {}
            if "cancel" in accepted:
                kwargs["cancel"] = job.is_cancelled
            result = backtester.run(job.underlying, job.start, job.end, **kwargs)
            # Convert here, not at read time: a result that cannot be serialised is a
            # failed job, not a job that 500s on every poll.
            payload = to_jsonable(result)
            with self._lock:
                job.result = payload
                job.status = "cancelled" if job.cancel_requested else "done"
        except IndbotError as exc:
            log.warning("backtest %s failed: %s", job.job_id, exc)
            with self._lock:
                job.error = error_payload(exc)
                job.status = "error"
        except Exception as exc:  # noqa: BLE001 - the UI must see the failure
            log.exception("backtest %s crashed", job.job_id)
            with self._lock:
                job.error = {"error": type(exc).__name__, "detail": str(exc)}
                job.status = "error"
        finally:
            with self._lock:
                job.finished_ts = clock.epoch_s(clock.now())


# ---------------------------------------------------------------------------
# websocket plumbing
# ---------------------------------------------------------------------------


class SnapshotStream:
    """Single-slot mailbox between the engine thread and one websocket.

    The engine publishes from its own thread and must never block on a slow UI, so a
    new snapshot overwrites an undelivered one: the client always gets the freshest
    state and the frames in between are dropped on purpose. `offer` never blocks and
    never raises, because it runs inside the engine's tick.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        self._closed = False
        self.dropped = 0

    def offer(self, snapshot: Any) -> None:
        """Engine-thread entry point: hand a snapshot to the socket's event loop."""
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._put, snapshot)
        except RuntimeError:
            # The loop is closed: this socket is already gone. Stop offering.
            self._closed = True
        except Exception:  # noqa: BLE001 - a UI socket must never break a tick
            log.debug("could not queue a snapshot for a websocket", exc_info=True)

    def _put(self, snapshot: Any) -> None:
        """Runs on the event-loop thread, where the queue is safe to touch."""
        if self._closed:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - raced with the consumer
                pass
        try:
            self._queue.put_nowait(snapshot)
        except asyncio.QueueFull:  # pragma: no cover - same race, other direction
            self.dropped += 1

    async def get(self, timeout: float) -> Any | None:
        """The next snapshot, or None when `timeout` seconds pass without one."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        """Stop accepting snapshots."""
        self._closed = True


async def _drain_client(websocket: WebSocket) -> str:
    """Read and discard client frames; returns when the client goes away.

    A socket that only sends does not notice a disconnect until its next write, so
    this task exists purely to detect the close promptly.
    """
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return "client disconnected"
    except (RuntimeError, ConnectionError):
        return "connection closed"
    except Exception:  # noqa: BLE001
        log.debug("websocket reader stopped", exc_info=True)
        return "reader error"


# ---------------------------------------------------------------------------
# error shaping
# ---------------------------------------------------------------------------


def status_for(exc: BaseException) -> int:
    """The HTTP status an indbot error maps to."""
    for kind, status in _ERROR_STATUS:
        if isinstance(exc, kind):
            return status
    return 500


def error_payload(exc: BaseException) -> dict[str, Any]:
    """`{"error": ..., "detail": ...}` plus whatever context the error carries."""
    payload: dict[str, Any] = {
        "error": type(exc).__name__,
        "detail": getattr(exc, "message", None) or str(exc) or type(exc).__name__,
    }
    as_dict = getattr(exc, "as_dict", None)
    if callable(as_dict):
        try:
            extra = as_dict()
        except Exception:  # noqa: BLE001
            extra = {}
        if isinstance(extra, dict):
            for key in ("context", "reasons", "status", "payload"):
                value = extra.get(key)
                if value not in (None, {}, []):
                    payload[key] = value
    user_message = getattr(exc, "user_message", None)
    if isinstance(user_message, str) and user_message:
        payload["user_message"] = user_message
    return payload


# ---------------------------------------------------------------------------
# static files
# ---------------------------------------------------------------------------


def _web_file(web_dir: Path, rel: str) -> Path:
    """Resolve `rel` inside `web_dir` or raise a 404.

    Rejects traversal, absolute paths, symlinks out of the tree and any suffix not in
    MEDIA_TYPES, so this serves the UI and nothing else — never the filesystem.
    """
    cleaned = (rel or "").strip().lstrip("/")
    if not cleaned or ".." in cleaned.split("/") or cleaned.startswith("~"):
        raise HTTPException(status_code=404, detail=f"no such file: {rel!r}")
    root = web_dir.resolve()
    candidate = (root / cleaned).resolve()
    if not candidate.is_relative_to(root):
        log.warning("blocked a path-traversal attempt: %r", rel)
        raise HTTPException(status_code=404, detail=f"no such file: {rel!r}")
    if candidate.suffix.lower() not in MEDIA_TYPES:
        raise HTTPException(status_code=404, detail=f"not a servable file: {rel!r}")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"no such file: {rel!r}")
    return candidate


def _file_response(path: Path) -> FileResponse:
    """FileResponse with an explicit media type and no caching of the UI shell."""
    return FileResponse(
        path,
        media_type=MEDIA_TYPES[path.suffix.lower()],
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# engine adapters
#
# engine.py is built by another agent against CONTRACTS.md section K. Everything the
# contract guarantees is called directly; anything it does not name is probed with
# getattr so a missing attribute degrades one field instead of breaking a route.
# ---------------------------------------------------------------------------


def _engine_running(engine: Any) -> bool | None:
    """Whether the tick loop is alive, or None when the engine does not say."""
    for name in ("running", "is_running", "alive", "started"):
        value = getattr(engine, name, None)
        if isinstance(value, bool):
            return value
        if callable(value):
            try:
                result = value()
            except Exception:  # noqa: BLE001
                return None
            if isinstance(result, bool):
                return result
    thread = getattr(engine, "thread", None) or getattr(engine, "_thread", None)
    is_alive = getattr(thread, "is_alive", None)
    if callable(is_alive):
        try:
            return bool(is_alive())
        except Exception:  # noqa: BLE001
            return None
    return None


def _leaderboard(engine: Any) -> list[Any]:
    """The learning leaderboard, from the engine's learner or from its snapshot."""
    learner = getattr(engine, "learner", None)
    board = getattr(learner, "leaderboard", None)
    if callable(board):
        return list(board())
    snap = engine.snapshot()
    rows = snap.get("leaderboard") if isinstance(snap, dict) else None
    return list(rows) if isinstance(rows, list) else []


def _make_backtester_factory(engine: Any, cfg: Config) -> Callable[[], Any]:
    """Build the callable that constructs a Backtester for a job.

    Imported lazily: `indbot.backtest` pulls in the API client, and the dashboard
    must still start (and still show why) when a backtest cannot be built.
    """

    def factory() -> Any:
        own = getattr(engine, "make_backtester", None)
        if callable(own):
            return own()
        from .backtest import Backtester  # local: optional at dashboard start-up

        # CONTRACTS.md K does not say the engine exposes its client; the real engine
        # keeps it on `_client`. Try both before giving up.
        client = getattr(engine, "client", None) or getattr(engine, "_client", None)
        if client is None:
            raise ConfigError(
                "a backtest needs the INDstocks API client, and the engine has none. "
                "Set INDSTOCKS_ACCESS_TOKEN and restart, then run `python3 run.py doctor`."
            )
        instruments = getattr(engine, "instruments", None)
        if instruments is None:
            from .instruments import Instruments

            instruments = Instruments(client)
        # A file-less clone: a backtest must never rewrite the user's config.json,
        # and copy_with() also forces armed=False so a sweep can never place an order.
        return Backtester(client, instruments, cfg.copy_with())

    return factory


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------


def create_app(
    engine: "Engine",
    journal: Journal,
    cfg: Config,
    *,
    web_dir: Path | None = None,
) -> FastAPI:
    """Build the dashboard app for a running engine.

    Raises ConfigError before anything is wired up when `cfg.host` is not a loopback
    address, so there is no way to obtain an app object for a routable bind.
    """
    host = assert_loopback(getattr(cfg, "host", "127.0.0.1"), what="configured host")
    root = (web_dir or WEB_DIR).resolve()

    runner = BacktestRunner(_make_backtester_factory(engine, cfg))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        """Own the backtest pool. The engine's own start/stop stays with run.py."""
        try:
            yield
        finally:
            runner.shutdown()

    app = FastAPI(
        title="indbot",
        description="Local control panel for the indbot index-options trading bot.",
        version="1.0",
        default_response_class=SafeJSONResponse,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(LoopbackOnlyMiddleware)

    app.state.engine = engine
    app.state.journal = journal
    app.state.cfg = cfg
    app.state.host = host
    app.state.web_dir = root
    app.state.backtests = runner

    if not root.is_dir():
        log.warning("web directory %s does not exist; the UI will 404 until it does", root)

    # -- errors --------------------------------------------------------------

    @app.exception_handler(IndbotError)
    async def _indbot_error(request: Request, exc: IndbotError) -> Response:
        """Map every indbot error onto a structured response in one place."""
        status = status_for(exc)
        level = logging.WARNING if status < 500 else logging.ERROR
        log.log(level, "%s %s -> %s: %s", request.method, request.url.path, status, exc)
        return SafeJSONResponse(status_code=status, content=error_payload(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
        """Same `{"error", "detail"}` shape for 404s and friends."""
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return SafeJSONResponse(
            status_code=exc.status_code,
            content={"error": f"HTTP{exc.status_code}", "detail": detail},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
        """A malformed body is a 422 with the field errors, not a stack trace."""
        return SafeJSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "detail": "the request body is not valid",
                "fields": to_jsonable(exc.errors()),
            },
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> Response:
        """Anything unmapped: log the traceback, answer in the same shape."""
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return SafeJSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    # -- state ---------------------------------------------------------------

    @app.get("/api/state", summary="Everything the UI renders")
    def get_state() -> Any:
        """The engine snapshot — identical to what `/ws` pushes."""
        return engine.snapshot()

    @app.get("/api/health", summary="Liveness and the two facts that matter")
    def get_health() -> dict[str, Any]:
        """Never raises: it is what the UI polls when the websocket is down."""
        now = clock.now()
        payload: dict[str, Any] = {
            "ok": True,
            "now": now.isoformat(),
            "market_open": clock.is_market_open(now),
            "mode": getattr(cfg, "mode", None),
            "armed": bool(getattr(cfg, "armed", False)),
            "engine_running": _engine_running(engine),
            "host": host,
        }
        try:
            journal.get_meta("schema_version")   # cheapest read that touches the file
            payload["db"] = True
        except Exception as exc:  # noqa: BLE001 - health must answer even when sick
            payload["ok"] = False
            payload["db"] = False
            payload["detail"] = f"journal unreadable: {exc}"
        return payload

    # -- configuration -------------------------------------------------------

    @app.get("/api/config", summary="Current settings")
    def get_config() -> dict[str, Any]:
        """Exactly the keys `POST /api/config` accepts."""
        return cfg.as_dict()

    @app.post("/api/config", summary="Patch settings")
    def post_config(body: ConfigPatchRequest) -> dict[str, Any]:
        """Validate and apply a patch; returns the changed keys and the new config.

        Unknown keys are an error rather than a silent no-op: a settings form that
        drops a misspelled key leaves the user believing a limit was raised when it
        was not.
        """
        patch = body.flatten()
        if "armed" in patch:
            # Arming is an act with a confirmation phrase, not a form field.
            raise ConfigError(
                '"armed" is not a setting: POST /api/arm with '
                f'{{"confirm": "{ARM_PHRASE}"}} to arm, or POST /api/disarm to stop. '
                "Resend this patch without the armed key.",
                key="armed",
                allowed="use /api/arm or /api/disarm",
            )
        if not patch:
            return {"changed": [], "config": cfg.as_dict()}
        changed = cfg.update(patch, strict_unknown=True)
        return {"changed": changed, "config": cfg.as_dict()}

    # -- mode, arming, panic -------------------------------------------------

    @app.post("/api/mode", summary="Switch between backtest, paper and live")
    def post_mode(body: ModeRequest) -> dict[str, Any]:
        """Hands the mode to the engine, which flattens positions on the way out of live."""
        engine.set_mode(body.mode)
        return {"ok": True, "mode": getattr(cfg, "mode", body.mode),
                "armed": bool(getattr(cfg, "armed", False))}

    @app.post("/api/arm", summary="Arm live trading")
    def post_arm(body: ArmRequest) -> dict[str, Any]:
        """Pass `confirm` through to the engine, which owns the comparison.

        The server deliberately does not check the phrase: one place in this
        codebase decides what arms this bot, and it is not the HTTP layer.
        """
        engine.arm(body.confirm)
        return {"ok": True, "armed": bool(getattr(cfg, "armed", False)),
                "mode": getattr(cfg, "mode", None)}

    @app.post("/api/disarm", summary="Block live orders")
    def post_disarm() -> dict[str, Any]:
        """Disarming never needs a confirmation — the safe direction is always free."""
        engine.disarm()
        return {"ok": True, "armed": bool(getattr(cfg, "armed", False))}

    @app.post("/api/panic", summary="Flatten everything and halt the day")
    def post_panic() -> Any:
        """Exits every open position now and stops trading for the rest of the day."""
        log.error("PANIC requested from the dashboard")
        return engine.panic()

    @app.post(
        "/api/positions/{position_id}/close",
        summary="Square off one position by hand",
    )
    def post_close_position(position_id: str) -> dict[str, Any]:
        """The positions panel's manual CLOSE button.

        Separate from /api/panic: closing one position is a normal act and leaves the
        day running, where panic flattens everything and halts. Errors propagate
        through the IndbotError handler so the UI shows the broker's own words — a
        failed exit on a live position is the one thing the user must not be able to
        mistake for a success.
        """
        log.warning("manual close requested for position %s", position_id)
        return {"ok": True, "closed": engine.close_position(position_id, reason="manual")}

    @app.post("/api/start", summary="Start the tick loop")
    def post_start() -> dict[str, Any]:
        engine.start()
        return {"ok": True, "engine_running": _engine_running(engine)}

    @app.post("/api/stop", summary="Stop the tick loop")
    def post_stop() -> dict[str, Any]:
        """Stops scanning. Open positions are left alone — use /api/panic to flatten."""
        engine.stop()
        return {"ok": True, "engine_running": _engine_running(engine)}

    # -- journal -------------------------------------------------------------

    @app.get("/api/trades", summary="Trade history")
    def get_trades(
        day: date | None = Query(None, description="IST calendar day, YYYY-MM-DD"),
        mode: str | None = Query(None, description="backtest | paper | live"),
        limit: int = Query(200, ge=1, le=2000),
        status: str | None = Query(None, description="open | closed"),
        bucket: str | None = Query(None),
    ) -> dict[str, Any]:
        """Trades filtered by day, mode, status or learning bucket, newest first."""
        rows = journal.trades(day=day, mode=mode, limit=limit, status=status, bucket=bucket)
        return {
            "trades": rows,
            "count": len(rows),
            "day": day.isoformat() if day else None,
            "mode": mode,
        }

    @app.get("/api/signals", summary="Signal feed, taken and rejected")
    def get_signals(
        day: date | None = Query(None, description="IST calendar day, YYYY-MM-DD"),
        mode: str | None = Query(None),
        limit: int = Query(200, ge=1, le=2000),
        taken: bool | None = Query(None, description="false is the why-no-trade feed"),
        underlying: str | None = Query(None),
    ) -> dict[str, Any]:
        """Every evaluated signal with the reason it was or was not traded."""
        rows = journal.signals(
            day=day, limit=limit, mode=mode, taken=taken, underlying=underlying
        )
        return {"signals": rows, "count": len(rows), "taken": taken}

    @app.get("/api/leaderboard", summary="What the bot has learned")
    def get_leaderboard() -> dict[str, Any]:
        """Per-bucket verdicts, best expectancy first."""
        rows = _leaderboard(engine)
        return {"leaderboard": rows, "count": len(rows)}

    @app.get("/api/equity", summary="Realised P&L per day")
    def get_equity(
        mode: str | None = Query(None),
        days: int = Query(30, ge=1, le=365),
    ) -> dict[str, Any]:
        """Not in section M, but the UI's equity curve needs a source outside a backtest."""
        chosen = mode or getattr(cfg, "mode", "paper")
        return {"mode": chosen, "days": days,
                "curve": journal.equity_curve(mode=chosen, days=days)}

    # -- backtests -----------------------------------------------------------

    @app.post("/api/backtest", summary="Start a backtest")
    def post_backtest(body: BacktestRequest) -> dict[str, Any]:
        """Queue a replay over real historical option premiums; returns its job id."""
        job = runner.submit(body.underlying, body.start, body.end)
        return {"job_id": job.job_id, "status": job.status,
                "underlying": job.underlying,
                "start": job.start.isoformat(), "end": job.end.isoformat()}

    @app.get("/api/backtest/{job_id}", summary="Backtest progress or result")
    def get_backtest(job_id: str) -> Any:
        """Progress while it runs, the full result once it is done."""
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no backtest job {job_id!r}")
        return job.as_dict()

    @app.post("/api/backtest/{job_id}/cancel", summary="Stop a running backtest")
    def cancel_backtest(job_id: str) -> dict[str, Any]:
        """Ask the job to stop at its next day boundary and return its state."""
        return runner.cancel(job_id).summary()

    @app.get("/api/backtests", summary="Known backtest jobs")
    def list_backtests() -> dict[str, Any]:
        """So the UI can find a job again after a page reload."""
        return {"jobs": runner.list()}

    # -- websocket -----------------------------------------------------------

    @app.websocket("/ws")
    async def ws_state(websocket: WebSocket) -> None:
        """Push every engine snapshot, starting with one sent on connect.

        Frames are dropped, never queued, when the client cannot keep up: the engine
        thread offers into a single slot and the newest snapshot wins. A stale frame
        is worth nothing to a trading dashboard, and a blocked engine tick is worth
        considerably less than nothing.
        """
        await websocket.accept()
        loop = asyncio.get_running_loop()
        stream = SnapshotStream(loop)
        unsubscribe: Callable[[], None] | None = None
        reader: asyncio.Task | None = None
        why = "closed"
        try:
            try:
                unsubscribe = engine.subscribe(stream.offer)
            except Exception:  # noqa: BLE001 - still serve heartbeat snapshots
                log.exception("could not subscribe a websocket to the engine")

            reader = asyncio.create_task(_drain_client(websocket))

            # Paint immediately: the UI should not show empty cards until the next tick.
            await _ws_send(websocket, _ws_frame(engine))

            while True:
                getter = asyncio.ensure_future(stream.get(WS_HEARTBEAT_S))
                done, _ = await asyncio.wait(
                    {getter, reader}, return_when=asyncio.FIRST_COMPLETED
                )
                if reader in done:
                    getter.cancel()
                    why = reader.result() if not reader.cancelled() else "reader stopped"
                    break
                frame = getter.result()
                if frame is None:
                    # Heartbeat. Re-read the engine instead of resending a stale frame,
                    # so a publish we missed cannot leave the UI showing old numbers.
                    frame = _ws_frame(engine)
                await _ws_send(websocket, frame)
        except WebSocketDisconnect:
            why = "client disconnected"
        except asyncio.TimeoutError:
            # A send that never completed: the client is stalled, not just slow.
            log.warning("dropping a websocket that stopped reading")
            why = "send timed out"
        except (RuntimeError, ConnectionError) as exc:
            why = f"{type(exc).__name__}: {exc}"
        except asyncio.CancelledError:
            # Server shutting down. Clean up in `finally`, then let it propagate.
            why = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - never kill the server over one socket
            log.exception("websocket failed")
            why = f"{type(exc).__name__}: {exc}"
        finally:
            stream.close()
            if unsubscribe is not None:
                try:
                    unsubscribe()
                except Exception:  # noqa: BLE001
                    log.debug("unsubscribe failed", exc_info=True)
            if reader is not None and not reader.done():
                reader.cancel()
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001 - already closed is the normal case
                pass
            log.debug("websocket gone (%s), %d frames dropped", why, stream.dropped)

    # -- static UI (declared last so no asset route can shadow an API route) --

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        """The dashboard shell."""
        path = root / "index.html"
        if not path.is_file():
            # A clear 503 beats a bare 404 while the UI is still being built.
            raise HTTPException(
                status_code=503,
                detail=f"{path} is missing — the dashboard UI has not been built yet. "
                       "The API is up: try GET /api/state.",
            )
        return _file_response(_web_file(root, "index.html"))

    @app.get("/static/{rel:path}", include_in_schema=False)
    def static_file(rel: str) -> Response:
        """Assets under web/, by the path section M specifies."""
        return _file_response(_web_file(root, rel))

    @app.get("/{rel:path}", include_in_schema=False)
    def root_file(rel: str) -> Response:
        """Also serve web/ at the root, so index.html can use relative hrefs."""
        return _file_response(_web_file(root, rel))

    log.info("dashboard app built: host=%s web=%s", host, root)
    return app


def _ws_frame(engine: Any) -> Any:
    """A snapshot for the socket, or an error frame when the engine cannot produce one."""
    try:
        return engine.snapshot()
    except IndbotError as exc:
        log.warning("snapshot failed: %s", exc)
        return error_payload(exc)
    except Exception as exc:  # noqa: BLE001
        log.exception("snapshot crashed")
        return {"error": type(exc).__name__, "detail": str(exc)}


async def _ws_send(websocket: WebSocket, payload: Any) -> None:
    """Send one JSON frame, giving up on a client that has stopped reading."""
    try:
        text = dumps(payload)
    except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the socket
        log.exception("could not encode a websocket frame")
        text = dumps({"error": "EncodeError", "detail": str(exc)})
    await asyncio.wait_for(websocket.send_text(text), WS_SEND_TIMEOUT_S)


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------


def run_server(
    engine: "Engine",
    journal: Journal,
    cfg: Config,
    *,
    host: str | None = None,
    port: int | None = None,
    log_level: str = "info",
) -> None:
    """Build the app and serve it with uvicorn on a loopback address only.

    The host is validated here and again inside `create_app`; the app also refuses
    non-local connections at the ASGI layer. Three checks, because this is the one
    mistake in this project that would expose an unauthenticated order endpoint.
    """
    # Validate before importing uvicorn: a dangerous bind address must be reported as
    # a refusal, not masked by a missing-dependency ImportError.
    bind = assert_loopback(host or getattr(cfg, "host", "127.0.0.1"), what="bind host")
    bind_port = int(port if port is not None else getattr(cfg, "port", 8765))
    if not 1024 <= bind_port <= 65535:
        raise ConfigError(
            f"port must be between 1024 and 65535, got {bind_port}", key="port"
        )

    import uvicorn

    app = create_app(engine, journal, cfg)
    log.warning(
        "serving the indbot dashboard on http://%s:%d (mode=%s, armed=%s)",
        bind, bind_port, getattr(cfg, "mode", "?"), getattr(cfg, "armed", False),
    )
    uvicorn.run(app, host=bind, port=bind_port, log_level=log_level, access_log=False)


__all__ = [
    "ArmRequest",
    "BacktestJob",
    "BacktestRequest",
    "BacktestRunner",
    "ConfigPatchRequest",
    "LoopbackOnlyMiddleware",
    "MEDIA_TYPES",
    "ModeRequest",
    "SafeJSONResponse",
    "SnapshotStream",
    "WEB_DIR",
    "assert_loopback",
    "create_app",
    "dumps",
    "error_payload",
    "is_loopback",
    "run_server",
    "status_for",
    "to_jsonable",
]

"""Where the access token lives, and what can safely be said about it.

The token is a bearer credential: anyone holding it can place orders on the
account. Two rules follow, and the rest of this module exists to enforce them.

1. **It is never written into config.json.** The config is serialised into every
   snapshot, pushed over the websocket and rendered in the settings form, so a
   token in there would be broadcast to the browser and shown on screen. It goes
   in its own file, mode 0600, which nothing serialises.

2. **It is never returned by the API.** `describe()` yields a masked form plus
   the metadata decoded from the JWT — enough for the UI to say "valid for 3
   more hours" and nothing an onlooker could use.

INDmoney tokens are JWTs that expire at 07:00 IST, so a token minted at midday
lives about 18 hours and the bot needs a fresh one most mornings. `describe()`
reads `exp` out of the payload so the dashboard can warn before it lapses rather
than surfacing it as a 403 mid-session.

The JWT signature is NOT verified here — it is signed by INDmoney with a key we
do not hold. Decoding is only used to read expiry and client id for display; the
broker remains the sole authority on whether a token actually works.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from . import clock
from .errors import ConfigError

log = logging.getLogger(__name__)

TOKEN_ENV = "INDSTOCKS_ACCESS_TOKEN"

#: Default location, beside the database. Mode 0600, and gitignored.
DEFAULT_TOKEN_FILE = Path(__file__).resolve().parent.parent / "data" / "token.txt"

#: A JWT is three dot-separated base64url segments. Used only to reject obvious
#: paste errors early — a wrapping quote, a stray newline, half a copy.
_JWT_SHAPE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")


def token_file(path: str | os.PathLike[str] | None = None) -> Path:
    """The token file path: the argument, $INDBOT_TOKEN_FILE, or the default."""
    if path:
        return Path(path).expanduser()
    env = os.environ.get("INDBOT_TOKEN_FILE", "").strip()
    return Path(env).expanduser() if env else DEFAULT_TOKEN_FILE


def clean(raw: str) -> str:
    """Normalise a pasted token: trim, drop a 'Bearer ' prefix and any quotes.

    People paste tokens out of terminals and JSON blobs, so a leading `Bearer `,
    surrounding quotes and embedded newlines are all common and all harmless to
    strip. Anything left that is not JWT-shaped is a paste error worth catching
    before it becomes a 403 twenty minutes into a session.
    """
    tok = (raw or "").strip().strip("'\"").strip()
    if tok.lower().startswith("bearer "):
        tok = tok[7:].strip()
    tok = "".join(tok.split())          # kill newlines a wrapped paste introduces
    return tok


def validate_shape(raw: str) -> str:
    """Clean a token and check it looks like a JWT. Raises ConfigError if not."""
    tok = clean(raw)
    if not tok:
        raise ConfigError("the token is empty", key="token")
    if len(tok) < 40:
        raise ConfigError(
            f"that token is only {len(tok)} characters — it looks truncated "
            "(a full INDstocks token is around 266)",
            key="token",
        )
    if not _JWT_SHAPE.match(tok):
        raise ConfigError(
            "that does not look like a JWT: an INDstocks token is three "
            "dot-separated segments. Check for a missing character or a stray space.",
            key="token",
        )
    return tok


def _b64(segment: str) -> dict[str, Any]:
    """Decode one base64url JWT segment, padding as needed."""
    padded = segment + "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))


def payload(raw: str) -> dict[str, Any]:
    """The JWT's claims, or {} if it cannot be decoded.

    Never raises: a token that works but does not decode is still a working
    token, so a decode failure must not block its use.
    """
    try:
        return _b64(clean(raw).split(".")[1])
    except (IndexError, ValueError, binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def mask(raw: str) -> str:
    """"eyJhbGciOi…3Q05Mw" — safe to log, to show, and to put in a snapshot."""
    tok = clean(raw)
    if not tok:
        return ""
    if len(tok) <= 16:
        return "…" * 3
    return f"{tok[:10]}…{tok[-6:]}"


def describe(raw: str | None) -> dict[str, Any]:
    """Everything the UI may know about the token. Never includes the token.

    `expires_in_seconds` is negative once it has lapsed, so the dashboard can
    show "expired 40 minutes ago" rather than a bare boolean.
    """
    if not raw:
        return {"present": False, "masked": "", "length": 0, "source": None}

    tok = clean(raw)
    claims = payload(tok)
    exp = claims.get("exp")
    iat = claims.get("iat")
    now = clock.now()

    out: dict[str, Any] = {
        "present": True,
        "masked": mask(tok),
        "length": len(tok),
        "client_id": claims.get("clientID"),
        "partner_id": claims.get("partnerID"),
        "token_id": claims.get("tokenID"),
        "issuer": claims.get("iss"),
        "jwt": bool(claims),
    }
    if isinstance(iat, (int, float)):
        issued = clock.to_ist(iat)
        out["issued_at"] = issued.isoformat()
        out["issued_at_text"] = issued.strftime("%d %b %H:%M IST")
    if isinstance(exp, (int, float)):
        expires = clock.to_ist(exp)
        left = (expires - now).total_seconds()
        out.update(
            expires_at=expires.isoformat(),
            expires_at_text=expires.strftime("%d %b %H:%M IST"),
            expires_in_seconds=int(left),
            expired=left <= 0,
            expiring_soon=0 < left <= 3600,
            lifetime_hours=round((exp - iat) / 3600, 1) if isinstance(iat, (int, float)) else None,
        )
        out["status_text"] = _status_text(left, expires)
    else:
        out["expired"] = False
        out["status_text"] = "No expiry in the token; the broker decides when it stops working."
    return out


def _status_text(left_seconds: float, expires: datetime) -> str:
    """One plain sentence about the token's remaining life."""
    stamp = expires.strftime("%d %b at %H:%M IST")
    if left_seconds <= 0:
        mins = int(-left_seconds // 60)
        if mins < 60:
            return f"Expired {mins} minute{'s' if mins != 1 else ''} ago ({stamp}). Paste a new one."
        return f"Expired {int(mins // 60)}h ago ({stamp}). Paste a new one."
    hours = left_seconds / 3600
    if hours < 1:
        return f"Expires in {int(left_seconds // 60)} minutes, at {stamp}. Get a fresh one now."
    if hours < 3:
        return f"Expires in {hours:.1f} hours, at {stamp}. Worth refreshing before the next session."
    return f"Valid for {hours:.1f} more hours, until {stamp}."


def load(path: str | os.PathLike[str] | None = None) -> tuple[str, str | None]:
    """The active token and where it came from: ("<token>", "env"|"file"|None).

    The environment wins, so an operator can override the stored token for one
    run without touching the file.
    """
    env = clean(os.environ.get(TOKEN_ENV, ""))
    if env:
        return env, "env"
    target = token_file(path)
    try:
        if target.is_file():
            stored = clean(target.read_text(encoding="utf-8"))
            if stored:
                return stored, "file"
    except OSError as exc:
        log.warning("could not read the token file %s: %s", target, exc)
    return "", None


def save(raw: str, path: str | os.PathLike[str] | None = None) -> Path:
    """Write the token 0600, atomically. Returns the path written.

    Written through a temp file in the same directory so an interrupted write
    cannot leave a half-token that authenticates as nothing. Created 0600 from
    the start rather than chmod-ed afterwards, so it is never briefly readable.
    """
    tok = validate_shape(raw)
    target = token_file(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, (tok + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)
        os.chmod(target, 0o600)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise ConfigError(f"could not write the token file {target}: {exc}", key="token") from exc
    log.warning("access token updated (%s), stored at %s", mask(tok), target)
    return target


def clear(path: str | os.PathLike[str] | None = None) -> bool:
    """Delete the stored token. True if a file was removed."""
    target = token_file(path)
    try:
        if target.is_file():
            target.unlink()
            log.warning("stored access token deleted (%s)", target)
            return True
    except OSError as exc:
        raise ConfigError(f"could not delete {target}: {exc}", key="token") from exc
    return False


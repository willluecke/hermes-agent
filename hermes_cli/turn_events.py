"""Per-session emitter registry so plugins can put an event on the run stream.

Plugin hooks receive keyword arguments, never the agent, so a hook that wants
to show the user something mid-turn (a judge's verdict, a budget decision) has
no path to the gateway's run event queue. The conversation loop binds the
agent's ``tool_progress_callback`` here under the session id at the start of
every turn; a plugin calls :func:`emit_turn_event` with the same session id
and the gateway turns it into a run event the browser and the archive both
understand.

An emitter is a callable with the ``tool_progress_callback`` signature,
``(event_type, tool_name, preview, args, **kwargs)``. Emitting to a session
without a bound emitter, or through one that raises, is a no-op: judging is
advisory and must never break a turn.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_emitters: Dict[str, Callable[..., Any]] = {}
_LIMIT = 512


def bind_turn_emitter(session_id: str, emitter: Optional[Callable[..., Any]]) -> None:
    """Bind (or with ``None`` unbind) the emitter for ``session_id``."""
    key = str(session_id or "")
    if not key:
        return
    with _lock:
        if emitter is None:
            _emitters.pop(key, None)
            return
        _emitters[key] = emitter
        if len(_emitters) > _LIMIT:
            for stale in list(_emitters)[: len(_emitters) - _LIMIT]:
                _emitters.pop(stale, None)


def turn_emitter(session_id: str) -> Optional[Callable[..., Any]]:
    with _lock:
        return _emitters.get(str(session_id or ""))


def emit_turn_event(
    session_id: str,
    event_type: str,
    *,
    text: str = "",
    source: str = "",
    **payload: Any,
) -> bool:
    """Push one event onto the session's run stream. Returns whether it was sent."""
    emitter = turn_emitter(session_id)
    if emitter is None:
        return False
    try:
        emitter(event_type, source or "", text or "", None, **payload)
        return True
    except Exception:
        logger.debug("turn event %s for %s failed", event_type, session_id, exc_info=True)
        return False


__all__ = ["bind_turn_emitter", "emit_turn_event", "turn_emitter"]

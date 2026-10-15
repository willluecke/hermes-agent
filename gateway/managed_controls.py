"""Managed-run admission is distinct from possession of the ordinary API key.

Private controller/operator capabilities are host service credentials, never
model inputs or tools. The steward's native runtime has no credential-reading
or network tool surface. Other trusted host administrators remain trusted.
"""
import hmac
import json
import os
from pathlib import Path
import sqlite3


def authorized(headers, role):
    path=Path(os.environ.get('HERMES_MANAGED_CONTROL_FILE',str(Path.home()/'.hermes/managed-control.json')))
    try:
        expected=json.loads(path.read_text())[role]
        supplied=headers.get('X-Hermes-Managed-'+role.title(),'')
        return isinstance(expected,str) and len(expected)>=32 and hmac.compare_digest(supplied,expected)
    except (OSError,ValueError,KeyError,TypeError):
        return False


# Retain positive observations through a transient outage; a successful read
# showing release removes the fence. Keys include the store for test isolation.
_known_managed = set()

def managed_owner(session_id=None, run_id=None):
    chat = isinstance(session_id,str) and session_id.startswith('hermes-chat-')
    if not chat and not run_id: return False
    path=Path(os.environ.get('HERMES_SYNC_DATA_DIR',str(Path.home()/'.hermes-chat-sync')))/'sync.db'
    key=(str(path),session_id)
    def observed(owned):
        if chat:
            if owned: _known_managed.add(key)
            else: _known_managed.discard(key)
        return owned
    db=None
    try:
        if not path.exists(): return key in _known_managed
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=1)
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'conversation_goals' not in tables: return observed(False)
        if run_id and 'goal_turns' in tables:
            if db.execute("SELECT 1 FROM goal_turns WHERE run_id=? AND json_extract(data,'$.delegation_id') IS NOT NULL",(run_id,)).fetchone(): return True
        if not chat: return False
        conversation_id=session_id[len('hermes-chat-'):]
        row=db.execute('SELECT data FROM conversation_goals WHERE conversation_id=?',(conversation_id,)).fetchone()
        goal=json.loads(row[0]) if row else {}
        # Only active managed work owns the thread; a paused goal (blocked on
        # Will, capped, held or user-paused) must never lock him out of it.
        if goal.get('delegation_id') and goal.get('status')=='active': return observed(True)
        return observed(bool('goal_turns' in tables and db.execute("SELECT 1 FROM goal_turns WHERE conversation_id=? AND state IN ('launching','running','judging') AND json_extract(data,'$.delegation_id') IS NOT NULL",(conversation_id,)).fetchone()))
    except (OSError,sqlite3.Error,ValueError):
        # Ordinary human chat never depends on a successful observation.
        # Run steering remains fail-closed, as do positively known owners.
        return bool(run_id) or key in _known_managed
    finally:
        if db: db.close()


def start_allowed(headers, body):
    if not managed_owner(session_id=body.get('session_id')): return True
    return authorized(headers,'operator') or (body.get('native_goal') is False and authorized(headers,'controller'))


def steer_allowed(headers, run_id, managed=False):
    return (not managed and not managed_owner(run_id=run_id)) or authorized(headers,'operator')


def review_allowed(headers):
    """A steward final review is a managed start: the same controller capability
    that admits native_goal:false goal turns (or Will's operator capability),
    never the ordinary API key alone."""
    return authorized(headers,'controller') or authorized(headers,'operator')

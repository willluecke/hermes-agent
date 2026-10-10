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


def managed_owner(session_id=None, run_id=None):
    path=Path(os.environ.get('HERMES_SYNC_DATA_DIR',str(Path.home()/'.hermes-chat-sync')))/'sync.db'
    if not path.exists(): return False
    db=None
    try:
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=1)
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'conversation_goals' not in tables: return False
        if run_id and 'goal_turns' in tables:
            if db.execute("SELECT 1 FROM goal_turns WHERE run_id=? AND json_extract(data,'$.delegation_id') IS NOT NULL",(run_id,)).fetchone(): return True
        if not str(session_id or '').startswith('hermes-chat-'): return False
        conversation_id=session_id[len('hermes-chat-'):]
        row=db.execute('SELECT data FROM conversation_goals WHERE conversation_id=?',(conversation_id,)).fetchone()
        goal=json.loads(row[0]) if row else {}
        if goal.get('delegation_id') and goal.get('status') in {'active','paused'}: return True
        return bool('goal_turns' in tables and db.execute("SELECT 1 FROM goal_turns WHERE conversation_id=? AND state IN ('launching','running','judging') AND json_extract(data,'$.delegation_id') IS NOT NULL",(conversation_id,)).fetchone())
    except (OSError,sqlite3.Error,ValueError):
        return True  # Admission/control cannot infer freedom from an outage.
    finally:
        if db: db.close()


def start_allowed(headers, body):
    if not managed_owner(session_id=body.get('session_id')): return True
    return authorized(headers,'operator') or (body.get('native_goal') is False and authorized(headers,'controller'))


def steer_allowed(headers, run_id, managed=False):
    return (not managed and not managed_owner(run_id=run_id)) or authorized(headers,'operator')

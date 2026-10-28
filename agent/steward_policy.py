"""The Talos steward's runtime policy: identity, review bindings and the one refusal.

Will, 2026-10-28: the steward has full permissions except project deletion.
This module is not model advice; the model reads docs/steward-instructions.md.
"""
import json
import os
from pathlib import Path
import re
import sys
import threading

# The serialized system-origin review session. It is not a hermes-chat-*
# session, so a review never appears as a message in Will's talos chat.
REVIEW_SESSION = 'hermes-steward-review'
# Trusted review binding, carried to the scoped MCP server as environment.
REVIEW_ENV = 'HERMES_STEWARD_REVIEW'
REVIEW_ACTIONS = ('review', 'review_content', 'review_ack')
_HASH = re.compile(r'^[0-9a-f]{64}$')
_IDENT = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')  # Talos's safe-ID alphabet


def is_steward(*, session='', project='', cwd=''):
    return (str(session) in {'hermes-chat-talos', REVIEW_SESSION} or str(project) == 'talos-steward'
            or (isinstance(cwd,(str,Path)) and bool(cwd) and Path(cwd).resolve() == Path.home()/'.hermes/talos-steward'))


def steward_agent(agent):
    return is_steward(session=getattr(agent,'session_id',''), project=getattr(agent,'session_project',''), cwd=getattr(agent,'session_cwd',''))


def turn_source(value, *, session='', project=''):
    """The submitted chat message a steward turn answers: {conversation_id, message_id}.

    Carried on the run request as `turn_source` and handed to the model as a
    per-turn note, never through the turn instructions: a message id in the
    instructions changed them every turn, so the gateway rebuilt the resident
    agent for every message ("Session resumed from disk ... after the turn
    instructions changed", 2026-10-19). Only the steward conversation has one.
    """
    if not isinstance(value, dict) or set(value) != {'conversation_id', 'message_id'}:
        raise ValueError('turn_source must be an object with exactly conversation_id and message_id')
    if not all(isinstance(v, str) and _IDENT.fullmatch(v) for v in value.values()):
        raise ValueError('turn_source ids must be safe identifiers')
    if not is_steward(session=session, project=project) or str(session) == REVIEW_SESSION:
        raise ValueError('turn_source belongs to the Talos conversation only')
    return {'conversation_id': value['conversation_id'], 'message_id': value['message_id']}


def source_note(source):
    """The per-turn note that tells Jim which submitted message he is answering."""
    return ('Submitted message source: ' + json.dumps(source, sort_keys=True, separators=(',', ':'))
            + '. Use this durable client turn ID for provenance and deduplication; composer drafts are not submissions.')


def review_request(value):
    """The `steward_review` request field: exactly {id, content_hash, queue_id}."""
    if not isinstance(value, dict) or set(value) != {'id', 'content_hash', 'queue_id'}:
        raise ValueError('steward_review must be an object with exactly id, content_hash and queue_id')
    id, content_hash, queue_id = value['id'], value['content_hash'], value['queue_id']
    if not isinstance(id, str) or not re.fullmatch(r'final_[0-9a-f]{64}', id):
        raise ValueError('steward_review.id must be final_ followed by 64 lowercase hex characters')
    if not isinstance(content_hash, str) or not _HASH.fullmatch(content_hash):
        raise ValueError('steward_review.content_hash must be 64 lowercase hex characters')
    if queue_id != 'jim_' + id:
        raise ValueError('steward_review.queue_id must be jim_ followed by the review id')
    return {'id': id, 'content_hash': content_hash, 'queue_id': queue_id}


def review_context(value):
    """A bound review context: the gateway's output and the MCP server's input."""
    if not isinstance(value, dict) or set(value) != {'id', 'content_hash', 'queue_id', 'review_run_id', 'source'}:
        raise ValueError('Review context must carry exactly id, content_hash, queue_id, review_run_id and source')
    request = review_request({k: value[k] for k in ('id', 'content_hash', 'queue_id')})
    run_id = value['review_run_id']
    if not isinstance(run_id, str) or not _IDENT.fullmatch(run_id):
        raise ValueError('Review context requires the review run id')
    if value['source'] != {'conversation_id': 'talos', 'message_id': run_id}:
        raise ValueError('Review provenance must be the talos conversation and the review run')
    return {**request, 'review_run_id': run_id, 'source': {'conversation_id': 'talos', 'message_id': run_id}}


def bind_review(request, run_id):
    """Bind a validated request to the gateway's own run id. The gateway is the
    only caller; neither user input nor model output can produce a binding."""
    request = review_request(request)
    if not isinstance(run_id, str) or not _IDENT.fullmatch(run_id):
        raise ValueError('A review binds to a gateway run id')
    return review_context({**request, 'review_run_id': run_id,
                           'source': {'conversation_id': 'talos', 'message_id': run_id}})


def review_from_env(env=None):
    raw = (os.environ if env is None else env).get(REVIEW_ENV)
    if not raw:
        return None
    try:
        return review_context(json.loads(raw))
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise ValueError(f'{REVIEW_ENV} is not a valid review context: {exc}') from None


def mcp_server(review=None):
    """The steward's hermes-tools MCP server definition.

    Will, 2026-10-28: "The main talos agent shouldn't have any guard rails
    except for project deletion." An ordinary steward turn therefore sees the
    whole hermes-tools surface (no HERMES_TOOLS_MCP_ONLY); HERMES_STEWARD only
    keeps the steward identity so StewardTools can apply that one refusal. A
    bound final review still sees exactly one tool.
    """
    env = {'PYTHONPATH':str(Path(__file__).resolve().parents[1]),'HERMES_STEWARD':'1'}
    if review is not None:
        # A review session sees one tool, bound to one final. The binding is
        # canonical JSON so the subprocess validates exactly what the gateway bound.
        env[REVIEW_ENV] = json.dumps(review_context(review), sort_keys=True, separators=(',', ':'))
        env['HERMES_TOOLS_MCP_ONLY'] = 'talos'
    return {'command':sys.executable,'args':['-m','agent.transports.hermes_tools_mcp_server'],'env':env}


class StewardTools:
    """One instance per native turn, enforced in the isolated MCP subprocess.

    Until 2026-10-28 this class fenced the steward: a tool allowlist, one
    dispatch then one acknowledgement per turn, one read per resource, no
    sends, goals or runtime changes, reminders only. Will removed all of it:
    "The main talos agent shouldn't have any guard rails except for project
    deletion, that's the only thing it can't do." An ordinary turn now refuses
    exactly two things: the trusted review actions outside a review invocation
    (they are gateway-bound provenance, not a capability) and project deletion
    (agent.project_deletion). Review mode is unchanged: one tool, one final.
    """
    def __init__(self, review=None):
        self.lock = threading.RLock()
        # Trusted review binding (gateway-set, never model-set). In review mode
        # only the three review actions exist, for the assigned ID, once each.
        self.review = review_context(review) if review is not None else None
        self.record = None             # the Talos record after `review`
        self.content_verified = False  # the complete reference was read and verified
        self.attempted = set()         # review actions whose remote call was attempted

    def run(self, name, args, handler):
        # MCP clients may issue tools concurrently; serialize the review fence
        # and all side effects, including uncertain/error responses.
        with self.lock:
            self.check(name,args)
            if self.review is not None:
                # The generic dispatcher never performs a review: the trusted
                # entry point receives this turn's binding and state directly.
                from tools.talos_tool import review_tool
                handler = lambda name, args: review_tool(self, **args)
            return handler(name,args)

    def check(self, name, args):
        action = (args or {}).get('action')
        if self.review is not None:
            return self.check_review(name, action, args)
        if action in REVIEW_ACTIONS:
            raise ValueError('Final reviews run only inside a trusted internal review invocation.')
        from agent.project_deletion import refuse_tool
        reason = refuse_tool(name, args)
        if reason:
            raise ValueError(reason)

    def check_review(self, name, action, args):
        """Review mode: the assigned final only, each action at most once, nothing else."""
        if 'review_ack' in self.attempted:
            raise ValueError('Acknowledgement attempted. Report its outcome and end this turn.')
        if name != 'talos' or action not in REVIEW_ACTIONS:
            raise ValueError('Internal review mode permits only talos review, review_content and review_ack '
                             'for the assigned final; no dispatch, controls, sends, goal operations, thread reads or other tools.')
        if args.get('id', self.review['id']) != self.review['id']:
            raise ValueError('Only the assigned review ' + self.review['id'] + ' may be read or acknowledged.')
        if action in self.attempted:
            raise ValueError(f'{action} was already called this turn; each review action runs at most once.')
        if action != 'review' and self.record is None:
            raise ValueError('Read the assigned record with review first.')
        if action == 'review_content' and self.record['body']['final']['complete'] is not False:
            raise ValueError('This final is complete in the record; there is no reference to read.')
        for key in ('content_hash', 'queue_id', 'review_run_id', 'source'):
            if key in args and args[key] != self.review[key]:
                raise ValueError(f'{key} is bound by the trusted review context and cannot be changed.')
        if 'final_sha256' in args and args['final_sha256'] != self.record['body']['final']['sha256']:
            raise ValueError('final_sha256 is bound to the assigned record and cannot be changed.')

    def completed(self, name, args, result):
        return  # no per-turn budgets remain; review attempts are recorded by the trusted entry point


PROFILE_LIMIT = 8000  # characters of Will's profile; USER.md is ~2.3 KB today


def instructions():
    """Jim's runtime contract: the steward instructions plus Will's Hermes
    profile, read fresh each turn so the project manager knows who he works
    for and what matters to him. A missing profile only drops that section."""
    text = (Path.home()/'.hermes/talos-steward/docs/steward-instructions.md').read_text()
    hermes_home = Path(os.environ.get('HERMES_HOME') or Path.home()/'.hermes')
    try:
        profile = (hermes_home/'memories'/'USER.md').read_text().strip()
    except OSError:
        profile = ''
    if profile:
        # USER.md separates entries with lone '§' markers.
        entries = [entry.strip() for entry in profile.split('§') if entry.strip()]
        text += ('\n\n## About Will (Hermes profile, read fresh each turn)\n\n'
                 + '\n\n'.join(entries)[:PROFILE_LIMIT])
    return text


def require_steward_runtime(agent):
    if steward_agent(agent) and getattr(agent,'api_mode',None) not in {'codex_app_server','claude_code'}:
        raise RuntimeError('Talos requires a restricted native Codex or Claude runtime; no unrestricted fallback.')

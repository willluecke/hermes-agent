"""Talos is a tool-only orchestrator. This policy is not model advice."""
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
    env = {'PYTHONPATH':str(Path(__file__).resolve().parents[1]),'HERMES_STEWARD':'1',
           'HERMES_TOOLS_MCP_ONLY':'talos,hermes_conversation'}
    if review is not None:
        # A review session sees one tool, bound to one final. The binding is
        # canonical JSON so the subprocess validates exactly what the gateway bound.
        env[REVIEW_ENV] = json.dumps(review_context(review), sort_keys=True, separators=(',', ':'))
        env['HERMES_TOOLS_MCP_ONLY'] = 'talos'
    return {'command':sys.executable,'args':['-m','agent.transports.hermes_tools_mcp_server'],'env':env}


class StewardTools:
    """One instance per native turn, enforced in the isolated MCP subprocess."""
    def __init__(self, review=None):
        self.lock = threading.RLock()
        self.dispatched = False
        self.confirmed = False
        self.reads = set()
        # Trusted review binding (gateway-set, never model-set). In review mode
        # only the three review actions exist, for the assigned ID, once each.
        self.review = review_context(review) if review is not None else None
        self.record = None             # the Talos record after `review`
        self.content_verified = False  # the complete reference was read and verified
        self.attempted = set()         # review actions whose remote call was attempted

    def run(self, name, args, handler):
        # MCP clients may issue tools concurrently; serialize the dispatch fence
        # and all side effects, including uncertain/error responses.
        with self.lock:
            self.check(name,args)
            if self.review is not None:
                # The generic dispatcher never performs a review: the trusted
                # entry point receives this turn's binding and state directly.
                from tools.talos_tool import review_tool
                handler = lambda name, args: review_tool(self, **args)
            try:
                return handler(name,args)
            finally:
                self.completed(name,args,None)

    def check(self, name, args):
        action = args.get('action')
        if self.review is not None:
            return self.check_review(name, action, args)
        if action in REVIEW_ACTIONS:
            raise ValueError('Final reviews run only inside a trusted internal review invocation.')
        if self.dispatched:
            if name!='talos' or action!='delegation' or self.confirmed:
                raise ValueError('Dispatch recorded. Report the acknowledgement/link and end this turn.')
            self.confirmed = True
        if name=='hermes_conversation':
            if action not in {'list','read','goal','set_runtime'}:
                raise ValueError('Steward cannot send extra turns, create goals, or supervise delegates. Use talos dispatch.')
            if action=='goal' and args.get('verb','status')!='status':
                raise ValueError('Only the goal controller manages delegated goals.')
        elif name=='talos':
            if action not in {'state','live','projects','project','mandate','mandate_save','directions','direction','direction_save',
                              'interview_save','dispatch','delegation','delegation_control','hold','sessions'}:
                raise ValueError('This tool action is not available to the steward.')
        else:
            raise ValueError('Steward tools are limited to talos and hermes_conversation.')
        if name=='hermes_conversation' and action=='set_runtime':
            from tools.hermes_conversation_tool import _goal
            current=_goal(args.get('id'),'status')
            goal=current.get('goal',current) if isinstance(current,dict) else {}
            if goal and goal.get('delegation_id') and goal.get('status') in {'active','paused'}:
                raise ValueError('Managed runtime is pinned; ask Will rather than changing live delegated work.')
        if action in {'list','read','goal','state','live','projects','project','mandate','directions','direction','delegation','sessions'}:
            key=(name,action,args.get('id'),args.get('project_id') or args.get('project'))
            if key in self.reads:
                raise ValueError('Already read this state this turn. Answer once; do not poll.')
            self.reads.add(key)

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
        if self.review is not None:
            return  # review attempts are recorded by the trusted entry point itself
        if name=='talos' and args.get('action')=='dispatch':
            # Even an uncertain dispatch response must be reported, never retried
            # by supervising the worker or repairing stores in this turn.
            self.dispatched=True


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

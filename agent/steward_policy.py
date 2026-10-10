"""Talos is a tool-only orchestrator. This policy is not model advice."""
from pathlib import Path
import sys
import threading


def is_steward(*, session='', project='', cwd=''):
    return (str(session) == 'hermes-chat-talos' or str(project) == 'talos-steward'
            or (isinstance(cwd,(str,Path)) and bool(cwd) and Path(cwd).resolve() == Path.home()/'.hermes/talos-steward'))


def steward_agent(agent):
    return is_steward(session=getattr(agent,'session_id',''), project=getattr(agent,'session_project',''), cwd=getattr(agent,'session_cwd',''))


def mcp_server():
    return {'command':sys.executable,'args':['-m','agent.transports.hermes_tools_mcp_server'],
            'env':{'PYTHONPATH':str(Path(__file__).resolve().parents[1]),'HERMES_STEWARD':'1',
                   'HERMES_TOOLS_MCP_ONLY':'talos,hermes_conversation'}}


class StewardTools:
    """One instance per native turn, enforced in the isolated MCP subprocess."""
    def __init__(self):
        self.lock = threading.RLock()
        self.dispatched = False
        self.confirmed = False
        self.reads = set()

    def run(self, name, args, handler):
        # MCP clients may issue tools concurrently; serialize the dispatch fence
        # and all side effects, including uncertain/error responses.
        with self.lock:
            self.check(name,args)
            try:
                return handler(name,args)
            finally:
                self.completed(name,args,None)

    def check(self, name, args):
        action = args.get('action')
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

    def completed(self, name, args, result):
        if name=='talos' and args.get('action')=='dispatch':
            # Even an uncertain dispatch response must be reported, never retried
            # by supervising the worker or repairing stores in this turn.
            self.dispatched=True


def instructions():
    return (Path.home()/'.hermes/talos-steward/docs/steward-instructions.md').read_text()


def require_steward_runtime(agent):
    if steward_agent(agent) and getattr(agent,'api_mode',None) not in {'codex_app_server','claude_code'}:
        raise RuntimeError('Talos requires a restricted native Codex or Claude runtime; no unrestricted fallback.')

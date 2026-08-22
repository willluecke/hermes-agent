"""Exact model contract for the Codex app-server runtime."""

from agent.transports.codex_app_server import CodexAppServerError
from agent.transports.codex_app_server_session import CodexAppServerSession


class _FakeClient:
    def __init__(self, efforts=("high", "xhigh")):
        self.efforts = efforts
        self.calls = []
        self.initialized = False

    def initialize(self, **kwargs):
        self.initialized = True
        return {}

    def request(self, method, params, timeout):
        self.calls.append((method, params))
        if method == "model/list":
            return {
                "data": [
                    {
                        "id": "gpt-5.6-sol",
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": effort} for effort in self.efforts
                        ],
                    }
                ],
                "nextCursor": None,
            }
        if method == "thread/start":
            return {"thread": {"id": "thread-exact"}}
        raise AssertionError(f"unexpected request: {method}")

    def close(self):
        pass


def test_exact_runtime_validates_and_pins_thread_model():
    client = _FakeClient()
    session = CodexAppServerSession(
        model="gpt-5.6-sol",
        effort="xhigh",
        require_exact=True,
        client_factory=lambda **kwargs: client,
    )

    assert session.ensure_started() == "thread-exact"
    assert client.calls == [
        ("model/list", {"includeHidden": True}),
        ("thread/start", {"cwd": session._cwd, "model": "gpt-5.6-sol"}),
    ]


def test_exact_runtime_rejects_unavailable_effort():
    client = _FakeClient(efforts=("high",))
    session = CodexAppServerSession(
        model="gpt-5.6-sol",
        effort="xhigh",
        require_exact=True,
        client_factory=lambda **kwargs: client,
    )

    try:
        session.ensure_started()
    except CodexAppServerError as exc:
        assert "xhigh" in exc.message
        assert "gpt-5.6-sol" in exc.message
    else:
        raise AssertionError("unavailable exact effort should fail closed")

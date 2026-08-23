from tools.environments.base import set_output_callback
from tools.environments.local import LocalEnvironment


def test_foreground_process_output_is_relayed_without_changing_result(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    chunks = []
    set_output_callback(chunks.append)
    try:
        result = env.execute("printf first; sleep 0.05; printf ' second\\n'")
    finally:
        set_output_callback(None)
        env.cleanup()

    assert result["returncode"] == 0
    assert "first second" in result["output"]
    assert "first second" in "".join(chunks)

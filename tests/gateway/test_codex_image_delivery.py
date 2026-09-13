"""Native Codex image events reach regular chat without hand-written MEDIA tags."""

import base64
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agent.codex_runtime import make_codex_app_server_event_bridge
from agent.transports.codex_event_projector import CodexEventProjector
from gateway.platforms.api_server import (
    _promote_current_run_codex_images,
    _resolve_media_to_data_urls,
)
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)
TURN = "turn_image_delivery"


def notification(path, kind="imageGeneration", turn_id=TURN, **extra):
    item = {"type": kind, "id": f"image_{path.name}", "status": "completed"}
    if kind == "imageGeneration":
        # The real schema places the saved path after a potentially long
        # revised prompt and base64 result. Opaque-note truncation lost it.
        item.update(revisedPrompt="Website mockup. " * 500, result="x" * 20_000,
                    savedPath=str(path))
    else:
        item["path"] = str(path)
    item.update(extra)
    return {"method": "item/completed", "params": {"turnId": turn_id, "item": item}}


def project(*notes):
    projector = CodexEventProjector()
    return [message for note in notes for message in projector.project(note).messages]


def deliver(messages, request="Make a homepage mockup.", text="Here it is."):
    return _resolve_media_to_data_urls(_promote_current_run_codex_images(
        text, user_message=request, messages=messages,
        run_started_at=time.time(), codex_turn_id=TURN,
    ))


@pytest.fixture
def image_file(tmp_path):
    image = tmp_path / "homepage.png"
    image.write_bytes(PNG)
    old = time.time() - 3600
    os.utime(image, (old, old))
    return image


def test_generation_keeps_reference_after_long_prompt(image_file):
    projection = CodexEventProjector().project(notification(image_file))
    assert projection.final_text is None
    assert not projection.is_final_answer
    payload = json.loads(projection.messages[0]["content"].split("] ", 1)[1])
    assert payload["path"] == str(image_file)
    assert payload["turnId"] == TURN
    assert "result" not in payload
    assert "revisedPrompt" not in payload
    output = deliver(projection.messages)
    assert f"data:image/png;base64,{base64.b64encode(PNG).decode()}" in output


def test_reposting_old_file_uses_current_view_event(image_file):
    output = deliver(project(notification(image_file, "imageView")),
                     request="Can you post the image here again?")
    assert "data:image/png;base64," in output


@pytest.mark.parametrize("kind", ["imageView", "imageGeneration"])
def test_prior_turn_images_are_never_automatically_reposted(image_file, kind):
    output = deliver(project(notification(image_file, kind, turn_id="old_turn")),
                     request="Show me the image.")
    assert output == "Here it is."


def test_inspection_and_reattached_reference_are_not_delivery_requests(image_file):
    output = deliver(project(notification(image_file, "imageView")), request=(
        "Fix the layout.\n(Re-attached for reference — image the user shared: show image.png.)"
    ))
    assert output == "Here it is."


@pytest.mark.parametrize("extra", [{"status": "failed"}, {"failure": {"message": "quota"}},
                                   {"savedPath": None}])
def test_unsuccessful_generation_does_not_attach_a_file(image_file, extra):
    assert deliver(project(notification(image_file, **extra))) == "Here it is."


def test_generated_images_are_deduplicated_and_preferred_to_inspected_references(image_file, tmp_path):
    second = tmp_path / "mobile.png"
    second.write_bytes(PNG)
    reference = tmp_path / "reference.png"
    reference.write_bytes(PNG)
    messages = project(notification(image_file), notification(image_file),
                       notification(second), notification(reference, "imageView"))
    output = _promote_current_run_codex_images(
        "Two mockups.", user_message="Show me both mockups as images.", messages=messages,
        run_started_at=time.time(), codex_turn_id=TURN,
    )
    assert output.count("MEDIA:") == 2
    assert str(image_file) in output and str(second) in output
    assert str(reference) not in output


@pytest.mark.parametrize("carrier", ["MEDIA:{path}", "![Mockup](<{path}>)"])
def test_explicit_delivery_prevents_duplicate_image(image_file, carrier):
    output = deliver(project(notification(image_file)), text=carrier.format(path=image_file))
    assert output.count("data:image/png;base64,") == 1


@pytest.mark.parametrize("invalid", ["missing", "non_image", "oversize", "denied"])
def test_automatic_delivery_keeps_existing_media_bounds(image_file, tmp_path, invalid):
    path = image_file
    if invalid == "missing":
        path = tmp_path / "missing.png"
    elif invalid == "non_image":
        path = tmp_path / "notes.txt"
        path.write_bytes(PNG)
    elif invalid == "oversize":
        path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    else:
        path = tmp_path / "secret.png"
        path.symlink_to("/etc/passwd")
    assert deliver(project(notification(path))) == "Here it is."


@pytest.mark.parametrize("kind,tool", [("imageGeneration", "image_generate"), ("imageView", "view_image")])
def test_native_image_progress_has_stable_identity_without_base64(image_file, kind, tool):
    callback = MagicMock()
    bridge = make_codex_app_server_event_bridge(SimpleNamespace(tool_progress_callback=callback))
    note = notification(image_file, kind)
    bridge({**note, "method": "item/started"})
    bridge(note)
    started, completed = callback.call_args_list
    assert started.args[0] == "tool.started"
    assert completed.args[0] == "tool.completed"
    assert started.args[1] == completed.args[1] == tool
    assert started.kwargs["tool_call_id"] == completed.kwargs["tool_call_id"]
    assert str(image_file) in completed.kwargs["result"]
    assert len(completed.kwargs["result"]) < 1000


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,partial", [("imageGeneration", False), ("imageView", False),
                                        ("imageGeneration", True)])
async def test_image_reaches_regular_run_stream_status_and_replay(image_file, kind, partial):
    adapter = _make_adapter()
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

    def run_conversation(**_kwargs):
        result = {
            "final_response": "Here is the mockup.",
            "messages": project(notification(image_file, kind)),
            "codex_turn_id": TURN,
        }
        if partial:
            result.update(partial=True, error="Turn interrupted after image generation")
        return result

    agent.run_conversation.side_effect = run_conversation
    async with TestClient(TestServer(_create_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post("/v1/runs", json={"input": "Show me the mockup image."})
            run_id = (await response.json())["run_id"]
            stream = await (await client.get(f"/v1/runs/{run_id}/events")).text()
            replay = await (await client.get(f"/v1/runs/{run_id}/events")).text()
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
    expected = f"![image](data:image/png;base64,{base64.b64encode(PNG).decode()})"
    for output in (stream, replay, status["output"]):
        assert expected in output
        assert str(image_file) not in output
    assert status["status"] == ("failed" if partial else "completed")

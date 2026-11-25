"""file:// image parts: accepted only under the configured roots, as real image files."""

from __future__ import annotations

import os

import pytest

from agent import local_images
from agent.local_images import MAX_LOCAL_IMAGE_BYTES, local_image_path, local_image_roots

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


@pytest.fixture
def roots(tmp_path):
    root = tmp_path / "conversation-images"
    root.mkdir()
    return {"agent": {"local_image_roots": [str(root)]}}, root


def test_default_root_is_the_sync_stores_image_directory():
    assert local_image_roots({}) == [os.path.realpath(os.path.expanduser("~/.hermes/conversation-images"))]
    assert local_image_roots({"agent": {"local_image_roots": "~/pics"}}) == [os.path.realpath(os.path.expanduser("~/pics"))]


def test_a_png_under_a_root_resolves_to_its_real_path(roots):
    config, root = roots
    (root / "c1").mkdir()
    (root / "c1" / "shot.png").write_bytes(PNG)
    assert local_image_path(f"file://{root}/c1/shot.png", config) == str(root / "c1" / "shot.png")
    assert local_image_path(f"file://{root}/c1/shot.png".replace(" ", "%20"), config) == str(root / "c1" / "shot.png")


def test_paths_outside_the_roots_are_refused_even_through_a_symlink(roots, tmp_path):
    config, root = roots
    outside = tmp_path / "secret.png"
    outside.write_bytes(PNG)
    with pytest.raises(ValueError, match="outside the configured image roots"):
        local_image_path(f"file://{outside}", config)
    (root / "link.png").symlink_to(outside)
    with pytest.raises(ValueError, match="outside the configured image roots"):
        local_image_path(f"file://{root}/link.png", config)
    with pytest.raises(ValueError, match="outside the configured image roots"):
        local_image_path(f"file://{root}/../secret.png", config)


def test_only_real_image_files_of_bounded_size_pass(roots):
    config, root = roots
    (root / "text.png").write_bytes(b"not a png at all" + b"\x00" * 32)
    with pytest.raises(ValueError, match="bytes match its suffix"):
        local_image_path(f"file://{root}/text.png", config)
    (root / "notes.txt").write_bytes(PNG)
    with pytest.raises(ValueError, match="bytes match its suffix"):
        local_image_path(f"file://{root}/notes.txt", config)
    (root / "photo.jpg").write_bytes(JPEG)
    assert local_image_path(f"file://{root}/photo.jpg", config).endswith("photo.jpg")
    with pytest.raises(ValueError, match="does not exist"):
        local_image_path(f"file://{root}/missing.png", config)
    big = root / "big.png"
    big.write_bytes(PNG)
    os.truncate(big, MAX_LOCAL_IMAGE_BYTES + 1)
    with pytest.raises(ValueError, match="between 1 byte and"):
        local_image_path(f"file://{root}/big.png", config)


def test_other_schemes_relative_paths_and_no_roots_are_refused(roots):
    config, root = roots
    with pytest.raises(ValueError, match="file:// URL"):
        local_image_path("https://example.com/a.png", config)
    with pytest.raises(ValueError, match="file:// URL"):
        local_image_path("file://otherhost/a.png", config)
    with pytest.raises(ValueError, match="no local image roots"):
        local_image_path(f"file://{root}/a.png", {"agent": {"local_image_roots": []}})


def test_the_default_config_carries_the_key():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["local_image_roots"] == ["~/.hermes/conversation-images"]
    assert local_images.CONFIG_KEY == "local_image_roots"


# ---------------------------------------------------------------------------
# inline_local_images: what a remote provider receives
# ---------------------------------------------------------------------------
#
# 2026-11-25: OpenRouter answered HTTP 400 "Invalid URL format: file:" for a
# RegWatch screenshot, and every later turn re-attached it and failed too.


def _shot(root, name="IMG_0212.png", data=PNG):
    (root / "c1").mkdir(exist_ok=True)
    path = root / "c1" / name
    path.write_bytes(data)
    return path


def test_a_file_image_becomes_a_data_url_and_the_history_keeps_its_path(roots):
    import base64

    config, root = roots
    path = _shot(root)
    url = f"file://{path}"
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "fix the map"},
            {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
        ]},
    ]
    out = local_images.inline_local_images(history, config)
    sent = out[0]["content"][1]
    assert sent["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert sent["image_url"]["detail"] == "high"
    assert out[0]["content"][0] == {"type": "text", "text": "fix the map"}
    # the stored history is untouched
    assert history[0]["content"][1]["image_url"]["url"] == url


def test_responses_style_string_image_url_is_inlined_too(roots):
    config, root = roots
    path = _shot(root, "photo.jpg", JPEG)
    out = local_images.inline_local_images(
        [{"role": "user", "content": [{"type": "input_image", "image_url": f"file://{path}"}]}], config,
    )
    assert out[0]["content"][0]["image_url"].startswith("data:image/jpeg;base64,")


def test_an_unreadable_file_image_becomes_a_text_note_not_a_400(roots, tmp_path):
    config, root = roots
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(PNG)
    out = local_images.inline_local_images([
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"file://{root}/c1/gone.png"}},
            {"type": "image_url", "image_url": {"url": f"file://{outside}"}},
        ]},
    ], config)
    parts = out[0]["content"]
    assert [p["type"] for p in parts] == ["text", "text"]
    assert "gone.png" in parts[0]["text"] and "does not exist" in parts[0]["text"]
    assert "outside the configured image roots" in parts[1]["text"]


def test_remote_and_data_urls_pass_through_and_no_file_image_means_no_copy(roots):
    config, _root = roots
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
        {"role": "assistant", "content": "ok"},
    ]
    assert local_images.inline_local_images(messages, config) is messages


def test_only_messages_with_file_images_are_copied(roots):
    config, root = roots
    path = _shot(root)
    plain = {"role": "assistant", "content": "earlier reply"}
    with_image = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{path}"}}]}
    out = local_images.inline_local_images([plain, with_image], config)
    assert out[0] is plain
    assert out[1] is not with_image

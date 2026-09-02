"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.
"""

import base64
import json
import os
import time
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import (  # noqa: E402
    _promote_current_run_codex_image_view,
    _resolve_media_to_data_urls,
)

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


class TestResolveMediaToDataUrls(unittest.TestCase):
    def _write_png(self, tmpdir_name="hermes_media_test"):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix=tmpdir_name))
        p = d / "shot.png"
        p.write_bytes(_PNG_BYTES)
        return p

    def test_media_tag_inlined(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"Here you go: MEDIA:{p}")
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)

    def test_backtick_wrapped_tag(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"See `MEDIA:{p}` above")
        self.assertIn("data:image/png;base64,", out)

    def test_local_markdown_link_inlined_as_image(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"[Shop owner controls]({p})")
        self.assertIn("![Shop owner controls](data:image/png;base64,", out)
        self.assertNotIn(str(p), out)

    def test_angle_wrapped_local_markdown_image_inlined(self):
        p = self._write_png("hermes media test ")
        out = _resolve_media_to_data_urls(f"![Owner view](<{p}>)")
        self.assertIn("![Owner view](data:image/png;base64,", out)
        self.assertNotIn(str(p), out)

    def test_missing_file_left_untouched(self):
        text = "MEDIA:/nonexistent/path/shot.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_non_image_left_untouched(self):
        text = "MEDIA:/tmp/archive.zip"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_non_image_markdown_link_left_untouched(self):
        text = "[Download](/tmp/archive.zip)"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    @staticmethod
    def _image_view_message(path):
        payload = {"type": "imageView", "id": "view_1", "path": str(path)}
        return {
            "role": "assistant",
            "content": f"[codex imageView] {json.dumps(payload)}",
        }

    def test_requested_current_run_codex_image_view_is_promoted(self):
        started_at = time.time()
        p = self._write_png()

        promoted = _promote_current_run_codex_image_view(
            "This is the catalog-only view.",
            user_message="Can you show me his catalog lower on the screen?",
            messages=[self._image_view_message(p)],
            run_started_at=started_at,
        )
        out = _resolve_media_to_data_urls(promoted)

        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn(str(p), out)

    def test_codex_image_view_is_not_promoted_without_delivery_request(self):
        started_at = time.time()
        p = self._write_png()
        text = "The visual regression test passed."

        promoted = _promote_current_run_codex_image_view(
            text,
            user_message="Fix the responsive layout regression.",
            messages=[self._image_view_message(p)],
            run_started_at=started_at,
        )

        self.assertEqual(promoted, text)

    def test_stale_codex_image_view_is_not_promoted(self):
        p = self._write_png()
        old = time.time() - 60
        os.utime(p, (old, old))
        text = "This is the requested screenshot."

        promoted = _promote_current_run_codex_image_view(
            text,
            user_message="Please show me the screenshot.",
            messages=[self._image_view_message(p)],
            run_started_at=time.time(),
        )

        self.assertEqual(promoted, text)

    def test_existing_remote_image_prevents_duplicate_promotion(self):
        started_at = time.time()
        p = self._write_png()
        text = "![Catalog](https://example.com/catalog.png)"

        promoted = _promote_current_run_codex_image_view(
            text,
            user_message="Please show me the catalog screenshot.",
            messages=[self._image_view_message(p)],
            run_started_at=started_at,
        )

        self.assertEqual(promoted, text)


if __name__ == "__main__":
    unittest.main()

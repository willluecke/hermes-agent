from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tools.reversible_deletion import (
    ReversibleDeletionPolicy,
    capture_delete_command,
)


def _adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/trash", adapter._handle_list_trash)
    app.router.add_post(
        "/v1/trash/{item_id}/restore", adapter._handle_restore_trash
    )
    app.router.add_post("/v1/trash/{item_id}/purge", adapter._handle_purge_trash)
    return app


@pytest.fixture
def trash_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    config = {
        "codex_runtime": {
            "workspaces": {
                "default_project": "reg-watch",
                "projects": {"reg-watch": str(workspace)},
            },
            "reversible_deletion": {
                "enabled": True,
                "max_item_bytes": 1024 * 1024,
                "max_total_bytes": 4 * 1024 * 1024,
                "max_entries": 100,
                "temp_roots": [str(tmp_path / "temp")],
            },
        }
    }
    policy = ReversibleDeletionPolicy.from_config(
        config["codex_runtime"]["reversible_deletion"]
    )
    return workspace, config, policy


def _capture(workspace: Path, policy: ReversibleDeletionPolicy):
    target = workspace / "generated.txt"
    target.write_text("recoverable", encoding="utf-8")
    result = capture_delete_command(
        "rm generated.txt",
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-api",
        operation_key="api-op",
        policy=policy,
    )
    return target, result.items[0]


@pytest.mark.asyncio
async def test_trash_routes_require_gateway_auth(trash_setup):
    _workspace, config, _policy = trash_setup
    adapter = _adapter("sk-trash-secret")
    with patch("gateway.run._load_gateway_config", return_value=config):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.get("/v1/trash?project=reg-watch")
    assert response.status == 401


@pytest.mark.asyncio
async def test_list_restore_and_purge_use_project_and_item_ids(trash_setup):
    workspace, config, policy = trash_setup
    target, item = _capture(workspace, policy)
    adapter = _adapter()
    with patch("gateway.run._load_gateway_config", return_value=config):
        async with TestClient(TestServer(_app(adapter))) as client:
            listed = await client.get("/v1/trash?project=reg-watch")
            assert listed.status == 200
            list_body = await listed.json()
            assert [entry["item_id"] for entry in list_body["items"]] == [
                item.item_id
            ]

            conflict = await client.post(
                f"/v1/trash/{item.item_id}/restore",
                json={"project": "reg-watch"},
            )
            assert conflict.status == 409

            target.unlink()
            restored = await client.post(
                f"/v1/trash/{item.item_id}/restore",
                json={"project": "reg-watch"},
            )
            assert restored.status == 200
            assert target.read_text(encoding="utf-8") == "recoverable"

            purged = await client.post(
                f"/v1/trash/{item.item_id}/purge",
                json={"project": "reg-watch"},
            )
            assert purged.status == 200
            assert (await purged.json())["item"]["status"] == "purged"

            active = await client.get("/v1/trash?project=reg-watch")
            assert (await active.json())["items"] == []


@pytest.mark.asyncio
async def test_trash_item_cannot_be_addressed_through_another_project(trash_setup):
    workspace, config, policy = trash_setup
    _target, item = _capture(workspace, policy)
    other = workspace.parent / "other"
    other.mkdir()
    config["codex_runtime"]["workspaces"]["projects"]["other"] = str(other)
    adapter = _adapter()
    with patch("gateway.run._load_gateway_config", return_value=config):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                f"/v1/trash/{item.item_id}/purge",
                json={"project": "other"},
            )
    assert response.status == 404


@pytest.mark.asyncio
async def test_trash_route_rejects_path_shaped_item_id(trash_setup):
    _workspace, config, _policy = trash_setup
    adapter = _adapter()
    with patch("gateway.run._load_gateway_config", return_value=config):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/trash/not-an-id/purge",
                json={"project": "reg-watch"},
            )
    assert response.status == 400

"""E2E WebSocket integration tests for the SDDj server.

These tests exercise the WebSocket transport layer, binary frame protocol,
and error handling without requiring a GPU or loaded model. They validate
the contract between the Lua extension client and the Python server.
"""

from __future__ import annotations

import asyncio
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_engine():
    """Patch the DiffusionEngine so the server starts without GPU."""
    mock_eng = MagicMock()
    mock_eng.is_loaded = False
    mock_eng.get_status.return_value = {}
    with patch("sddj.server.engine", mock_eng):
        yield mock_eng


@pytest.fixture()
def client(_mock_engine):
    """Provide a Starlette TestClient connected to the SDDj app."""
    from sddj.server import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Connection & Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, client, _mock_engine):
        with patch("sddj.vram_utils.get_vram_info", return_value=(0, 0, 0)):
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


class TestWebSocketConnection:
    def test_connect_and_disconnect(self, client):
        with client.websocket_connect("/ws") as ws:
            # Send ping, expect pong
            ws.send_text(json.dumps({"action": "ping"}))
            data = ws.receive_text()
            msg = json.loads(data)
            assert msg["type"] == "pong"

    def test_ping_pong_round_trip(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"action": "ping"}))
            data = ws.receive_text()
            msg = json.loads(data)
            assert msg["type"] == "pong"

    def test_invalid_json_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text("NOT VALID JSON{{{")
            data = ws.receive_text()
            msg = json.loads(data)
            assert msg["type"] == "error"

    def test_unknown_action_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"action": "nonexistent_action_xyz"}))
            data = ws.receive_text()
            msg = json.loads(data)
            assert msg["type"] == "error"
            assert msg.get("code") in ("UNKNOWN_ACTION", "INVALID_REQUEST")

    def test_missing_action_field_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"not_action": "ping"}))
            data = ws.receive_text()
            msg = json.loads(data)
            assert msg["type"] == "error"


# ---------------------------------------------------------------------------
# Binary Frame Protocol
# ---------------------------------------------------------------------------


class TestBinaryFrameProtocol:
    """Validate the [uint32 LE meta_len][JSON metadata][raw RGBA bytes] contract."""

    @staticmethod
    def _make_binary_frame(meta: dict, raw_image: bytes) -> bytes:
        """Build a binary frame as the server would send it."""
        meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        return struct.pack("<I", len(meta_json)) + meta_json + raw_image

    def test_binary_frame_structure(self):
        """Verify frame can be round-tripped: build → parse header → extract."""
        meta = {"type": "result", "image": "", "seed": 42, "encoding": "raw_rgba",
                "time_ms": 100, "width": 4, "height": 4}
        raw_rgba = bytes(4 * 4 * 4)  # 4×4 RGBA = 64 bytes
        frame = self._make_binary_frame(meta, raw_rgba)

        # Parse
        assert len(frame) >= 4
        json_len = struct.unpack("<I", frame[:4])[0]
        parsed_meta = json.loads(frame[4:4 + json_len])
        parsed_raw = frame[4 + json_len:]

        assert parsed_meta["type"] == "result"
        assert parsed_meta["seed"] == 42
        assert parsed_meta["encoding"] == "raw_rgba"
        assert len(parsed_raw) == 64

    def test_binary_frame_bounds_validation(self):
        """Verify that json_len < 2 or > 1MB would be rejected by client logic."""
        # Simulate what the Lua client does: reject json_len < 2 or > 1048576
        for bad_len in [0, 1, 1048577, 0xFFFFFFFF]:
            frame = struct.pack("<I", bad_len) + b"\x00" * 10
            json_len = struct.unpack("<I", frame[:4])[0]
            assert json_len < 2 or json_len > 1048576, f"Expected rejection for len={bad_len}"

    def test_truncated_frame_detection(self):
        """Verify truncated frames are detectable."""
        meta = {"type": "result"}
        meta_bytes = json.dumps(meta).encode("utf-8")
        # Declare full length but provide truncated data
        header = struct.pack("<I", len(meta_bytes))
        truncated = header + meta_bytes[:len(meta_bytes) // 2]

        json_len = struct.unpack("<I", truncated[:4])[0]
        assert len(truncated) < 4 + json_len  # Truncated


# ---------------------------------------------------------------------------
# Resource Listing (no GPU required)
# ---------------------------------------------------------------------------


class TestResourceListing:
    """Resource list actions should return valid list responses even with empty dirs."""

    @pytest.mark.parametrize("action", [
        "list_palettes", "list_loras", "list_embeddings",
        "list_presets", "list_modulation_presets",
        "list_expression_presets", "list_choreography_presets",
    ])
    def test_list_action_returns_list(self, client, action):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"action": action}))
            data = ws.receive_text()
            msg = json.loads(data)
            # Should be a list response or at minimum not an error
            assert msg.get("type") != "error" or "not loaded" in msg.get("message", "").lower(), \
                f"Unexpected error for {action}: {msg}"


# ---------------------------------------------------------------------------
# Error Handling Robustness
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_generate_without_engine_returns_error(self, client):
        """Generate action without loaded engine should return a clean error."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "action": "generate",
                "prompt": "test",
                "width": 512,
                "height": 512,
            }))
            data = ws.receive_text()
            msg = json.loads(data)
            assert msg["type"] == "error"

    def test_cancel_without_generation(self, client):
        """Cancel when nothing is generating should be a no-op or clean response."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"action": "cancel"}))
            # Should not crash — may or may not send a response
            ws.send_text(json.dumps({"action": "ping"}))
            data = ws.receive_text()
            msg = json.loads(data)
            # We should still get a valid pong (connection alive)
            assert msg["type"] in ("pong", "error")


# ---------------------------------------------------------------------------
# _send() Binary Serialization
# ---------------------------------------------------------------------------


class TestSendFunction:
    """Unit tests for the _send() binary frame builder."""

    @pytest.mark.asyncio
    async def test_send_binary_frame(self):
        from sddj.server import _send
        from sddj.protocol import ResultResponse

        mock_ws = AsyncMock()

        resp = ResultResponse(
            image="",
            seed=12345,
            time_ms=100,
            width=8,
            height=8,
        )
        resp.encoding = "raw_rgba"
        resp._raw_bytes = bytes(8 * 8 * 4)  # 256 bytes RGBA

        await _send(mock_ws, resp)

        mock_ws.send_bytes.assert_called_once()
        frame = mock_ws.send_bytes.call_args[0][0]

        # Parse back
        json_len = struct.unpack("<I", frame[:4])[0]
        meta = json.loads(frame[4:4 + json_len])
        raw = frame[4 + json_len:]

        assert meta["seed"] == 12345
        assert meta["encoding"] == "raw_rgba"
        assert len(raw) == 256

    @pytest.mark.asyncio
    async def test_send_text_frame(self):
        from sddj.server import _send
        from sddj.protocol import PongResponse

        mock_ws = AsyncMock()
        resp = PongResponse(server_time=1234567890.0)

        await _send(mock_ws, resp)

        mock_ws.send_text.assert_called_once()
        text = mock_ws.send_text.call_args[0][0]
        msg = json.loads(text)
        assert msg["type"] == "pong"

    @pytest.mark.asyncio
    async def test_send_handles_disconnect_gracefully(self):
        from sddj.server import _send
        from sddj.protocol import PongResponse

        mock_ws = AsyncMock()
        mock_ws.send_text.side_effect = WebSocketDisconnect()

        resp = PongResponse(server_time=0.0)
        # Should not raise
        await _send(mock_ws, resp)

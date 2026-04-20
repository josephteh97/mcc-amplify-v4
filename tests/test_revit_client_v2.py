"""
Phase 1 — Revit Add-in v2 wire-format tests (Python side).

Covers the two schema transitions:
  • X-Revit-Warnings header: v1 string list → v2 object list (with element IDs)
  • /session/{id}/query-elements: new endpoint shape expected from the Add-in

The C# endpoint itself is verified manually on a Windows host running Revit
2023; these tests lock down the Python client contract so rolling deploys of
the Add-in don't silently break Ubuntu parsing.
"""
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from backend.services.revit_client import RevitClient, _parse_warning_header


class TestParseWarningHeader:
    def test_v1_string_list(self):
        headers = {"x-revit-warnings": '["warn A", "warn B"]'}
        texts, details = _parse_warning_header(headers)
        assert texts == ["warn A", "warn B"]
        assert details == [
            {"text": "warn A", "element_ids": []},
            {"text": "warn B", "element_ids": []},
        ]

    def test_v1_explicit_version_header(self):
        headers = {
            "x-revit-warnings": '["single"]',
            "x-revit-warnings-version": "1",
        }
        texts, details = _parse_warning_header(headers)
        assert texts == ["single"]
        assert details == [{"text": "single", "element_ids": []}]

    def test_v2_object_list(self):
        headers = {
            "x-revit-warnings": (
                '[{"text":"join fail","element_ids":[42,99]},'
                ' {"text":"off axis","element_ids":[7]}]'
            ),
            "x-revit-warnings-version": "2",
        }
        texts, details = _parse_warning_header(headers)
        assert texts == ["join fail", "off axis"]
        assert details == [
            {"text": "join fail", "element_ids": [42, 99]},
            {"text": "off axis", "element_ids": [7]},
        ]

    def test_v2_missing_element_ids_defaults_empty(self):
        headers = {
            "x-revit-warnings": '[{"text":"no ids here"}]',
            "x-revit-warnings-version": "2",
        }
        texts, details = _parse_warning_header(headers)
        assert texts == ["no ids here"]
        assert details == [{"text": "no ids here", "element_ids": []}]

    def test_v2_with_string_element_gracefully_degrades(self):
        # Malformed v2 payload (strings mixed in) — preserve text, empty IDs.
        headers = {
            "x-revit-warnings": '["legacy string", {"text":"object","element_ids":[1]}]',
            "x-revit-warnings-version": "2",
        }
        texts, details = _parse_warning_header(headers)
        assert texts == ["legacy string", "object"]
        assert details[0]["element_ids"] == []
        assert details[1]["element_ids"] == [1]

    def test_empty_header(self):
        texts, details = _parse_warning_header({"x-revit-warnings": "[]"})
        assert texts == []
        assert details == []

    def test_missing_header(self):
        texts, details = _parse_warning_header({})
        assert texts == []
        assert details == []

    def test_malformed_json_returns_empty(self):
        headers = {"x-revit-warnings": "not-json{"}
        texts, details = _parse_warning_header(headers)
        assert texts == []
        assert details == []

    def test_non_list_payload_returns_empty(self):
        headers = {"x-revit-warnings": '{"not":"a list"}'}
        texts, details = _parse_warning_header(headers)
        assert texts == []
        assert details == []


class TestBuildModelHeaderCapture:
    @pytest.mark.asyncio
    async def test_v2_stores_element_ids_on_client(self, tmp_path):
        """_build_via_http populates last_warning_details with element IDs."""
        # Minimal valid RVT magic so content validation passes
        fake_rvt = b"\xd0\xcf\x11\xe0" + b"\x00" * 1024
        tx_path = tmp_path / "tx.json"
        tx_path.write_text("{}")

        client = RevitClient()

        response = MagicMock()
        response.status_code = 200
        response.content = fake_rvt
        response.headers = {
            "x-revit-warnings": (
                '[{"text":"beam off axis","element_ids":[1001,1002]}]'
            ),
            "x-revit-warnings-version": "2",
        }

        mock_async_client = MagicMock()
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None
        mock_async_client.post = AsyncMock(return_value=response)

        with patch("backend.services.revit_client.httpx.AsyncClient",
                   return_value=mock_async_client), \
             patch("backend.services.revit_client.Path") as mock_path:
            # Stub filesystem I/O — we don't want to write /data/models/rvt/…
            mock_path.return_value.parent.mkdir = MagicMock()
            with patch("builtins.open", MagicMock()):
                rvt_path, warnings = await client._build_via_http(
                    str(tx_path), "job123", "test.pdf",
                )

        assert warnings == ["beam off axis"]
        assert client.last_warning_details == [
            {"text": "beam off axis", "element_ids": [1001, 1002]}
        ]

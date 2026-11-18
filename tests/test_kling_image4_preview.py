"""Offline tests of experimental Image 4.0 REST request construction (no paid calls)."""
import json

import numpy as np
import pytest
import torch

import kling_image4_preview as m


def test_disabled_preview_does_not_call_api(monkeypatch):
    monkeypatch.setattr(m, "_client_from_auth", lambda auth: pytest.fail("API was called"))
    with pytest.raises(RuntimeError, match="EXPERIMENTAL"):
        m.KlingImage4Preview_TextToImage().generate({}, "test", "2k", "auto", 1, False)


def test_text_to_image_uses_vendor_native_fields(monkeypatch):
    outgoing = {}
    monkeypatch.setattr(m, "_run", lambda auth, body, enabled, path: outgoing.update(body) or ("ok",))
    m.KlingImage4Preview_TextToImage().generate({}, "product photo", "8k", "3:4", 2, True)
    assert outgoing == {"result_type":"single", "prompt":"product photo",
                        "resolution":"8k", "aspect_ratio":"3:4", "n":2}


def test_image_edit_multiple_references(monkeypatch):
    outgoing = {}
    monkeypatch.setattr(m, "_run", lambda auth, body, enabled, path: outgoing.update(body) or ("ok",))
    img = torch.ones((1, 320, 320, 3), dtype=torch.float32)
    m.KlingImage4Preview_ImageEdit().generate({}, "change @image_1", img, "2k", "auto", 1,
                                               True, reference_2=img)
    assert [x["id"] for x in outgoing["image_list"]] == ["image_1", "image_2"]
    assert all(x["image_url"] for x in outgoing["image_list"])


def test_series_and_long_image_modes(monkeypatch):
    bodies = []
    monkeypatch.setattr(m, "_run", lambda auth, body, enabled, path: bodies.append(body) or ("ok",))
    m.KlingImage4Preview_Series().generate({}, "outfit series", "4k", "16:9", "10", True)
    m.KlingImage4Preview_LongImage().generate({}, "product landing page", "ecommerce",
                                              "long", "2k", True)
    assert bodies[0]["series_amount"] == 10
    assert bodies[1]["result_type"] == "long_image"
    assert bodies[1]["skill"] == "ecommerce"


def test_new_and_legacy_image_response_urls():
    assert m._urls({"outputs":[{"type":"image","url":"https://cdn.example.com/one.png"}]}) == [
        "https://cdn.example.com/one.png"
    ]
    assert m._urls({"task_result":{"images":[{"url":"https://cdn.example.com/a.png"}]}}) == [
        "https://cdn.example.com/a.png"
    ]


def test_request_mode_requires_explicit_opt_in():
    assert m._preview_enabled(False) is False
    assert m._preview_enabled(True) is True  # backwards-compatible old workflows
    assert m._preview_enabled(m.PREVIEW_REQUEST_MODES[0]) is False
    assert m._preview_enabled(m.PREVIEW_REQUEST_MODES[1]) is True
    assert m._preview_enabled("anything else") is False


def test_disabled_combo_blocks_api_calls(monkeypatch):
    monkeypatch.setattr(m, "_client_from_auth", lambda auth: pytest.fail("API called"))
    with pytest.raises(RuntimeError, match="EXPERIMENTAL"):
        m.KlingImage4Preview_TextToImage().generate(
            {}, "fashion studio", "2k", "3:4", 1, m.PREVIEW_REQUEST_MODES[0]
        )

"""Offline tests of experimental Image 4.0 REST request construction (no paid calls)."""
import json

import numpy as np
import pytest
import torch

import kling_image4_preview as m


def test_disabled_preview_does_not_call_api(monkeypatch):
    monkeypatch.setattr(m, "_client_from_auth", lambda auth: pytest.fail("API was called"))
    with pytest.raises(RuntimeError, match="KLING IMAGE 4.0 REST"):
        m.KlingImage4Preview_TextToImage().generate({}, "test", "2k", "auto", 1, False)


def test_text_to_image_uses_vendor_native_fields(monkeypatch):
    outgoing = {}
    monkeypatch.setattr(m, "_run", lambda auth, body, enabled, path: outgoing.update(body) or ("ok",))
    m.KlingImage4Preview_TextToImage().generate({}, "product photo", "8k", "3:4", 2, True)
    assert outgoing == {
        "contents": [{"type": "prompt", "text": "product photo"}],
        "settings": {"result_type":"single", "resolution":"8k", "aspect_ratio":"3:4", "n":2},
    }


def test_image_edit_multiple_references(monkeypatch):
    outgoing = {}
    monkeypatch.setattr(m, "_run", lambda auth, body, enabled, path: outgoing.update(body) or ("ok",))
    img = torch.ones((1, 320, 320, 3), dtype=torch.float32)
    m.KlingImage4Preview_ImageEdit().generate({}, "change @image_1", img, "2k", "auto", 1,
                                               True, reference_2=img)
    assert [x["id"] for x in outgoing["contents"][1:]] == ["image_1", "image_2"]
    assert all(x["url"] for x in outgoing["contents"][1:])
    assert all(x["type"] == "refer_image" for x in outgoing["contents"][1:])


def test_series_and_long_image_modes(monkeypatch):
    bodies = []
    monkeypatch.setattr(m, "_run", lambda auth, body, enabled, path: bodies.append(body) or (torch.ones(1,60,5,3), "urls", "id", "metadata"))
    m.KlingImage4Preview_Series().generate({}, "outfit series", "4k", "16:9", "10", True)
    m.KlingImage4Preview_LongImage().generate({}, "product landing page", "webpage",
                                              "long", "2k", True)
    assert bodies[0]["settings"]["series_amount"] == 10
    assert bodies[1]["settings"] == {"skill": "webpage", "resolution": "2k"}


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
    with pytest.raises(RuntimeError, match="KLING IMAGE 4.0 REST"):
        m.KlingImage4Preview_TextToImage().generate(
            {}, "fashion studio", "2k", "3:4", 1, m.PREVIEW_REQUEST_MODES[0]
        )


def test_confirmed_routes_are_used(monkeypatch):
    calls = []
    monkeypatch.setattr(m, "_run", lambda auth, body, enabled, path: calls.append((body, path)) or (torch.ones(1,60,5,3), "urls", "id", "metadata"))
    m.KlingImage4Preview_TextToImage().generate({}, "test photo", "2k", "1:1", 1, True)
    assert calls[-1][1] == "/image-generation/kling-4.0"
    img = torch.ones((1, 320, 320, 3))
    m.KlingImage4Preview_LongImage().generate(
        {}, "Product detail page for @image_1", "ecommerce", "short", "1k", True, reference_1=img)
    assert calls[-1][1] == "/long-image-generation/kling-4.0"
    assert calls[-1][0]["settings"] == {"skill":"ecommerce", "resolution":"1k", "length":"short"}


def test_ecommerce_requires_product_reference(monkeypatch):
    monkeypatch.setattr(m, "_run", lambda *args: pytest.fail("API request made"))
    with pytest.raises(ValueError, match="requires 1-3"):
        m.KlingImage4Preview_LongImage().generate({}, "page", "ecommerce", "short", "2k", True)


def test_native_task_polling_single_submission(monkeypatch):
    calls = []
    class Client:
        def _request(self, method, path, *, body, timeout):
            calls.append((method, path, body))
            return {"code":0, "data":{"id":"task_123"}}
        def poll_new_task(self, task_id):
            assert task_id == "task_123"
            return {"id":task_id,"status":"succeeded",
                    "outputs":[{"type":"image","url":"https://cdn.example.com/test.png"}]}
    monkeypatch.setattr(m, "_client_from_auth", lambda auth: Client())
    monkeypatch.setattr(m, "download_to_tensor", lambda url: torch.ones((1,16,16,3)))
    out=m._run({}, {"contents":[{"type":"prompt","text":"cat"}],"settings":{"resolution":"2k"}}, True, m.PREVIEW_PATH)
    assert len(calls) == 1
    assert out[0].shape == (1,16,16,3)
    assert out[2] == "task_123"


def test_long_image_split_sections_keeps_all_original_pixels():
    # 11 rows / 3 sections => each tile 4 rows, last tile pads one row.
    src = torch.arange(11, dtype=torch.float32).reshape(1, 11, 1, 1).expand(1, 11, 5, 3)
    chunks = m._split_long_image_batch(src, 3)
    assert tuple(chunks.shape) == (3, 4, 5, 3)
    restored = chunks.reshape(1, 12, 5, 3)[:, :11]
    assert torch.equal(restored, src)
    assert torch.equal(chunks[-1, -1], src[0, -1])


def test_split_remainder_distribution_does_not_drop_rows():
    # H=10, N=6 used to create empty chunks with a fixed ceil-height stride.
    src = torch.arange(10, dtype=torch.float32).reshape(1,10,1,1).expand(1,10,4,3)
    tiles = m._split_long_image_batch(src, 6)
    assert tiles.shape == (6, 2, 4, 3)
    restored = torch.cat([tiles[i,:2 if i < 4 else 1] for i in range(6)], dim=0)
    assert torch.equal(restored, src[0])


def test_split_long_image_is_offline_and_adjustable(monkeypatch):
    monkeypatch.setattr(m, "_client_from_auth", lambda auth: pytest.fail("API called"))
    local_image = torch.arange(120, dtype=torch.float32).reshape(1, 12, 10, 1).expand(1,12,10,3)
    tiles, n = m.KlingImage4_SplitLongImage().split(local_image, 6)
    assert n == 6
    assert tuple(tiles.shape) == (6, 2, 10, 3)


def test_long_image_returns_full_image_and_sections_without_changing_original_output_order(monkeypatch):
    source = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3, 1).expand(1,8,3,3)
    monkeypatch.setattr(m, "_run", lambda *args: (source, "urls", "id", "metadata"))
    outputs = m.KlingImage4Preview_LongImage().generate(
        {}, "lookbook page", "webpage", "long", "2k", True, section_count=4)
    assert len(outputs) == 5
    assert outputs[0] is source
    assert outputs[1:4] == ("urls", "id", "metadata")
    assert tuple(outputs[4].shape) == (4, 2, 3, 3)
    assert m.KlingImage4Preview_LongImage.RETURN_NAMES[-1] == "sections"


def test_split_long_image_rejects_more_sections_than_height():
    with pytest.raises(ValueError, match="height"):
        m._split_long_image_batch(torch.zeros(1, 3, 8, 3), 6)


def test_existing_long_image_generate_without_section_count_still_works(monkeypatch):
    monkeypatch.setattr(
        m, "_run",
        lambda *args: (torch.ones(1,60,5,3), "urls", "id", "metadata")
    )
    out = m.KlingImage4Preview_LongImage().generate(
        {}, "landing page", "webpage", "long", "2k", True
    )
    assert out[4].shape[0] == 6

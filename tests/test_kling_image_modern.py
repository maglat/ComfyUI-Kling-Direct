import json
import torch
import pytest

import kling_image_modern as ki


def test_capabilities_filters_image_models(monkeypatch):
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {
            "availableModels": {
                "text_to_image": [{"model": "kling-image-v3_0", "arguments": [], "inputs": []}],
                "image_to_image": [{"model": "kling-image-o1", "arguments": [], "inputs": []}],
                "text_to_video": [{"model": "kling-video-v4_0"}],
            }
        },
    )
    ready, models, payload = ki.KlingImageCLI_Capabilities().check()
    assert ready is True
    assert "kling-image-v3_0" in models
    assert "kling-image-o1" in models
    data = json.loads(payload)
    assert "text_to_video" not in data


def test_text_to_image_uses_live_declared_arguments(monkeypatch):
    spec = {
        "model": "kling-image-v3_0",
        "arguments": [
            {"name": "imageCount", "allowedValues": ["1", "2", "3"]},
            {"name": "imgResolution", "allowedValues": ["1k", "2k"]},
            {"name": "aspectRatio", "allowedValues": ["1:1", "16:9"]},
        ],
        "inputs": [],
    }
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {"availableModels": {"text_to_image": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "g1",
            "status": "COMPLETED",
            "works": [{"contentType": "image", "url": "https://example.com/a.png"}],
        }

    monkeypatch.setattr(ki, "_run_cli", fake_run)
    monkeypatch.setattr(ki, "_finish_images", lambda payload: ("ok",))

    out = ki.KlingImageCLI_TextToImage().generate(
        "product shot",
        "kling-image-v3_0",
        2,
        "2k",
        "1:1",
        300,
        "",
    )
    assert out == ("ok",)
    args = captured["args"]
    assert args[:3] == ["text_to_image", "--model", "kling-image-v3_0"]
    assert "--imageCount" in args and "2" in args
    assert "--imgResolution" in args and "2k" in args
    assert "--aspectRatio" in args and "1:1" in args


def test_text_to_image_rejects_live_disallowed_value(monkeypatch):
    spec = {
        "model": "kling-image-v3_0",
        "arguments": [
            {"name": "imgResolution", "allowedValues": ["1k", "2k"]},
        ],
        "inputs": [],
    }
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {"availableModels": {"text_to_image": [spec]}},
    )
    with pytest.raises(ki.Kling4CLIError, match="Allowed values"):
        ki.KlingImageCLI_TextToImage().generate(
            "x", "kling-image-v3_0", 1, "4k", "auto", 300, ""
        )


def test_image_to_image_uses_repeated_image_flags(monkeypatch):
    spec = {
        "model": "kling-image-v3_0",
        "arguments": [{"name": "imageCount"}],
        "inputs": [{"name": "subject_image_1"}, {"name": "subject_image_2"}],
    }
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_image": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "g2",
            "status": "COMPLETED",
            "works": [{"contentType": "image", "url": "https://example.com/b.png"}],
        }

    monkeypatch.setattr(ki, "_run_cli", fake_run)
    monkeypatch.setattr(ki, "_finish_images", lambda payload: ("ok",))

    image = torch.zeros((1, 320, 320, 3), dtype=torch.float32)
    out = ki.KlingImageCLI_ImageToImage().generate(
        "keep the garment",
        "kling-image-v3_0",
        1,
        "auto",
        "auto",
        "",
        "",
        300,
        "",
        refer_image_1=image,
        refer_image_2=image,
    )
    assert out == ("ok",)
    assert captured["args"].count("--image") == 2


def test_rest_omni_can_select_o1(monkeypatch):
    captured = {}

    class FakeClient:
        def create_legacy_task(self, endpoint, body):
            captured["endpoint"] = endpoint
            captured["body"] = body
            return "task-o1"

        def poll_legacy_task(self, endpoint, task_id):
            return {
                "task_result": {
                    "images": [{"url": "https://example.com/o1.png"}]
                }
            }

    monkeypatch.setattr(ki, "_client_from_auth", lambda auth: FakeClient())
    monkeypatch.setattr(ki, "_download_image_batch", lambda urls: "images")

    out = ki.KlingImageREST_Omni().generate(
        {"api_key": "x"},
        "kling-image-o1",
        "studio product image",
        "4k",
        "auto",
        "single",
        1,
        "auto",
        "",
        False,
    )
    assert out[0] == "images"
    assert captured["endpoint"] == "/v1/images/omni-image"
    assert captured["body"]["model_name"] == "kling-image-o1"


def test_future_image_model_name_is_not_hardcoded(monkeypatch):
    future = {
        "model": "kling-image-v4_0",
        "arguments": [{"name": "imageCount"}],
        "inputs": [],
    }
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {"availableModels": {"text_to_image": [future]}},
    )
    monkeypatch.setattr(
        ki,
        "_run_cli",
        lambda args, timeout=60: {
            "generationId": "future",
            "status": "COMPLETED",
            "works": [{"contentType": "image", "url": "https://example.com/f.png"}],
        },
    )
    monkeypatch.setattr(ki, "_finish_images", lambda payload: ("future-ok",))

    assert ki.KlingImageCLI_TextToImage().generate(
        "future test", "kling-image-v4_0", 1, "auto", "auto", 300, ""
    ) == ("future-ok",)



def test_image_to_image_required_empty_prompt_is_explicit(monkeypatch):
    spec = {
        "model": "image-empty-prompt",
        "arguments": [{"name": "prompt", "required": True}],
        "inputs": [{"name": "subject_image_1", "required": True}],
    }
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_image": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "g-empty",
            "status": "COMPLETED",
            "works": [{"contentType": "image", "url": "https://example.com/empty.png"}],
        }

    monkeypatch.setattr(ki, "_run_cli", fake_run)
    monkeypatch.setattr(ki, "_finish_images", lambda payload: ("ok",))

    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    assert ki.KlingImageCLI_ImageToImage().generate(
        "",
        "image-empty-prompt",
        1,
        "auto",
        "auto",
        "",
        "",
        300,
        "",
        refer_image_1=image,
    ) == ("ok",)
    idx = captured["args"].index("--prompt")
    assert captured["args"][idx + 1] == ""


def test_image_to_image_named_input_uses_exact_live_slot(monkeypatch):
    spec = {
        "model": "image-named-slot",
        "arguments": [{"name": "prompt", "required": False}],
        "inputs": [{"name": "reference_front", "required": True}],
    }
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_image": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "g-named",
            "status": "COMPLETED",
            "works": [{"contentType": "image", "url": "https://example.com/named.png"}],
        }

    monkeypatch.setattr(ki, "_run_cli", fake_run)
    monkeypatch.setattr(ki, "_finish_images", lambda payload: ("ok",))

    assert ki.KlingImageCLI_ImageToImage().generate(
        "keep product shape",
        "image-named-slot",
        1,
        "auto",
        "auto",
        "",
        '{"reference_front":"https://example.com/front.png"}',
        300,
        "",
    ) == ("ok",)
    joined = " ".join(captured["args"])
    assert "reference_front=https://example.com/front.png" in joined
    assert "--image" not in captured["args"]


def test_image_to_image_rejects_mixing_image_and_named_input(monkeypatch):
    spec = {
        "model": "image-no-mix",
        "arguments": [],
        "inputs": [{"name": "reference_front", "required": False}],
    }
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_image": [spec]}},
    )
    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="does not allow mixing"):
        ki.KlingImageCLI_ImageToImage().generate(
            "x",
            "image-no-mix",
            1,
            "auto",
            "auto",
            "",
            '{"reference_front":"https://example.com/front.png"}',
            300,
            "",
            refer_image_1=image,
        )


def test_image_elements_require_new_binding_shape(monkeypatch):
    spec = {
        "model": "image-elements",
        "arguments": [{"name": "elements"}, {"name": "prompt", "required": False}],
        "inputs": [],
    }
    monkeypatch.setattr(
        ki,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_image": [spec]}},
    )
    with pytest.raises(ValueError, match="bindName"):
        ki.KlingImageCLI_ImageToImage().generate(
            "",
            "image-elements",
            1,
            "auto",
            "auto",
            '[{"id":"element-1"}]',
            "",
            300,
            "",
        )

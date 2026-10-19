import json

import pytest
import torch

import kling4_cli as k4
import kling_video_dynamic as kv


def test_cli_envelope_unwrap_prefers_body_over_wrapper_status():
    payload = k4._unwrap_payload({
        "ok": True,
        "status": "success",
        "body": {
            "availableModels": {
                "text_to_video": [{"model": "server-video-model"}]
            }
        },
    })
    assert payload["availableModels"]["text_to_video"][0]["model"] == "server-video-model"


def test_cli_polled_generation_unwraps_result():
    payload = k4._unwrap_payload({
        "ok": True,
        "status": "success",
        "body": {
            "polled": True,
            "generations": [{
                "generationId": "g-1",
                "status": "COMPLETED",
                "result": {
                    "works": [{"contentType": "video", "url": "https://example.com/v.mp4"}]
                },
            }],
        },
    })
    assert payload["generationId"] == "g-1"
    assert payload["status"] == "COMPLETED"
    assert payload["works"][0]["contentType"] == "video"


def test_capabilities_are_dynamic_and_not_kling4_id_filtered(monkeypatch):
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {
            "availableModels": {
                "text_to_video": [
                    {"model": "future-video-x", "arguments": [], "inputs": []}
                ],
                "image_to_video": [
                    {"model": "another-rollout-id", "arguments": [], "inputs": []}
                ],
                "text_to_image": [{"model": "image-only"}],
            }
        },
    )
    ready, summary, raw = kv.KlingVideoCLI_Capabilities().check()
    assert ready is True
    assert "future-video-x" in summary
    assert "another-rollout-id" in summary
    assert "image-only" not in summary
    parsed = json.loads(raw)
    assert "text_to_image" not in parsed


def test_text_to_video_uses_live_model_and_allowed_values(monkeypatch):
    spec = {
        "model": "future-video-x",
        "arguments": [
            {"name": "duration", "allowedValues": ["5", "10", "30"]},
            {"name": "aspectRatio", "allowedValues": ["16:9", "9:16"]},
            {"name": "resolution", "allowedValues": ["720p", "1080p", "4k"]},
            {"name": "enableAudio", "allowedValues": ["true", "false"]},
            {"name": "dynamicRange", "allowedValues": ["SDR", "HDR"]},
        ],
        "inputs": [],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"text_to_video": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "g1",
            "status": "COMPLETED",
            "works": [{"contentType": "video", "url": "https://example.com/v.mp4"}],
        }

    monkeypatch.setattr(kv, "_run_cli", fake_run)
    monkeypatch.setattr(kv, "_finish_video", lambda payload: ("ok",))

    result = kv.KlingVideoCLI_TextToVideo().generate(
        "cinematic shot",
        "future-video-x",
        30,
        "16:9",
        "4k",
        "on",
        "hdr",
        300,
        "",
    )
    assert result == ("ok",)
    args = captured["args"]
    assert args[:3] == ["text_to_video", "--model", "future-video-x"]
    assert "--duration" in args and "30" in args
    assert "--resolution" in args and "4k" in args
    assert "--enableAudio" in args and "true" in args
    assert "--dynamicRange" in args and "HDR" in args


def test_text_to_video_rejects_value_not_allowed_by_live_schema(monkeypatch):
    spec = {
        "model": "server-video",
        "arguments": [
            {"name": "duration", "allowedValues": ["5", "10"]},
        ],
        "inputs": [],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"text_to_video": [spec]}},
    )
    with pytest.raises(kv.Kling4CLIError, match="Allowed values"):
        kv.KlingVideoCLI_TextToVideo().generate(
            "x", "server-video", 30, "auto", "auto", "auto", "auto", 300, ""
        )


def test_image_to_video_can_be_prompt_only_when_live_model_requires_no_image(monkeypatch):
    spec = {
        "model": "hybrid-video-model",
        "arguments": [],
        "inputs": [],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_video": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "g2",
            "status": "COMPLETED",
            "works": [{"contentType": "video", "url": "https://example.com/v2.mp4"}],
        }

    monkeypatch.setattr(kv, "_run_cli", fake_run)
    monkeypatch.setattr(kv, "_finish_video", lambda payload: ("ok",))

    result = kv.KlingVideoCLI_ImageToVideo().generate(
        "prompt only",
        "hybrid-video-model",
        0,
        "auto",
        "auto",
        "auto",
        "",
        300,
        "",
    )
    assert result == ("ok",)
    assert "--input" not in captured["args"]
    assert captured["args"][-1] == "prompt only"


def test_image_to_video_named_input_uses_exact_live_slot(monkeypatch):
    spec = {
        "model": "keyframe-model",
        "arguments": [],
        "inputs": [
            {"name": "strange_middle_frame", "required": True},
        ],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_video": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "g3",
            "status": "COMPLETED",
            "works": [{"contentType": "video", "url": "https://example.com/v3.mp4"}],
        }

    monkeypatch.setattr(kv, "_run_cli", fake_run)
    monkeypatch.setattr(kv, "_finish_video", lambda payload: ("ok",))

    result = kv.KlingVideoCLI_ImageToVideo().generate(
        "",
        "keyframe-model",
        0,
        "auto",
        "auto",
        "auto",
        '{"strange_middle_frame":"https://example.com/frame.png"}',
        300,
        "",
    )
    assert result == ("ok",)
    command = " ".join(captured["args"])
    assert "strange_middle_frame=https://example.com/frame.png" in command


def test_omni_combined_item_cap_comes_from_live_schema(monkeypatch):
    spec = {
        "model": "omni-future",
        "arguments": [
            {"name": "elements"},
        ],
        "inputs": [
            {"name": "image_1", "required": False},
            {"name": "image_2", "required": False},
            {"name": "video_1", "required": False},
        ],
        "combinedItemCap": {
            "maxItems": 2,
            "inputPrefixes": ["image_", "video_"],
            "arguments": ["elements"],
        },
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"omni_ref_video": [spec]}},
    )

    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    with pytest.raises(kv.Kling4CLIError, match="at most 2 combined"):
        kv.KlingVideoCLI_OmniReference().generate(
            "edit",
            "omni-future",
            0,
            "auto",
            "auto",
            "auto",
            "auto",
            False,
            "",
            "",
            "",
            '[{"id":"e1","bindName":"hero","type":"character"}]',
            "",
            300,
            "",
            refer_image_1=image,
            refer_image_2=image,
        )


def test_motion_control_accepts_any_server_advertised_model(monkeypatch):
    spec = {
        "model": "motion-next",
        "arguments": [],
        "inputs": [],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"motion_control": [spec]}},
    )
    monkeypatch.setattr(
        kv,
        "_run_cli",
        lambda args, timeout=60: {
            "generationId": "gm",
            "status": "COMPLETED",
            "works": [{"contentType": "video", "url": "https://example.com/m.mp4"}],
        },
    )
    monkeypatch.setattr(kv, "_finish_video", lambda payload: ("motion-ok",))

    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    assert kv.KlingVideoCLI_MotionControl().generate(
        image,
        "motion-next",
        "",
        "https://example.com/motion.mp4",
        "",
        300,
        "",
    ) == ("motion-ok",)



def test_i2v_required_empty_prompt_is_sent_explicitly(monkeypatch):
    spec = {
        "model": "i2v-empty-prompt",
        "arguments": [{"name": "prompt", "required": True}],
        "inputs": [],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_video": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "gp",
            "status": "COMPLETED",
            "works": [{"contentType": "video", "url": "https://example.com/p.mp4"}],
        }

    monkeypatch.setattr(kv, "_run_cli", fake_run)
    monkeypatch.setattr(kv, "_finish_video", lambda payload: ("ok",))

    assert kv.KlingVideoCLI_ImageToVideo().generate(
        "",
        "i2v-empty-prompt",
        0,
        "auto",
        "auto",
        "auto",
        "",
        300,
        "",
    ) == ("ok",)
    idx = captured["args"].index("--prompt")
    assert captured["args"][idx + 1] == ""


def test_i2v_optional_empty_prompt_is_omitted(monkeypatch):
    spec = {
        "model": "i2v-optional-prompt",
        "arguments": [{"name": "prompt", "required": False}],
        "inputs": [],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_video": [spec]}},
    )
    captured = {}
    monkeypatch.setattr(
        kv,
        "_run_cli",
        lambda args, timeout=60: captured.setdefault("result", list(args)) or {},
    )
    monkeypatch.setattr(kv, "_finish_video", lambda payload: ("ok",))

    assert kv.KlingVideoCLI_ImageToVideo().generate(
        "",
        "i2v-optional-prompt",
        0,
        "auto",
        "auto",
        "auto",
        "",
        300,
        "",
    ) == ("ok",)
    assert "--prompt" not in captured["result"]


def test_omni_audio_reference_uses_official_audio_flag(monkeypatch):
    spec = {
        "model": "omni-audio-model",
        "arguments": [],
        "inputs": [{"name": "audio_1", "required": True}],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"omni_ref_video": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {
            "generationId": "ga",
            "status": "COMPLETED",
            "works": [{"contentType": "video", "url": "https://example.com/a.mp4"}],
        }

    monkeypatch.setattr(kv, "_run_cli", fake_run)
    monkeypatch.setattr(kv, "_finish_video", lambda payload: ("ok",))

    assert kv.KlingVideoCLI_OmniReference().generate(
        "speak with the reference audio",
        "omni-audio-model",
        0,
        "auto",
        "auto",
        "auto",
        "auto",
        False,
        "",
        "",
        "https://example.com/voice.mp3",
        "",
        "",
        300,
        "",
    ) == ("ok",)
    assert "--audio" in captured["args"]
    assert "https://example.com/voice.mp3" in captured["args"]


def test_omni_audio_reference_rejected_when_live_model_has_no_audio_slot(monkeypatch):
    spec = {
        "model": "omni-no-audio",
        "arguments": [],
        "inputs": [{"name": "image_1", "required": False}],
    }
    monkeypatch.setattr(
        kv,
        "_who_am_i",
        lambda: {"availableModels": {"omni_ref_video": [spec]}},
    )
    with pytest.raises(kv.Kling4CLIError, match="no audio input slots"):
        kv.KlingVideoCLI_OmniReference().generate(
            "x",
            "omni-no-audio",
            0,
            "auto",
            "auto",
            "auto",
            "auto",
            False,
            "",
            "",
            "https://example.com/voice.mp3",
            "",
            "",
            300,
            "",
        )


def test_element_binding_requires_id_bindname_and_type():
    with pytest.raises(ValueError, match="bindName"):
        k4._parse_elements_json('[{"id":"123"}]')

    parsed = k4._parse_elements_json(
        '[{"id":"123","bindName":"hero","type":"character"}]'
    )
    assert parsed[0]["type"] == "character"

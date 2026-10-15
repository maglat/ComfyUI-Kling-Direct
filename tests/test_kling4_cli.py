import torch
import pytest

import kling4_cli as k4


def test_json_parser_accepts_direct_and_last_line():
    assert k4._parse_json_output('{"status":"ok"}')["status"] == "ok"
    assert k4._parse_json_output('banner\n{"generationId":"g1"}')["generationId"] == "g1"


def test_flash_limits_are_enforced_before_submit():
    k4._validate_model_basics(k4.KLING4_FLASH, 20, "720p")
    with pytest.raises(ValueError):
        k4._validate_model_basics(k4.KLING4_FLASH, 21, "720p")
    with pytest.raises(ValueError):
        k4._validate_model_basics(k4.KLING4_FLASH, 10, "1080p")


def test_model_must_be_advertised_by_who_am_i():
    who = {
        "availableModels": {
            "text_to_video": [
                {"model": k4.KLING4_STANDARD, "arguments": [], "inputs": []}
            ]
        }
    }
    assert (
        k4._model_spec("text_to_video", k4.KLING4_STANDARD, who)["model"]
        == k4.KLING4_STANDARD
    )
    with pytest.raises(k4.Kling4CLIError):
        k4._model_spec("text_to_video", k4.KLING4_FLASH, who)


def test_t2v_uses_live_declared_flags(monkeypatch):
    spec = {
        "model": k4.KLING4_STANDARD,
        "arguments": [
            {"name": "duration"},
            {"name": "aspect_ratio"},
            {"name": "resolution"},
            {"name": "enable_audio"},
        ],
        "inputs": [],
    }
    monkeypatch.setattr(
        k4,
        "_who_am_i",
        lambda: {"availableModels": {"text_to_video": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {"generationId": "g1", "status": "COMPLETED", "works": []}

    monkeypatch.setattr(k4, "_run_cli", fake_run)
    monkeypatch.setattr(k4, "_finish_video", lambda payload: ("ok",))

    result = k4.Kling4CLI_TextToVideo().generate(
        "hello",
        k4.KLING4_STANDARD,
        30,
        "21:9",
        "4k",
        "on",
        300,
        "",
    )
    assert result == ("ok",)
    args = captured["args"]
    assert args[:3] == ["text_to_video", "--model", k4.KLING4_STANDARD]
    assert "--duration" in args and "30" in args
    assert "--aspectRatio" in args and "21:9" in args
    assert "--resolution" in args and "4k" in args
    assert "--enableAudio" in args and "true" in args
    assert args[-1] == "hello"


def test_i2v_maps_first_middle_and_tail_to_declared_slots(monkeypatch):
    spec = {
        "model": k4.KLING4_STANDARD,
        "arguments": [{"name": "duration"}],
        "inputs": [
            {"name": "first_image"},
            {"name": "keyframe_1"},
            {"name": "tail_image"},
        ],
    }
    monkeypatch.setattr(
        k4,
        "_who_am_i",
        lambda: {"availableModels": {"image_to_video": [spec]}},
    )
    captured = {}

    def fake_run(args, timeout=60):
        captured["args"] = list(args)
        return {"generationId": "g1", "status": "COMPLETED", "works": []}

    monkeypatch.setattr(k4, "_run_cli", fake_run)
    monkeypatch.setattr(k4, "_finish_video", lambda payload: ("ok",))

    image = torch.zeros((1, 320, 320, 3), dtype=torch.float32)
    result = k4.Kling4CLI_ImageToVideo().generate(
        image,
        "prompt",
        k4.KLING4_STANDARD,
        8,
        "auto",
        "auto",
        300,
        "",
        tail_frame=image,
        keyframe_1=image,
    )
    assert result == ("ok",)
    command = " ".join(captured["args"])
    assert "first_image=" in command
    assert "keyframe_1=" in command
    assert "tail_image=" in command


def test_advanced_args_cannot_override_transport_controls():
    cmd = []
    with pytest.raises(ValueError):
        k4._append_advanced_args(cmd, "--model bad")

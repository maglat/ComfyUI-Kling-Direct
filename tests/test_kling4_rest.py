import pytest
import torch

import kling4_rest as k4


class FakeClient:
    def __init__(self, captured):
        self.captured = captured

    def create_new_video(self, product, model, body):
        self.captured.update(product=product, model=model, body=body)
        return "task-v4"

    def poll_new_task(self, task_id):
        return {
            "id": task_id,
            "status": "succeeded",
            "outputs": [
                {
                    "type": "video",
                    "id": "video-1",
                    "url": "https://example.com/out.mp4",
                    "duration": "5",
                }
            ],
        }


def _patch_network(monkeypatch, captured):
    monkeypatch.setattr(k4, "_client_from_auth", lambda auth: FakeClient(captured))
    monkeypatch.setattr(
        k4,
        "_finish",
        lambda task, task_id: (
            None,
            "out.mp4",
            None,
            task["outputs"][0]["url"],
            task_id,
            "{}",
        ),
    )


def _image():
    return torch.zeros((1, 320, 320, 3), dtype=torch.float32)


def test_text_to_video_uses_exact_early_access_endpoint_and_body(monkeypatch):
    captured = {}
    _patch_network(monkeypatch, captured)

    node = k4.Kling4REST_TextToVideo()
    node.generate(
        {"api_key": "x"},
        "A cinematic water drop.",
        "720p",
        "21:9",
        4,
        False,
        "",
        "external-1",
    )

    assert captured["product"] == "text-to-video"
    assert captured["model"] == "kling-4.0"
    assert captured["body"]["settings"] == {
        "duration": 4,
        "resolution": "720p",
        "aspect_ratio": "21:9",
    }
    assert captured["body"]["options"]["external_task_id"] == "external-1"
    assert "audio" not in captured["body"]["settings"]


def test_image_to_video_builds_first_middle_last_keyframes(monkeypatch):
    captured = {}
    _patch_network(monkeypatch, captured)
    monkeypatch.setattr(k4, "_encode_image", lambda image, label: f"b64-{label}")

    node = k4.Kling4REST_ImageToVideo()
    node.generate(
        {"api_key": "x"},
        "Move from @image_1 through @image_2 to @image_3.",
        "1080p",
        8,
        False,
        "",
        "",
        first_frame=_image(),
        middle_frame_1=_image(),
        last_frame=_image(),
    )

    assert captured["product"] == "image-to-video"
    rows = captured["body"]["contents"]
    assert rows[0]["type"] == "prompt"
    assert [(r["type"], r["id"]) for r in rows[1:]] == [
        ("first_frame", "image_1"),
        ("middle_frame", "image_2"),
        ("last_frame", "image_3"),
    ]
    assert captured["body"]["settings"] == {
        "duration": 8,
        "resolution": "1080p",
    }


def test_omni_builds_mixed_reference_payload(monkeypatch):
    captured = {}
    _patch_network(monkeypatch, captured)
    monkeypatch.setattr(k4, "_encode_image", lambda image, label: f"b64-{label}")

    node = k4.Kling4REST_OmniVideo()
    node.generate(
        {"api_key": "x"},
        "Use @image_1, @video_1, @video_2, @element_1 and @voice_1.",
        "1080p",
        "adaptive",
        10,
        "https://example.com/feature.mp4",
        "https://example.com/base.mp4",
        "164,136",
        "voice-a",
        False,
        "",
        "omni-test",
        refer_image_1=_image(),
    )

    assert captured["product"] == "omni-video"
    rows = captured["body"]["contents"]
    assert rows[0] == {
        "type": "prompt",
        "text": "Use @image_1, @video_1, @video_2, @element_1 and @voice_1.",
    }
    assert any(r.get("type") == "refer_image" and r.get("id") == "image_1" for r in rows)
    assert any(r.get("type") == "base_video" and r.get("id") == "video_1" for r in rows)
    assert any(r.get("type") == "feature_video" and r.get("id") == "video_2" for r in rows)
    assert any(r.get("type") == "element" and r.get("id") == "element_1" for r in rows)
    assert any(r.get("type") == "voice" and r.get("id") == "voice_1" for r in rows)
    assert captured["body"]["settings"]["aspect_ratio"] == "adaptive"


def test_omni_rejects_more_than_15_combined_items():
    node = k4.Kling4REST_OmniVideo()
    with pytest.raises(ValueError, match="15 combined"):
        node.generate(
            {"api_key": "x"},
            "prompt",
            "1080p",
            "adaptive",
            10,
            "\n".join(
                [
                    "https://example.com/a.mp4",
                    "https://example.com/b.mp4",
                    "https://example.com/c.mp4",
                    "https://example.com/d.mp4",
                    "https://example.com/e.mp4",
                ]
            ),
            "",
            "1,2,3,4,5,6,7",
            "v1,v2,v3",
            False,
            "",
            "",
            refer_image_1=_image(),
        )


def test_prompt_limit_enforced():
    with pytest.raises(ValueError, match="mixed-language limit"):
        k4._validate_prompt("a" * 26001, required=True)


def test_image_minimum_dimension_enforced():
    with pytest.raises(ValueError, match=">= 300px"):
        k4._validate_image(
            torch.zeros((1, 299, 320, 3), dtype=torch.float32),
            "image",
        )

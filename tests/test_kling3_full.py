import pytest

import kling3_full as k3


def test_auth_uses_explicit_api_key():
    auth = k3._auth_from_input("abc123", "https://example.test", True)
    assert auth["api_key"] == "abc123"
    assert auth["base_url"] == "https://example.test"
    assert auth["debug"] is True


def test_auth_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("KLING_API_KEY", "env-key")
    auth = k3._auth_from_input("", k3.DEFAULT_BASE_URL, False)
    assert auth["api_key"] == "env-key"


def test_storyboard_builds_prompt_and_duration():
    node = k3.Kling3Full_Storyboard()
    prompt, total, multi = node.build(
        shot_1_duration=3, shot_1_prompt="Close up",
        shot_2_duration=4, shot_2_prompt="Pull back",
        shot_3_duration=0, shot_3_prompt="",
        shot_4_duration=0, shot_4_prompt="",
        shot_5_duration=0, shot_5_prompt="",
        shot_6_duration=0, shot_6_prompt="",
    )
    assert total == 7
    assert multi is True
    assert "shot 1, 3, Close up;" in prompt
    assert "shot 2, 4, Pull back;" in prompt


def test_storyboard_rejects_over_15_seconds():
    node = k3.Kling3Full_Storyboard()
    with pytest.raises(ValueError, match="3–15"):
        node.build(
            shot_1_duration=10, shot_1_prompt="A",
            shot_2_duration=6, shot_2_prompt="B",
            shot_3_duration=0, shot_3_prompt="",
            shot_4_duration=0, shot_4_prompt="",
            shot_5_duration=0, shot_5_prompt="",
            shot_6_duration=0, shot_6_prompt="",
        )


def test_client_uses_direct_bearer_header():
    c = k3.Kling3Client("secret-key")
    assert c.headers["Authorization"] == "Bearer secret-key"
    assert c.headers["Content-Type"] == "application/json"


def test_turbo_rejects_4k_before_network():
    node = k3.Kling3Full_TextToVideo()
    with pytest.raises(ValueError, match="not 4K"):
        node.generate(
            {"api_key": "x"}, "prompt", "kling-3.0-turbo", "4k", "16:9", 5,
            "native", False, False,
        )


def test_turbo_rejects_audio_off_before_network():
    node = k3.Kling3Full_TextToVideo()
    with pytest.raises(ValueError, match="always on"):
        node.generate(
            {"api_key": "x"}, "prompt", "kling-3.0-turbo", "1080p", "16:9", 5,
            "off", False, False,
        )


def test_text_to_video_builds_current_kling30_payload(monkeypatch):
    captured = {}

    class FakeClient:
        def create_new_video(self, product, model, body):
            captured.update(product=product, model=model, body=body)
            return "task-1"

        def poll_new_task(self, task_id):
            return {
                "id": task_id,
                "status": "succeeded",
                "outputs": [{"type": "video", "id": "v1", "url": "https://example.com/v.mp4"}],
            }

    monkeypatch.setattr(k3, "_client_from_auth", lambda auth: FakeClient())
    monkeypatch.setattr(
        k3,
        "_finish_video",
        lambda task, task_id: (None, "", None, task["outputs"][0]["url"], task_id, "{}"),
    )

    node = k3.Kling3Full_TextToVideo()
    node.generate(
        {"api_key": "x"}, "hello", "kling-3.0", "4k", "9:16", 15,
        "native", True, False,
    )

    assert captured["product"] == "text-to-video"
    assert captured["model"] == "kling-3.0"
    assert captured["body"]["prompt"] == "hello"
    assert captured["body"]["settings"] == {
        "resolution": "4k",
        "aspect_ratio": "9:16",
        "duration": 15,
        "audio": "native",
        "multi_shot": True,
    }
    assert "external_task_id" in captured["body"]["options"]

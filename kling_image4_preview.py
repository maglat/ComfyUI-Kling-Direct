"""Experimental Kling Image 4.0 *direct REST* integration.

Kling announced developer API availability on 2026-11-18, but as of that date
its public developer reference still documents Image 3.0 rather than the native
Image 4.0 REST contract. The path below is INFERRED from a third-party's
provider-native workflow, NOT confirmed by Kling. No paid task is submitted
until the user explicitly enables the preview switch.

Source references:
  https://kling.ai/blog/kling-image-4-stunningly-realistic-sharp-precise
  https://kling.ai/document-api/apiReference/model/imageGeneration
  https://picsart.com/api-platform/models/kling-image-4.0
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import torch

try:
    from .kling_nodes import tensor_to_base64_string, download_to_tensor
    from .kling3_full import _client_from_auth, Kling3APIError
except ImportError:
    from kling_nodes import tensor_to_base64_string, download_to_tensor
    from kling3_full import _client_from_auth, Kling3APIError


# Provider-native route inferred from Picsart's kling/v1/images/kling-v4-image
# workflow. DO NOT describe this as an officially documented Kling endpoint.
PREVIEW_PATH = "/v1/images/kling-v4-image"
RATIOS = ["auto", "16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"]
SIZES = ["2k", "4k", "8k"]
GUIDANCE = (
    "EXPERIMENTAL: Kling has not yet published a native Image 4.0 REST schema. "
    "The inferred endpoint and task format may differ. No request will be sent "
    "unless you enable the preview switch. API access and billing may require "
    "account-specific authorization."
)


def _validate_prompt(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        raise ValueError("Kling Image 4.0 requires a nonempty prompt.")
    if len(text) > 21000:
        raise ValueError("Preview safety limit: prompt must be <= 21,000 characters.")
    return text


def _references(images: List[Any], max_count: int = 10) -> List[Dict[str, str]]:
    out = []
    for image in images:
        if image is None:
            continue
        frame = image[0] if image.ndim == 4 else image
        if frame.ndim != 3:
            raise ValueError("Each reference must be one ComfyUI IMAGE tensor.")
        height, width = frame.shape[:2]
        if height < 300 or width < 300 or not 0.4 <= width / height <= 2.5:
            raise ValueError("Kling references must be >= 300px on each side, aspect 2:5..5:2.")
        # Image 4.0's binary/URL handling is not confirmed by Kling; use existing
        # ComfyUI-Kling-Direct bare base64 convention. Test before production use.
        out.append({"image_url": tensor_to_base64_string(image), "id": f"image_{len(out)+1}"})
    if len(out) > max_count:
        raise ValueError(f"At most {max_count} reference images may be provided.")
    return out


def _urls(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    result = data.get("task_result") or data.get("result") or {}
    items = []
    for obj in (data, result):
        if not isinstance(obj, dict):
            continue
        for field in ("images", "series_images", "outputs", "items"):
            value = obj.get(field)
            if isinstance(value, list):
                items.extend(value)
        if isinstance(obj.get("image"), str):
            items.append(obj["image"])
    urls = []
    for item in items:
        if isinstance(item, str):
            url = item
        elif isinstance(item, dict):
            url = item.get("url") or item.get("image_url") or item.get("image")
        else:
            url = None
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            urls.append(url)
    return list(dict.fromkeys(urls))


# A combo box is used rather than a BOOLEAN widget because some ComfyUI
# frontends hide BOOLEAN controls inside migrated/previously saved nodes.
# String values are deliberately compared exactly: a nonempty "BLOCKED" must
# never be mistaken for authorization to send a potentially billable request.
PREVIEW_REQUEST_MODES = [
    "BLOCKED - do not send API request",
    "ALLOW - send unverified REST test request",
]


def _preview_enabled(value: Any) -> bool:
    # Preserve explicit opt-in for workflows saved with the old BOOLEAN input.
    return value is True or value == PREVIEW_REQUEST_MODES[1]


def _run(auth: Dict[str, Any], body: Dict[str, Any], enabled: bool, api_path: str):
    if not enabled:
        raise RuntimeError(GUIDANCE)
    if api_path != PREVIEW_PATH:
        raise ValueError("Preview endpoint override is disabled until the official Kling spec is published.")
    client = _client_from_auth(auth)
    try:
        created = client._request("POST", api_path, body=body)
    except Kling3APIError as exc:
        raise RuntimeError(
            "Kling Image 4.0 preview REST request failed. Kling has not publicly "
            f"confirmed this path. Original API error: {exc}"
        ) from exc

    data = created.get("data") or created
    urls = _urls(data)
    task_id = str(data.get("task_id") or data.get("id") or "") if isinstance(data, dict) else ""
    result = data
    if not urls and task_id:
        # Existing Kling APIs have both /tasks (new standard) and
        # /v1/images/.../{task_id} (legacy). Select based on create response.
        if "task_id" in data:
            result = client.poll_legacy_task(api_path, task_id)
        else:
            result = client.poll_new_task(task_id)
        urls = _urls(result)
    if not urls:
        raise RuntimeError(
            "Kling Image 4.0 task did not return image URLs. No automatic "
            "resubmission was attempted. Inspect the returned task using the "
            f"task ID {task_id or '<missing>'} and the latest Kling documentation."
        )
    tensors = [download_to_tensor(url) for url in urls]
    first_shape = tuple(tensors[0].shape[1:])
    if any(tuple(t.shape[1:]) != first_shape for t in tensors):
        raise RuntimeError("Generated images have different dimensions; ComfyUI IMAGE "
                           "batches require equal dimensions. URLs: " + json.dumps(urls))
    batch = torch.cat(tensors, dim=0)
    return batch, json.dumps(urls, ensure_ascii=False), task_id, json.dumps(result, ensure_ascii=False)


def _optional_refs():
    return {f"reference_{i}": ("IMAGE",) for i in range(1, 11)}


class KlingImage4Preview_TextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "resolution": (SIZES, {"default": "2k"}),
            "aspect_ratio": (RATIOS, {"default": "auto"}),
            "n": ("INT", {"default": 1, "min": 1, "max": 9}),
            "enable_unverified_rest_preview": (PREVIEW_REQUEST_MODES, {"default": PREVIEW_REQUEST_MODES[0], "tooltip": GUIDANCE}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls_json", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Image 4.0 REST (EXPERIMENTAL)"

    def generate(self, auth, prompt, resolution, aspect_ratio, n, enable_unverified_rest_preview):
        body = {"result_type": "single", "prompt": _validate_prompt(prompt),
                "resolution": resolution, "aspect_ratio": aspect_ratio, "n": n}
        return _run(auth, body, _preview_enabled(enable_unverified_rest_preview), PREVIEW_PATH)


class KlingImage4Preview_ImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "Edit @image_1 while preserving product details.", "multiline": True}),
            "image": ("IMAGE",),
            "resolution": (SIZES, {"default": "2k"}),
            "aspect_ratio": (RATIOS, {"default": "auto"}),
            "n": ("INT", {"default": 1, "min": 1, "max": 9}),
            "enable_unverified_rest_preview": (PREVIEW_REQUEST_MODES, {"default": PREVIEW_REQUEST_MODES[0], "tooltip": GUIDANCE}),
        }, "optional": {f"reference_{i}": ("IMAGE",) for i in range(2, 11)}}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls_json", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Image 4.0 REST (EXPERIMENTAL)"

    def generate(self, auth, prompt, image, resolution, aspect_ratio, n,
                 enable_unverified_rest_preview, **kwargs):
        images = [image] + [kwargs.get(f"reference_{i}") for i in range(2, 11)]
        body = {"result_type": "single", "prompt": _validate_prompt(prompt),
                "image_list": _references(images),
                "resolution": resolution, "aspect_ratio": aspect_ratio, "n": n}
        return _run(auth, body, _preview_enabled(enable_unverified_rest_preview), PREVIEW_PATH)


class KlingImage4Preview_Series:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "resolution": (SIZES, {"default": "2k"}),
            "aspect_ratio": (RATIOS, {"default": "auto"}),
            "series_amount": (["auto"] + [str(x) for x in range(2, 11)], {"default": "4"}),
            "enable_unverified_rest_preview": (PREVIEW_REQUEST_MODES, {"default": PREVIEW_REQUEST_MODES[0], "tooltip": GUIDANCE}),
        }, "optional": _optional_refs()}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls_json", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Image 4.0 REST (EXPERIMENTAL)"

    def generate(self, auth, prompt, resolution, aspect_ratio, series_amount,
                 enable_unverified_rest_preview, **kwargs):
        body = {"result_type": "series", "prompt": _validate_prompt(prompt),
                "resolution": resolution, "aspect_ratio": aspect_ratio,
                "series_amount": ("auto" if series_amount == "auto" else int(series_amount))}
        refs = _references([kwargs.get(f"reference_{i}") for i in range(1, 11)])
        if refs:
            body["image_list"] = refs
        return _run(auth, body, _preview_enabled(enable_unverified_rest_preview), PREVIEW_PATH)


class KlingImage4Preview_LongImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "auth": ("KLING_AUTH",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "skill": (["ecommerce", "webpage"], {"default": "ecommerce"}),
            "length": (["short", "medium", "long"], {"default": "medium"}),
            "resolution": (["1k", "2k"], {"default": "2k"}),
            "enable_unverified_rest_preview": (PREVIEW_REQUEST_MODES, {"default": PREVIEW_REQUEST_MODES[0], "tooltip": GUIDANCE}),
        }, "optional": {f"reference_{i}": ("IMAGE",) for i in range(1, 4)}}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "urls_json", "task_id", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = "Kling Image 4.0 REST (EXPERIMENTAL)"

    def generate(self, auth, prompt, skill, length, resolution,
                 enable_unverified_rest_preview, **kwargs):
        refs = _references([kwargs.get(f"reference_{i}") for i in range(1, 4)], max_count=3)
        body = {"result_type": "long_image", "prompt": _validate_prompt(prompt),
                "resolution": resolution, "skill": skill, "length": length}
        if refs:
            body["image_list"] = refs
        return _run(auth, body, _preview_enabled(enable_unverified_rest_preview), PREVIEW_PATH)


NODE_CLASS_MAPPINGS = {
    "KlingImage4Preview_TextToImage": KlingImage4Preview_TextToImage,
    "KlingImage4Preview_ImageEdit": KlingImage4Preview_ImageEdit,
    "KlingImage4Preview_Series": KlingImage4Preview_Series,
    "KlingImage4Preview_LongImage": KlingImage4Preview_LongImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingImage4Preview_TextToImage": "Kling Image 4.0 REST (Preview) - Text to Image",
    "KlingImage4Preview_ImageEdit": "Kling Image 4.0 REST (Preview) - Image Edit",
    "KlingImage4Preview_Series": "Kling Image 4.0 REST (Preview) - Image Series",
    "KlingImage4Preview_LongImage": "Kling Image 4.0 REST (Preview) - Long Image",
}

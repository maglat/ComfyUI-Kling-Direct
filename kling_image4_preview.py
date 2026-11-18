"""Kling Image 4.0 REST integration (official release November 17, 2026).

The official Kling API release notes dated 2026-11-17 publish the model
routes /image-generation/kling-4.0 and /long-image-generation/kling-4.0
and polling via GET /tasks. The contents/settings/options request shape
is corroborated by a published third-party integration; its detailed native
field validation has not yet been exercised with a live Kling account.

Source references:
  https://kling.ai/document-api/updates/api
  https://api-cn.apimaster.ai/docs/en/api/images/kling-image-4.0
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


# Both paths are explicitly listed in Kling's 2026-11-17 official API updates.
PREVIEW_PATH = "/image-generation/kling-4.0"
LONG_IMAGE_PATH = "/long-image-generation/kling-4.0"
RATIOS = ["auto", "16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"]
SIZES = ["2k", "4k", "8k"]
GUIDANCE = (
    "KLING IMAGE 4.0 REST: Official Kling 4.0 image endpoints are available. "
    "This switch explicitly authorizes a possibly billable request. Your API "
    "key may need Image 4.0 entitlement or a compatible resource package. "
    "Native end-to-end request acceptance is not yet verified."
)


def _validate_prompt(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        raise ValueError("Kling Image 4.0 requires a nonempty prompt.")
    english = sum(1 for char in text if ord(char) < 128)
    other = len(text) - english
    if english / 50000 + other / 21000 > 1:
        raise ValueError("Kling Image 4.0 prompt exceeds mixed-language limit (50k English / 21k non-English).")
    return text


def _references(images: List[Any], max_count: int = 10) -> List[Dict[str, str]]:
    """Make Image 4.0 contents references, preserving order and @image_N IDs."""
    out = []
    for image in images:
        if image is None:
            continue
        frame = image[0] if image.ndim == 4 else image
        if frame.ndim != 3:
            raise ValueError("Each reference must be a ComfyUI IMAGE tensor.")
        height, width = frame.shape[:2]
        if height < 300 or width < 300 or not 0.4 <= width / height <= 2.5:
            raise ValueError("Kling references must be >= 300px on both sides, aspect 2:5..5:2.")
        b64 = tensor_to_base64_string(image)
        # PNG can be larger than the requested resolution; enforce the API's 30 MB cap.
        approx_size = len(b64) * 3 // 4
        if approx_size > 30 * 1024 * 1024:
            raise ValueError("Kling Image 4.0 reference exceeds 30 MB.")
        out.append({"type": "refer_image", "id": f"image_{len(out)+1}", "url": b64})
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
    """Submit a new-standard image task once, then poll the existing task.

    Never submit again automatically after a timeout; polling an accepted task
    does not incur a new generation charge.
    """
    if not enabled:
        raise RuntimeError(GUIDANCE + " Set the request dropdown to ALLOW to submit.")
    if api_path not in (PREVIEW_PATH, LONG_IMAGE_PATH):
        raise ValueError("Unknown Kling Image 4.0 REST API path.")
    client = _client_from_auth(auth)
    try:
        created = client._request("POST", api_path, body=body, timeout=180)
        data = created.get("data") or {}
        task_id = str(data.get("id") or data.get("task_id") or "")
        if not task_id:
            raise Kling3APIError("Kling Image 4.0 creation response has no data.id/task_id: "
                                 + json.dumps(created)[:1200])
        print(f"[KLING IMAGE 4.0] Task submitted: {task_id}")
        task = client.poll_new_task(task_id)
    except Kling3APIError as exc:
        raise RuntimeError(
            f"Kling Image 4.0 REST request failed on {api_path}: {exc}. "
            "Check the Kling API key's Image 4.0 entitlement and developer pricing."
        ) from exc
    urls = _urls(task)
    if not urls:
        # Keep task metadata discoverable without submitting a new, paid task.
        raise RuntimeError(
            f"Kling Image 4.0 task {task_id} succeeded but returned no image URLs. "
            "Check GET /tasks for this ID. Response: " + json.dumps(task)[:2500]
        )
    tensors = [download_to_tensor(url) for url in urls]
    first_shape = tuple(tensors[0].shape[1:])
    if any(tuple(t.shape[1:]) != first_shape for t in tensors):
        raise RuntimeError(
            f"Kling Image 4.0 task {task_id} returned different image sizes. "
            "The individual URLs are: " + json.dumps(urls)
        )
    return (torch.cat(tensors, dim=0), json.dumps(urls, ensure_ascii=False),
            task_id, json.dumps(task, ensure_ascii=False))

def _split_long_image_batch(images: torch.Tensor, section_count: int = 6) -> torch.Tensor:
    """Divide each tall image into a batch of full-width sections, top to bottom.

    Long Image returns *one* tall composition. These are local crops, not new
    or independently generated Kling images. Preserve every source pixel; only
    pad the last tile by repeating its bottom row if the height is indivisible.
    ComfyUI IMAGE batches require all tiles to share the same H/W.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("long_image must be a ComfyUI IMAGE batch [B,H,W,C].")
    count = int(section_count)
    if count < 2 or count > 18:
        raise ValueError("section_count must be between 2 and 18.")
    batch, height, width, channels = images.shape
    if batch < 1 or width < 1 or channels not in (3, 4) or height < count:
        raise ValueError("The input image must have sufficient height for the requested sections.")
    rows_per_section, longer_sections = divmod(height, count)
    tile_height = rows_per_section + (1 if longer_sections else 0)
    result = []
    for image in images:
        for idx in range(count):
            # Distribute any leftover rows among the earliest sections.
            # For example, H=10 and count=6 -> heights 2,2,2,2,1,1.
            start = idx * rows_per_section + min(idx, longer_sections)
            section_height = rows_per_section + (1 if idx < longer_sections else 0)
            tile = image[start:start + section_height]
            if tile.shape[0] < tile_height:
                # Duplicate the edge pixel row to keep all sections equal-height;
                # the full original image remains unchanged on the main output.
                padding = tile[-1:].expand(tile_height - tile.shape[0], -1, -1)
                tile = torch.cat([tile, padding], dim=0)
            result.append(tile)
    return torch.stack(result, dim=0).contiguous()


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
    CATEGORY = "Kling Image 4.0 REST/Images"

    def generate(self, auth, prompt, resolution, aspect_ratio, n, enable_unverified_rest_preview):
        body = {
            "contents": [{"type": "prompt", "text": _validate_prompt(prompt)}],
            "settings": {
                "result_type": "single",
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "n": n,
            },
        }
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
    CATEGORY = "Kling Image 4.0 REST/Images"

    def generate(self, auth, prompt, image, resolution, aspect_ratio, n,
                 enable_unverified_rest_preview, **kwargs):
        images = [image] + [kwargs.get(f"reference_{i}") for i in range(2, 11)]
        body = {
            "contents": [{"type": "prompt", "text": _validate_prompt(prompt)}] + _references(images),
            "settings": {
                "result_type": "single",
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "n": n,
            },
        }
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
    CATEGORY = "Kling Image 4.0 REST/Images"

    def generate(self, auth, prompt, resolution, aspect_ratio, series_amount,
                 enable_unverified_rest_preview, **kwargs):
        refs = _references([kwargs.get(f"reference_{i}") for i in range(1, 11)])
        body = {
            "contents": [{"type": "prompt", "text": _validate_prompt(prompt)}] + refs,
            "settings": {
                "result_type": "series",
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "series_amount": "auto" if series_amount == "auto" else int(series_amount),
            },
        }
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
        }, "optional": {
            **{f"reference_{i}": ("IMAGE",) for i in range(1, 4)},
            "section_count": ("INT", {
                "default": 6, "min": 2, "max": 18, "step": 1,
                "tooltip": "Locally split the returned long image into 2-18 individual sections. No extra Kling request.",
            }),
        }}
    # Append a new output to keep the existing 0-3 output indices stable.
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "urls_json", "task_id", "metadata_json", "sections")
    FUNCTION = "generate"
    CATEGORY = "Kling Image 4.0 REST/Images"

    def generate(self, auth, prompt, skill, length, resolution,
                 enable_unverified_rest_preview, section_count=6, **kwargs):
        refs = _references([kwargs.get(f"reference_{i}") for i in range(1, 4)], max_count=3)
        settings = {"skill": skill, "resolution": resolution}
        if skill == "ecommerce":
            if not 1 <= len(refs) <= 3:
                raise ValueError("Kling Image 4.0 e-commerce long image requires 1-3 product reference images.")
            settings["length"] = length
        else:
            if len(refs) > 2:
                raise ValueError("Kling Image 4.0 webpage accepts at most two references (style and subject).")
            if len(refs) >= 1:
                refs[0]["type"] = "style_refer_image"
            if len(refs) == 2:
                refs[1]["type"] = "subject_refer_image"
        body = {
            "contents": [{"type": "prompt", "text": _validate_prompt(prompt)}] + refs,
            "settings": settings,
        }
        full_image, urls_json, task_id, metadata_json = _run(
            auth, body, _preview_enabled(enable_unverified_rest_preview), LONG_IMAGE_PATH
        )
        sections = _split_long_image_batch(full_image, section_count)
        return full_image, urls_json, task_id, metadata_json, sections


class KlingImage4_SplitLongImage:
    """Re-split an existing long image locally, without another paid API call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "long_image": ("IMAGE",),
            "section_count": ("INT", {
                "default": 6, "min": 2, "max": 18, "step": 1,
                "tooltip": "Number of equally tall top-to-bottom crops in the IMAGE batch.",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("sections", "actual_sections")
    FUNCTION = "split"
    CATEGORY = "Kling Image 4.0 REST/Utilities"

    def split(self, long_image, section_count):
        pieces = _split_long_image_batch(long_image, section_count)
        return pieces, int(pieces.shape[0])


NODE_CLASS_MAPPINGS = {
    "KlingImage4Preview_TextToImage": KlingImage4Preview_TextToImage,
    "KlingImage4Preview_ImageEdit": KlingImage4Preview_ImageEdit,
    "KlingImage4Preview_Series": KlingImage4Preview_Series,
    "KlingImage4Preview_LongImage": KlingImage4Preview_LongImage,
    "KlingImage4_SplitLongImage": KlingImage4_SplitLongImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingImage4Preview_TextToImage": "Kling Image 4.0 REST - Text to Image",
    "KlingImage4Preview_ImageEdit": "Kling Image 4.0 REST - Image Edit",
    "KlingImage4Preview_Series": "Kling Image 4.0 REST - Image Series",
    "KlingImage4Preview_LongImage": "Kling Image 4.0 REST - Long Image",
    "KlingImage4_SplitLongImage": "Kling Image 4.0 - Split Long Image (Local)",
}

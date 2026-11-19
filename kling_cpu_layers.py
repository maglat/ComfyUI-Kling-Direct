"""CPU-only masking and layer compositing helpers for ComfyUI-Kling-Direct.

No Seedream API or GPU required. RemBG is OPTIONAL and imported only when
Remove Background runs. Image values are RGB and layer masks are separate:
ComfyUI IMAGE itself does not represent a true Photoshop layer stack.
"""
from __future__ import annotations

import io
from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageFilter


def _image_batch(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a ComfyUI IMAGE tensor.")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] not in (3, 4):
        raise ValueError("IMAGE must be [B,H,W,3/4].")
    return value.detach().cpu().float().clamp(0, 1)


def _mask_batch(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a ComfyUI MASK tensor.")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError("MASK must be [B,H,W].")
    return value.detach().cpu().float().clamp(0, 1)


def _pil_rgb(frame):
    rgb = (frame[..., :3].numpy() * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _pil_mask(frame):
    v = (frame.numpy() * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(v, "L")


def _fit_mask(mask, image):
    b, h, w = image.shape[:3]
    mb = mask.shape[0]
    if mb not in (1, b):
        raise ValueError(f"MASK batch size must be 1 or {b}, got {mb}.")
    result = []
    for k in range(b):
        m = mask[0 if mb == 1 else k]
        if tuple(m.shape) != (h, w):
            m = torch.from_numpy(
                np.asarray(_pil_mask(m).resize((w,h), Image.Resampling.BILINEAR),
                           dtype=np.float32).copy() / 255.0
            )
        result.append(m)
    return torch.stack(result)


def _straight_rgb(image, mask):
    # Don't premultiply soft edges; Composite multiplies alpha exactly ONCE.
    return torch.where(mask.unsqueeze(-1) > 0, image[..., :3], 0.0)


class KlingCPURemoveBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "model": (["u2net", "u2netp", "isnet-general-use"], {"default": "u2netp"}),
            "alpha_matting": ("BOOLEAN", {"default": False}),
            "mask_blur": ("INT", {"default": 0, "min": 0, "max": 32}),
            "threshold": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": 0.01}),
        }}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cutout", "mask")
    FUNCTION = "run"
    CATEGORY = "Kling CPU Layer Tools"

    def run(self, image, model="u2netp", alpha_matting=False, mask_blur=0, threshold=0.):
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise RuntimeError(
                "Optional dependency rembg is missing. Install "
                "ComfyUI-Kling-Direct/requirements-cpu-layers.txt "
                "into the ComfyUI Python environment."
            ) from exc
        images = _image_batch(image)
        session = new_session(model)
        cutouts, masks = [], []
        for frame in images:
            rgba = remove(_pil_rgb(frame), session=session, alpha_matting=bool(alpha_matting))
            if isinstance(rgba, bytes):
                rgba = Image.open(io.BytesIO(rgba))
            rgba = rgba.convert("RGBA")
            alpha = rgba.getchannel("A")
            if threshold > 0:
                alpha = alpha.point(lambda px: 255 if px/255. >= threshold else 0)
            if mask_blur > 0:
                alpha = alpha.filter(ImageFilter.GaussianBlur(radius=int(mask_blur)))
            m = torch.from_numpy(np.asarray(alpha, dtype=np.float32).copy() / 255.)
            rgb = torch.from_numpy(np.asarray(rgba.convert("RGB"), dtype=np.float32).copy() / 255.)
            cutouts.append(_straight_rgb(rgb.unsqueeze(0), m.unsqueeze(0))[0])
            masks.append(m)
        return torch.stack(cutouts), torch.stack(masks)


class KlingCPUExtractLayerFromMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "invert_mask": ("BOOLEAN", {"default": False}),
            "feather_radius": ("INT", {"default": 0, "min": 0, "max": 64}),
        }}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("layer", "layer_mask")
    FUNCTION = "run"
    CATEGORY = "Kling CPU Layer Tools"

    def run(self, image, mask, invert_mask=False, feather_radius=0):
        images = _image_batch(image)
        alpha = _fit_mask(_mask_batch(mask), images)
        if invert_mask:
            alpha = 1. - alpha
        if feather_radius:
            alpha = torch.stack([
                torch.from_numpy(
                    np.asarray(_pil_mask(m).filter(
                        ImageFilter.GaussianBlur(radius=int(feather_radius))),
                        dtype=np.float32).copy() / 255.)
                for m in alpha
            ])
        return _straight_rgb(images, alpha), alpha


class KlingCPUMoveLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "layer": ("IMAGE",),
            "mask": ("MASK",),
            "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
            "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
        }}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("moved_layer", "moved_mask")
    FUNCTION = "run"
    CATEGORY = "Kling CPU Layer Tools"

    def run(self, layer, mask, offset_x=0, offset_y=0):
        images = _image_batch(layer)[..., :3]
        alpha = _fit_mask(_mask_batch(mask), images)
        x, y = int(offset_x), int(offset_y)
        result_images, result_masks = [], []
        for frame, m in zip(images, alpha):
            h, w = frame.shape[:2]
            dst = torch.zeros_like(frame)
            dst_mask = torch.zeros_like(m)
            sx, sy = max(0, -x), max(0, -y)
            dx, dy = max(0, x), max(0, y)
            cw, ch = max(0, min(w-sx,w-dx)), max(0, min(h-sy,h-dy))
            if cw and ch:
                dst[dy:dy+ch, dx:dx+cw] = frame[sy:sy+ch, sx:sx+cw]
                dst_mask[dy:dy+ch, dx:dx+cw] = m[sy:sy+ch, sx:sx+cw]
            result_images.append(dst)
            result_masks.append(dst_mask)
        return torch.stack(result_images), torch.stack(result_masks)


class KlingCPUInpaintBackground:
    """OpenCV Telea approximation. Large removed subjects can produce artifacts."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "object_mask": ("MASK",),
            "mask_expand_px": ("INT", {"default": 8, "min": 0, "max": 128}),
            "inpaint_radius": ("INT", {"default": 5, "min": 1, "max": 20}),
        }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("background",)
    FUNCTION = "run"
    CATEGORY = "Kling CPU Layer Tools"

    def run(self, image, object_mask, mask_expand_px=8, inpaint_radius=5):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install repository requirements.txt (OpenCV).") from exc
        images = _image_batch(image)
        masks = _fit_mask(_mask_batch(object_mask), images)
        result = []
        for frame, alpha in zip(images, masks):
            rgb = np.asarray(_pil_rgb(frame)).copy()
            m = (alpha.numpy() > 0.1).astype(np.uint8) * 255
            expansion = int(mask_expand_px)
            if expansion:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2*expansion+1, 2*expansion+1))
                m = cv2.dilate(m, kernel)
            if m.any():
                rgb = cv2.inpaint(rgb, m, float(inpaint_radius), cv2.INPAINT_TELEA)
            result.append(torch.from_numpy(rgb.copy()).float().div_(255.))
        return (torch.stack(result),)


class KlingCPUCompositeLayers:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for k in range(1, 5):
            opt[f"layer_{k}"] = ("IMAGE",)
            opt[f"mask_{k}"] = ("MASK",)
        return {"required": {"background": ("IMAGE",)}, "optional": opt}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Kling CPU Layer Tools"

    def run(self, background, **kwargs):
        result = _image_batch(background)[..., :3].clone()
        b,h,w = result.shape[:3]
        for k in range(1, 5):
            layer = kwargs.get(f"layer_{k}")
            if layer is None:
                continue
            if kwargs.get(f"mask_{k}") is None:
                raise ValueError(f"mask_{k} required for layer_{k}. "
                                 "ComfyUI IMAGE has no built-in transparency.")
            frame = _image_batch(layer)[..., :3]
            if frame.shape[0] == 1 and b > 1:
                frame = frame.repeat(b,1,1,1)
            if frame.shape[:3] != (b,h,w):
                raise ValueError(f"layer_{k} size/batch must match background.")
            a = _fit_mask(_mask_batch(kwargs[f"mask_{k}"]), result).unsqueeze(-1)
            result = frame*a + result*(1-a)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "KlingCPURemoveBackground": KlingCPURemoveBackground,
    "KlingCPUExtractLayerFromMask": KlingCPUExtractLayerFromMask,
    "KlingCPUMoveLayer": KlingCPUMoveLayer,
    "KlingCPUInpaintBackground": KlingCPUInpaintBackground,
    "KlingCPUCompositeLayers": KlingCPUCompositeLayers,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingCPURemoveBackground": "Kling CPU - Remove Background",
    "KlingCPUExtractLayerFromMask": "Kling CPU - Extract Layer from Mask",
    "KlingCPUMoveLayer": "Kling CPU - Move Layer",
    "KlingCPUInpaintBackground": "Kling CPU - Inpaint Background (OpenCV)",
    "KlingCPUCompositeLayers": "Kling CPU - Composite Layers",
}

"""Offline CPU layer tests; no network or rembg model download required."""
import importlib
import numpy as np
import pytest
import torch

import kling_cpu_layers as m


def test_move_and_composite():
    image = torch.ones((1,8,8,3))
    mask = torch.zeros((1,8,8)); mask[:,2:4,2:4]=1
    layer, alpha = m.KlingCPUExtractLayerFromMask().run(image,mask)
    moved, shifted = m.KlingCPUMoveLayer().run(layer,alpha,2,1)
    out, = m.KlingCPUCompositeLayers().run(torch.zeros_like(image),layer_1=moved,mask_1=shifted)
    assert torch.allclose(out[0,3:5,4:6], torch.ones(2,2,3))
    assert out[0,2,2,0].item() == 0


def test_soft_alpha_is_not_double_multiplied():
    img = torch.ones((1,3,3,3))
    mask = torch.full((1,3,3),0.5)
    layer,alpha = m.KlingCPUExtractLayerFromMask().run(img,mask)
    out, = m.KlingCPUCompositeLayers().run(torch.zeros_like(img),layer_1=layer,mask_1=alpha)
    assert out[0,0,0,0].item()==pytest.approx(.5)


def test_missing_mask_fails_safe():
    with pytest.raises(ValueError,match="mask_1 required"):
        m.KlingCPUCompositeLayers().run(torch.zeros(1,3,3,3),layer_1=torch.ones(1,3,3,3))


def test_cpu_inpaint_empty_mask():
    image = torch.ones((1,8,8,3))*0.2
    out, = m.KlingCPUInpaintBackground().run(image,torch.zeros(1,8,8))
    assert torch.allclose(image,out,atol=1/255)


def test_registered_nodes():
    assert len(m.NODE_CLASS_MAPPINGS)==6
    for cls in m.NODE_CLASS_MAPPINGS.values():
        assert hasattr(cls, cls.FUNCTION)



def test_easy_layers_manual_mask_generates_native_compositor_document():
    source = torch.ones((1,6,6,3))
    source[:,:,:,1:] = 0.2
    mask = torch.zeros((1,6,6))
    mask[:,2:4,2:4] = 1.
    doc, preview, selection = m.KlingCPUEasyLayers().prepare(
        source, "Manual: painted mask", "Keep original background", mask=mask
    )
    assert doc["version"] == 1
    assert doc["canvas"] == (6,6)
    assert len(doc["layers"]) == 2
    assert doc["layers"][0]["name"].startswith("01")
    assert doc["layers"][1]["name"].startswith("02")
    assert torch.equal(doc["layers"][1]["image"], source)
    # Core compositor uses 1=transparent; ours returns 1=selected.
    assert torch.equal(doc["layers"][1]["mask"], 1-mask)
    assert torch.equal(selection, mask)
    assert torch.allclose(preview,source)


def test_easy_layers_manual_mode_requires_mask_without_loading_model(monkeypatch):
    monkeypatch.setattr(m.KlingCPURemoveBackground, "run",
                        lambda *args, **kwargs: pytest.fail("rembg should not run"))
    with pytest.raises(ValueError, match="painted MASK"):
        m.KlingCPUEasyLayers().prepare(
            torch.ones((1,8,8,3)),
            "Manual: painted mask", "Keep original background",
        )


def test_easy_layers_auto_mode_calls_rembg_only_once(monkeypatch):
    called = []
    def fake_rembg(self, source, **kwargs):
        called.append(tuple(source.shape))
        mask = torch.zeros(source.shape[:3])
        mask[:,1:3,2:4]=0.5
        return source, mask
    monkeypatch.setattr(m.KlingCPURemoveBackground,"run",fake_rembg)
    src=torch.ones((1,5,7,3))
    doc,preview,mask = m.KlingCPUEasyLayers().prepare(
        src, "Auto: product/person (CPU)", "Keep original background"
    )
    assert called == [(1,5,7,3)]
    assert doc["layers"][1]["mask"][0,1,2].item() == pytest.approx(0.5)
    assert tuple(preview.shape) == (1,5,7,3)


def test_easy_layers_can_pick_one_frame_from_image_series():
    src = torch.zeros((3,5,5,3))
    src[1] = 0.5
    mask = torch.ones((1,5,5))
    doc, preview, selected=m.KlingCPUEasyLayers().prepare(
        src, "Manual: painted mask", "Keep original background",
        mask=mask, image_number=2
    )
    assert doc["layers"][1]["image"][0,0,0,0].item()==pytest.approx(.5)
    assert preview.shape == (1,5,5,3)
    with pytest.raises(ValueError, match="outside the batch"):
        m.KlingCPUEasyLayers().prepare(
            src,"Manual: painted mask","Keep original background",
            mask=mask,image_number=4
        )


def test_easy_layers_repaired_background_is_in_stack(monkeypatch):
    bg = torch.zeros((1,6,6,3))
    monkeypatch.setattr(m.KlingCPUInpaintBackground,"run",
                        lambda *args, **kwargs: (bg,))
    source = torch.ones((1,6,6,3))
    mask = torch.zeros((1,6,6))
    mask[:,2:4,2:4]=1.
    doc,prev, _ = m.KlingCPUEasyLayers().prepare(
        source,"Manual: painted mask","Repair behind object (CPU)",mask
    )
    assert torch.equal(doc["layers"][0]["image"],bg)
    assert prev[0,2,2,0]==1.
    assert prev[0,0,0,0]==0.

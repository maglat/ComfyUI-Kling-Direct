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
    assert len(m.NODE_CLASS_MAPPINGS)==5
    for cls in m.NODE_CLASS_MAPPINGS.values():
        assert hasattr(cls, cls.FUNCTION)

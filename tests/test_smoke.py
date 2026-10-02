"""Fast checks that run without the dataset (python -m pytest)."""
import numpy as np
import pytest
import torch

from src.data import ArrayDataset, baseline_transform, pretrained_transform
from src.gradcam import GradCAM, overlay
from src.models import BACKBONES, MLP, DeepCNN, SimpleCNN, build_transfer_model


@pytest.mark.parametrize("model,size", [(MLP(64), 64), (SimpleCNN(), 64),
                                        (SimpleCNN(bn=True), 64), (DeepCNN(), 96)])
def test_baseline_output_shape(model, size):
    assert model(torch.rand(2, 3, size, size)).shape == (2, 6)


@pytest.mark.parametrize("name", ["mobilenet_v3_large", "resnet50"])
def test_transfer_model_heads(name):
    m = build_transfer_model(name, pretrained=False).eval()
    assert m(torch.rand(1, 3, 224, 224)).shape == (1, 6)
    f = build_transfer_model(name, pretrained=False, feature_only=True).eval()
    assert f(torch.rand(1, 3, 224, 224)).shape == (1, f.feat_dim)


def test_all_backbones_registered():
    assert set(BACKBONES) >= {"mobilenet_v3_large", "efficientnet_b0", "convnext_tiny"}


def test_transforms_and_dataset():
    imgs = np.random.randint(0, 256, (3, 256, 256, 3), dtype=np.uint8)
    ds = ArrayDataset(imgs, np.array([0, 1, 2]), pretrained_transform(224, augment=True))
    x, y = ds[1]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32 and y == 1
    x, _ = ArrayDataset(imgs, None, baseline_transform(64))[0]
    assert x.shape == (3, 64, 64) and 0 <= x.min() and x.max() <= 1


def test_gradcam_range_and_overlay():
    m = build_transfer_model("mobilenet_v3_large", pretrained=False).eval()
    cam, probs = GradCAM(m)(torch.rand(1, 3, 224, 224))
    assert cam.shape == (224, 224) and cam.min() >= 0 and cam.max() <= 1
    assert np.isclose(probs.sum(), 1, atol=1e-4)
    out = overlay(np.zeros((224, 224, 3), np.uint8), cam)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8

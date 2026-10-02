"""Grad-CAM: which pixels pushed the model towards its prediction?

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization", ICCV 2017.
"""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps


def default_target_layer(model):
    """Last convolutional stage of the supported torchvision backbones."""
    if hasattr(model, "features"):        # efficientnet / mobilenet / convnext
        return model.features[-1]
    if hasattr(model, "layer4"):          # resnet
        return model.layer4
    raise ValueError("Pass target_layer explicitly for this architecture.")


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        layer = target_layer or default_target_layer(model)
        self._acts = self._grads = None
        layer.register_forward_hook(self._save_acts)
        layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _module, _inp, out):
        self._acts = out.detach()

    def _save_grads(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """x: [1,3,H,W] normalized tensor. Returns (cam[H,W] in 0..1, probs[n_cls])."""
        x = x.requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, 1)[0].detach()
            idx = int(probs.argmax()) if class_idx is None else class_idx
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)          # GAP over space
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), probs.cpu().numpy()


def overlay(image_uint8, cam, alpha=0.45, cmap="jet"):
    """Blend a [H,W] CAM (0..1) onto an RGB uint8 image of the same size."""
    heat = (colormaps[cmap](cam)[..., :3] * 255).astype(np.float32)
    out = (1 - alpha) * image_uint8.astype(np.float32) + alpha * heat
    return out.clip(0, 255).astype(np.uint8)

"""Load a trained checkpoint and classify single images (used by the demo app)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import CACHE_SIZE, CLASSES, MODELS_DIR
from .data import pretrained_transform
from .gradcam import GradCAM, overlay
from .models import build_transfer_model


class Predictor:
    def __init__(self, ckpt_path=None):
        ckpt_path = Path(ckpt_path or MODELS_DIR / "recycle_classifier.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.img_size = ckpt["img_size"]
        self.model = build_transfer_model(ckpt["model"], pretrained=False)
        self.model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
        self.model.eval()
        self.tf = pretrained_transform(self.img_size)
        self.cam = GradCAM(self.model)

    def _prepare(self, image):
        """Same pre-processing as training: squash to 256x256, then resize to img_size."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img256 = np.array(image.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE),
                                                        Image.BILINEAR))
        x = self.tf(torch.from_numpy(img256).permute(2, 0, 1)).unsqueeze(0)
        return img256, x

    @torch.inference_mode()
    def predict(self, image):
        _, x = self._prepare(image)
        p = (torch.softmax(self.model(x), 1) +
             torch.softmax(self.model(torch.flip(x, dims=[3])), 1)) / 2
        return dict(zip(CLASSES, p[0].tolist()))

    def explain(self, image):
        """Return (probabilities dict, Grad-CAM overlay as uint8 RGB array)."""
        img256, x = self._prepare(image)
        cam, _ = self.cam(x)
        probs = self.predict(image)
        small = np.asarray(Image.fromarray(img256).resize((self.img_size, self.img_size)))
        return probs, overlay(small, cam)

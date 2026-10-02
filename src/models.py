"""Model zoo.

* Baseline models (MLP -> CNN -> DeepCNN) re-implement the step-by-step models from the
  original competition notebook so they can be re-evaluated on the same hold-out split.
* Transfer-learning models wrap ImageNet-pretrained torchvision backbones.
"""
import torch.nn as nn
import torchvision.models as tvm

N_CLASSES = 6


# ------------------------------------------------------------------ baselines
class MLP(nn.Module):
    """Step 0 - the starter code: flattened pixels -> fully connected layers."""

    def __init__(self, img_size=64, n_cls=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * img_size * img_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def _conv_block(c_in, c_out, bn):
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers += [nn.ReLU(), nn.MaxPool2d(2)]
    return layers


class SimpleCNN(nn.Module):
    """Steps 1-4 (3 conv blocks) and the final DeepCNN (4 conv blocks, up to 256 ch)."""

    def __init__(self, channels=(32, 64, 128), bn=False, p_drop=0.3, n_cls=N_CLASSES):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers += _conv_block(c_in, c, bn)
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(p_drop), nn.Linear(c_in, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def DeepCNN(n_cls=N_CLASSES):
    """The final competition model (garbage_classification_3.py)."""
    return SimpleCNN(channels=(32, 64, 128, 256), bn=True, n_cls=n_cls)


# ------------------------------------------------------------ transfer learning
# name -> (constructor, weights, attribute path of the final Linear layer)
BACKBONES = {
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, "IMAGENET1K_V2", ("classifier", 3)),
    "efficientnet_b0": (tvm.efficientnet_b0, "IMAGENET1K_V1", ("classifier", 1)),
    "efficientnet_v2_s": (tvm.efficientnet_v2_s, "IMAGENET1K_V1", ("classifier", 1)),
    "resnet50": (tvm.resnet50, "IMAGENET1K_V2", ("fc",)),
    "convnext_tiny": (tvm.convnext_tiny, "IMAGENET1K_V1", ("classifier", 2)),
    "swin_t": (tvm.swin_t, "IMAGENET1K_V1", ("head",)),
}


def _get_parent_and_key(model, path):
    parent = model
    for p in path[:-1]:
        parent = getattr(parent, p) if isinstance(p, str) else parent[p]
    return parent, path[-1]


def _set(parent, key, module):
    if isinstance(key, str):
        setattr(parent, key, module)
    else:
        parent[key] = module


def build_transfer_model(name, n_cls=N_CLASSES, pretrained=True, feature_only=False):
    """ImageNet backbone with its last Linear replaced.

    feature_only=True replaces it with Identity (penultimate-layer embeddings),
    otherwise with a fresh Linear(in_features, n_cls).
    """
    ctor, weights, path = BACKBONES[name]
    model = ctor(weights=weights if pretrained else None)
    parent, key = _get_parent_and_key(model, path)
    old = parent[key] if isinstance(key, int) else getattr(parent, key)
    new = nn.Identity() if feature_only else nn.Linear(old.in_features, n_cls)
    _set(parent, key, new)
    model.feat_dim = old.in_features
    return model

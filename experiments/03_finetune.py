"""Experiment 3 - Fine-tune an ImageNet-pretrained backbone end-to-end.

Recipe: AdamW, 1-epoch warm-up + cosine decay, lower LR for the pretrained backbone
than for the new head, label smoothing 0.1, inverse-frequency class weights (for the
small `trash` class) and stronger augmentation (random resized crop, h/v flips,
rotation, color jitter). The number of epochs is fixed in advance and the final-epoch
weights are used, so the hold-out split is never used for model selection.

    python -m experiments.03_finetune --model efficientnet_b0 --epochs 10
"""
import argparse
import json

import torch

from src.config import MODELS_DIR, RESULTS_DIR
from src.data import ArrayDataset, holdout_split, load_test, load_train, pretrained_transform
from src.engine import class_weights, fit, predict_proba, seed_everything
from src.metrics import save_proba, save_result, summarize
from src.models import BACKBONES, build_transfer_model


def head_and_backbone_params(model, name):
    path = BACKBONES[name][2]
    head = model
    for p in path:
        head = getattr(head, p) if isinstance(p, str) else head[p]
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return list(head.parameters()), backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0", choices=list(BACKBONES))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per split")
    args = ap.parse_args()

    seed_everything()
    x, y, _ = load_train()
    x_test, _ = load_test()
    dev, hold = holdout_split(y)
    if args.limit:
        dev, hold, x_test = dev[::len(dev) // args.limit], hold[::len(hold) // args.limit], x_test[:8]

    model = build_transfer_model(args.model)
    head, backbone = head_and_backbone_params(model, args.model)
    groups = [{"params": backbone, "lr": args.lr * args.backbone_lr_mult},
              {"params": head, "lr": args.lr}]

    train_ds = ArrayDataset(x[dev], y[dev], pretrained_transform(args.size, augment=True))
    hold_ds = ArrayDataset(x[hold], y[hold], pretrained_transform(args.size))
    tag = f"ft_{args.model}" + ("_debug" if args.limit else "")
    log_file = RESULTS_DIR / f"{tag}_history.jsonl"
    log_file.unlink(missing_ok=True)

    model, info = fit(model, train_ds, hold_ds, epochs=args.epochs, lr=args.lr,
                      batch_size=args.batch, optimizer="adamw",
                      weight_decay=args.weight_decay, scheduler="cosine", warmup_epochs=1,
                      class_w=None if args.no_class_weights else class_weights(y[dev], 6),
                      label_smoothing=args.label_smoothing, param_groups=groups,
                      select_best=False, name=tag, log_file=log_file)

    proba = predict_proba(model, hold_ds, tta_hflip=True)
    save_result("finetune", tag, summarize(y[hold], proba), config=vars(args))
    save_proba(f"{tag}_holdout", proba)
    test_ds = ArrayDataset(x_test, None, pretrained_transform(args.size))
    save_proba(f"{tag}_test", predict_proba(model, test_ds, tta_hflip=True))

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"model": args.model, "img_size": args.size,
                "state_dict": {k: v.half() if v.is_floating_point() else v
                               for k, v in model.state_dict().items()}},
               MODELS_DIR / f"{tag}.pt")
    (MODELS_DIR / f"{tag}.json").write_text(json.dumps(vars(args), indent=2))
    print(f"saved {MODELS_DIR / (tag + '.pt')}")


if __name__ == "__main__":
    main()

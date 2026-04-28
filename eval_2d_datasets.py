import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch
from tqdm import tqdm

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils import calculate_metric_percase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TransUNet model on a 2D PNG/JPG dataset "
                    "and compute Dice, HD95 and their 95% confidence intervals."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained model checkpoint (.pth).")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory of input images (PNG/JPG).")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory of corresponding masks (PNG/JPG).")
    parser.add_argument("--dataset_name", type=str, default="testset",
                        help="Name of this dataset (for logging).")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Network input size (same as training, e.g. 224).")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (binary=2).")
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16",
                        help="ViT backbone name (must match training).")
    parser.add_argument("--n_skip", type=int, default=3,
                        help="Number of skip connections (must match training).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'. Default: cuda if available.")
    return parser.parse_args()


def find_same_stem_file(directory: str, stem: str,
                        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> str:
    """Find a file in directory whose name matches stem with any of the given extensions."""
    candidates: List[str] = []
    for ext in exts:
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p):
            candidates.append(p)
        else:
            p_upper = os.path.join(directory, stem + ext.upper())
            if os.path.exists(p_upper):
                candidates.append(p_upper)
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Mask not found for stem='{stem}' in '{directory}'. "
            f"Expected one of: {', '.join([stem + e for e in exts])}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple masks found for stem='{stem}': {candidates}. Keep exactly one."
        )
    return candidates[0]


def mean_ci95(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, lower_95, upper_95) assuming normal approximation."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    margin = 1.96 * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / 16),
            int(args.img_size / 16),
        )
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    state = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def evaluate_dataset(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    net = load_model(args, device)

    exts = (".png", ".jpg", ".jpeg")
    img_files = sorted(
        f for f in os.listdir(args.img_dir) if f.lower().endswith(exts)
    )

    dice_list: List[float] = []
    hd95_list: List[float] = []

    for fname in tqdm(img_files, desc=f"Evaluating {args.dataset_name}"):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(args.img_dir, fname)
        mask_path = find_same_stem_file(args.mask_dir, stem, exts)

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path)

        img_np = np.array(img, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.int32)
        # assume binary mask encoded as 0 / 255 or 0 / 1
        mask_np = (mask_np > 0).astype(np.uint8)

        h, w = img_np.shape
        if h != args.img_size or w != args.img_size:
            img_resized = zoom(
                img_np,
                (args.img_size / h, args.img_size / w),
                order=3,
            )
        else:
            img_resized = img_np

        input_tensor = (
            torch.from_numpy(img_resized)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        with torch.no_grad():
            logits = net(input_tensor)
            pred_resized = (
                torch.argmax(torch.softmax(logits, dim=1), dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

        if h != args.img_size or w != args.img_size:
            pred = zoom(
                pred_resized,
                (h / args.img_size, w / args.img_size),
                order=0,
            ).astype(np.uint8)
        else:
            pred = pred_resized

        d, h95 = calculate_metric_percase(pred, mask_np)
        dice_list.append(float(d))
        hd95_list.append(float(h95))

    dice_mean, dice_l, dice_u = mean_ci95(dice_list)
    hd95_mean, hd95_l, hd95_u = mean_ci95(hd95_list)

    print(f"Dataset: {args.dataset_name}")
    print(f"  Num cases: {len(dice_list)}")
    print(
        f"  Dice:  mean={dice_mean:.4f}, 95% CI=({dice_l:.4f}, {dice_u:.4f})"
    )
    print(
        f"  HD95:  mean={hd95_mean:.4f}, 95% CI=({hd95_l:.4f}, {hd95_u:.4f})"
    )


if __name__ == "__main__":
    args = parse_args()
    evaluate_dataset(args)


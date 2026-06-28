# infer_dir.py — 批量推理：输入图像目录，输出所有掩码到指定输出目录
import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch
from tqdm import tqdm

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def parse_args():
    parser = argparse.ArgumentParser("TransUNet batch inference on an image directory")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="输入图像目录 (包含 png/jpg 文件)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="输出掩码保存目录 (文件名与输入一致，扩展名为 .png)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="网络输入尺寸 (训练时用的尺寸，例如 224)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数（二分类=2）")
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16",
                        help="ViT 骨干名称 (需与训练一致)")
    parser.add_argument("--n_skip", type=int, default=3,
                        help="skip 连接数量 (需与训练一致)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' 或 'cpu'，默认优先用 cuda")
    return parser.parse_args()


def collect_images(img_dir):
    """收集目录下所有图像文件 (png/jpg/jpeg)，返回排序后的 (stem, filepath) 列表"""
    exts = (".png", ".jpg", ".jpeg")
    files = []
    for fname in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in exts:
            files.append((stem, os.path.join(img_dir, fname)))
    return files


def load_model(args, device):
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


def infer_one(net, img_path, img_size, device):
    """对单张图像推理，返回原图尺寸的预测掩码 (numpy uint8)"""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32)
    h, w = img_np.shape

    # resize 到网络输入大小
    if h != img_size or w != img_size:
        img_resized = zoom(img_np, (img_size / h, img_size / w), order=3)
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

    # resize 回原图大小
    if h != img_size or w != img_size:
        pred = zoom(pred_resized, (h / img_size, w / img_size), order=0).astype(np.uint8)
    else:
        pred = pred_resized

    return pred


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    net = load_model(args, device)

    files = collect_images(args.img_dir)
    if not files:
        print(f"[ERROR] No images found in: {args.img_dir}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    for stem, img_path in tqdm(files, desc="Inferring"):
        pred = infer_one(net, img_path, args.img_size, device)
        mask = Image.fromarray((pred > 0).astype(np.uint8) * 255)
        out_path = os.path.join(args.out_dir, stem + ".png")
        mask.save(out_path)

    print(f"Done. {len(files)} masks saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

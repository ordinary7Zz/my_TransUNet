# infer_one.py
import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def parse_args():
    parser = argparse.ArgumentParser("TransUNet single-image inference")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--img_path", type=str, required=True,
                        help="输入图像路径 (png/jpg)")
    parser.add_argument("--out_path", type=str, required=True,
                        help="输出掩码保存路径 (png)")
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


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    net = load_model(args, device)

    # 1. 读入图像，转灰度
    img = Image.open(args.img_path).convert("L")
    img_np = np.array(img, dtype=np.float32)
    h, w = img_np.shape

    # 2. resize 到网络输入大小
    if h != args.img_size or w != args.img_size:
        img_resized = zoom(
            img_np,
            (args.img_size / h, args.img_size / w),
            order=3,
        )
    else:
        img_resized = img_np

    # 3. 转成 tensor，送入网络
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

    # 4. resize 回原图大小
    if h != args.img_size or w != args.img_size:
        pred = zoom(
            pred_resized,
            (h / args.img_size, w / args.img_size),
            order=0,
        ).astype(np.uint8)
    else:
        pred = pred_resized

    # 5. 保存掩码：0/1 -> 0/255，单通道 PNG
    mask_img = Image.fromarray((pred > 0).astype(np.uint8) * 255)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    mask_img.save(args.out_path)
    print(f"Saved mask to: {args.out_path}")


if __name__ == "__main__":
    main()
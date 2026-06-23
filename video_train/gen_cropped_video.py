"""
Generate video showing what the model actually sees during training:
Reads already-cropped images from images_cropped/, resizes to square, saves mp4.

Usage:
    python gen_cropped_video.py                        # all sessions
    python gen_cropped_video.py --session 0212_193402   # single session
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms


DATA_ROOT = Path(r"D:\Image_DP\data\GelloAgent")
IMG_SIZE = 128
FPS = 10  # downsampled from 30hz by 3x


def make_resize_transform(img_size: int):
    """Same as gello_video_dataset.py _default_img_transform, but no ToTensor/Normalize."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    ])


def gen_video(session_dir: Path, out_path: Path, img_size: int, fps: int):
    # Read from downsampled CSV to get images_cropped/ paths
    csv_path = session_dir / "joint_with_images_downsampled.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "image_relpath" in df.columns:
            img_paths = [session_dir / str(x) for x in df["image_relpath"].tolist()]
        else:
            # fallback to images_cropped folder
            cropped_dir = session_dir / "images_cropped"
            if not cropped_dir.exists():
                print(f"[SKIP] No images_cropped/ in {session_dir.name}")
                return
            img_paths = sorted(cropped_dir.glob("*.jpg"))
    else:
        cropped_dir = session_dir / "images_cropped"
        if not cropped_dir.exists():
            print(f"[SKIP] No images_cropped/ in {session_dir.name}")
            return
        img_paths = sorted(cropped_dir.glob("*.jpg"))

    if len(img_paths) == 0:
        print(f"[SKIP] No images found in {session_dir.name}")
        return

    resize_tf = make_resize_transform(img_size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (img_size, img_size))

    for p in img_paths:
        pil_img = Image.open(p).convert("RGB")
        pil_resized = resize_tf(pil_img)  # (224, 224)
        bgr = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"[DONE] {out_path.name}  ({len(img_paths)} frames, {len(img_paths)/fps:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None,
                        help="Single session folder name, e.g. 0212_193402")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: each session folder)")
    args = parser.parse_args()

    if args.session:
        sessions = [DATA_ROOT / args.session]
    else:
        sessions = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])

    for sess_dir in sessions:
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{sess_dir.name}_cropped.mp4"
        else:
            out_path = sess_dir / "cropped.mp4"

        gen_video(sess_dir, out_path, args.img_size, args.fps)


if __name__ == "__main__":
    main()

import os
import io
import json
import re
import argparse
from PIL import Image
from tqdm import tqdm
import webdataset as wds

DEFAULT_WDS_ROOT = r"D:\Deduplication_framework_wds_shards"


def extract_label(filepath, num_classes):
    try:
        parent = os.path.basename(os.path.dirname(filepath))
        if parent.isdigit():
            return int(parent)
    except Exception:
        pass
    # fallback for legacy naming
    try:
        filename = os.path.basename(filepath)
        match = re.search(r'_(\d+)(?:_aug|\.)', filename)
        if match:
            return int(match.group(1)) % num_classes
    except Exception:
        pass
    return 0


def load_image_paths(root_dir, json_path):
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths


def build_one(root, json_path, out_dir, shard_size, resize, num_classes, quality):
    os.makedirs(out_dir, exist_ok=True)

    image_paths = load_image_paths(root, json_path)
    total = len(image_paths)
    if total == 0:
        print(f"[ERROR] No images found for {json_path or 'no_dedup'}. Skipped.")
        return

    pattern = os.path.join(out_dir, "shard-%06d.tar")
    pattern_url = "file:" + pattern.replace("\\", "/")
    count = 0
    errors = 0

    with wds.ShardWriter(pattern_url, maxcount=shard_size) as sink:
        for idx, path in enumerate(tqdm(image_paths, desc=f"Building shards -> {os.path.basename(out_dir)}")):
            try:
                img = Image.open(path).convert("RGB")
                if resize:
                    img = img.resize((resize, resize), Image.BILINEAR)

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                img_bytes = buffer.getvalue()

                label = extract_label(path, num_classes)
                sample = {
                    "__key__": f"{idx:09d}",
                    "jpg": img_bytes,
                    "cls": str(label),
                }
                sink.write(sample)
                count += 1
            except Exception:
                errors += 1

    index = {
        "count": count,
        "errors": errors,
        "resize": resize,
        "num_classes": num_classes,
        "source": json_path if json_path else root,
    }
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {count} samples to {out_dir} (errors: {errors})")


def main():
    parser = argparse.ArgumentParser(description="Build WebDataset shards for fast training.")
    parser.add_argument("--root", required=True, help="Raw image root directory")
    parser.add_argument("--json", default=None, help="Keep-list JSON (absolute path)")
    parser.add_argument("--json-list", nargs="*", help="Multiple keep-list JSON paths")
    parser.add_argument("--include-no-dedup", action="store_true", help="Also build no_dedup shards")
    parser.add_argument("--out", default=None, help="Output shard directory (single mode)\n建议放到: D:\\Deduplication_framework_wds_shards")
    parser.add_argument("--out-root", default=DEFAULT_WDS_ROOT, help="Output root for batch mode")
    parser.add_argument("--shard-size", type=int, default=10000, help="Samples per shard")
    parser.add_argument("--resize", type=int, default=224, help="Resize to NxN before saving")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = parser.parse_args()

    if args.json_list or args.include_no_dedup:
        os.makedirs(args.out_root, exist_ok=True)

        if args.include_no_dedup:
            out_dir = os.path.join(args.out_root, "no_dedup")
            build_one(args.root, None, out_dir, args.shard_size, args.resize, args.num_classes, args.quality)

        if args.json_list:
            for json_path in args.json_list:
                name = os.path.splitext(os.path.basename(json_path))[0]
                out_dir = os.path.join(args.out_root, name)
                build_one(args.root, json_path, out_dir, args.shard_size, args.resize, args.num_classes, args.quality)
        return

    if not args.out:
        raise SystemExit("[ERROR] --out is required in single mode.")

    build_one(args.root, args.json, args.out, args.shard_size, args.resize, args.num_classes, args.quality)


if __name__ == "__main__":
    main()

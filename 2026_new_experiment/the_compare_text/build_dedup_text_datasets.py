import os
import csv
import glob
import hashlib
from typing import Dict, List

import pandas as pd
from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from tqdm import tqdm

# ================= 配置区域 =================
DATA_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_text"
OUTPUT_DIR = r"D:\Deduplication_framework_text_deduped"
TEXT_COLUMN = "content"
LABEL_COLUMN = "label"
TITLE_COLUMN = "title"
CHUNK_SIZE = 10000
MAX_FILES = int(os.environ.get("MAX_FILES", "10"))

# 算法参数
NUM_PERM = 128
THRESHOLD = 0.8
SIMHASH_DIST = 10
SIMHASH_WINDOW = 200
# 速度优化：限制特征数量
MAX_WORDS = 200
MAX_CHAR_GRAMS = 200
MAX_SIMHASH_TOKENS = 200
# ===========================================

METHODS = [
    "no_dedup",
    "md5",
    "simhash",
    "minhash_lsh",
    "ours_lsh",
]
# 可选：只运行部分方法
ENABLE_METHODS = set(METHODS)


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def char_3grams(text: str):
    t_clean = text.replace(" ", "")
    if len(t_clean) < 3:
        return []
    return [t_clean[i:i + 3] for i in range(len(t_clean) - 2)]


def get_input_files() -> List[str]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "part_*.csv")))
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
    return files


def init_outputs() -> Dict[str, csv.writer]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writers = {}
    for method in METHODS:
        out_path = os.path.join(OUTPUT_DIR, f"dedup_{method}.csv")
        f = open(out_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=[LABEL_COLUMN, TITLE_COLUMN, TEXT_COLUMN])
        writer.writeheader()
        writers[method] = (f, writer)
    return writers


def _build_output_paths() -> Dict[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return {m: os.path.join(OUTPUT_DIR, f"dedup_{m}.csv") for m in METHODS}


def close_outputs(writers: Dict[str, csv.writer]):
    for f, _ in writers.values():
        f.close()


def run():
    files = get_input_files()
    if not files:
        print("[ERROR] No input files found.")
        return

    writers = init_outputs()
    out_paths = _build_output_paths()
    print("[Info] Enabled methods and output files:")
    for m in METHODS:
        enabled = "(ENABLED)" if m in ENABLE_METHODS else "(disabled)"
        print(f"  - {m}: {out_paths[m]} {enabled}")

    md5_seen = set()
    simhash_seen = []
    lsh_std = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    lsh_ours = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)

    counts = {m: 0 for m in METHODS}

    for path in tqdm(files, desc="Files"):
        try:
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[File] {os.path.basename(path)} - size {file_size_mb:.1f} MB")
            reader = pd.read_csv(
                path,
                usecols=[LABEL_COLUMN, TITLE_COLUMN, TEXT_COLUMN],
                on_bad_lines="skip",
                encoding="utf-8",
                chunksize=CHUNK_SIZE,
            )
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue

        # 使用手动 chunk 计数的 tqdm，确保在每个 chunk 处理后立即刷新
        chunks_pbar = tqdm(desc=f"Chunks {os.path.basename(path)}", unit='chunk', leave=False)
        processed_rows = 0
        for chunk in reader:
            for row in chunk.itertuples(index=False):
                label = getattr(row, LABEL_COLUMN)
                title = getattr(row, TITLE_COLUMN)
                text = getattr(row, TEXT_COLUMN)
                if pd.isna(text):
                    continue

                text_clean = clean_text(text)
                record = {
                    LABEL_COLUMN: label,
                    TITLE_COLUMN: title,
                    TEXT_COLUMN: text,
                }

                # no_dedup
                if "no_dedup" in ENABLE_METHODS:
                    writers["no_dedup"][1].writerow(record)
                    counts["no_dedup"] += 1
                    if counts["no_dedup"] % 1000 == 0:
                        print(f"[Write] no_dedup: {counts['no_dedup']} rows written -> {out_paths['no_dedup']}")

                # md5
                if "md5" in ENABLE_METHODS:
                    md5_val = hashlib.md5(text_clean.encode("utf-8")).hexdigest()
                    if md5_val not in md5_seen:
                        md5_seen.add(md5_val)
                        writers["md5"][1].writerow(record)
                        counts["md5"] += 1
                        if counts["md5"] % 1000 == 0:
                            print(f"[Write] md5: {counts['md5']} rows written -> {out_paths['md5']}")

                # simhash
                if "simhash" in ENABLE_METHODS:
                    tokens = text_clean.split()[:MAX_SIMHASH_TOKENS]
                    if not tokens:
                        continue
                    curr = Simhash(tokens)
                    is_dup = False
                    window = simhash_seen[-SIMHASH_WINDOW:] if len(simhash_seen) > SIMHASH_WINDOW else simhash_seen
                    for seen in window:
                        if curr.distance(seen) <= SIMHASH_DIST:
                            is_dup = True
                            break
                    if not is_dup:
                        simhash_seen.append(curr)
                        writers["simhash"][1].writerow(record)
                        counts["simhash"] += 1
                        if counts["simhash"] % 1000 == 0:
                            print(f"[Write] simhash: {counts['simhash']} rows written -> {out_paths['simhash']}")

                # minhash_lsh (char 3-gram)
                if "minhash_lsh" in ENABLE_METHODS:
                    m = MinHash(num_perm=NUM_PERM)
                    grams = char_3grams(text_clean)[:MAX_CHAR_GRAMS]
                    for g in grams:
                        m.update(g.encode("utf-8"))
                    if len(lsh_std.query(m)) == 0:
                        lsh_std.insert(f"doc_{counts['minhash_lsh']}", m)
                        writers["minhash_lsh"][1].writerow(record)
                        counts["minhash_lsh"] += 1
                        if counts["minhash_lsh"] % 1000 == 0:
                            print(f"[Write] minhash_lsh: {counts['minhash_lsh']} rows written -> {out_paths['minhash_lsh']}")

                # ours_lsh (word + char 3-gram)
                if "ours_lsh" in ENABLE_METHODS:
                    m2 = MinHash(num_perm=NUM_PERM)
                    for w in text_clean.split()[:MAX_WORDS]:
                        m2.update(w.encode("utf-8"))
                    for g in char_3grams(text_clean)[:MAX_CHAR_GRAMS]:
                        m2.update(g.encode("utf-8"))
                    if len(lsh_ours.query(m2)) == 0:
                        lsh_ours.insert(f"doc_{counts['ours_lsh']}", m2)
                        writers["ours_lsh"][1].writerow(record)
                        counts["ours_lsh"] += 1
                        if counts["ours_lsh"] % 1000 == 0:
                            print(f"[Write] ours_lsh: {counts['ours_lsh']} rows written -> {out_paths['ours_lsh']}")
            # 本 chunk 处理完，更新计数与进度条，并偶尔打印总体进度以避免长时间无输出
            processed_rows += len(chunk)
            chunks_pbar.update(1)
            try:
                # 全局累计行数（用于较长运行的周期性汇报）
                global_processed = globals().get("_GLOBAL_TEXT_ROWS_PROCESSED", 0) + len(chunk)
                globals()['_GLOBAL_TEXT_ROWS_PROCESSED'] = global_processed
                if global_processed % 50000 < len(chunk):
                    print(f"[Progress] processed ~{global_processed} rows so far (file={os.path.basename(path)})")
            except Exception:
                pass
        chunks_pbar.close()

    close_outputs(writers)

    print("\n=== Dedup Summary ===")
    for m in METHODS:
        print(f"{m}: {counts[m]}")
    # 列出最终输出文件与大小
    try:
        print("\nFinal output files:")
        for m in METHODS:
            p = out_paths.get(m)
            if p and os.path.exists(p):
                size_mb = os.path.getsize(p) / (1024 * 1024)
                print(f"  - {m}: {p} ({size_mb:.2f} MB, rows={counts[m]})")
            else:
                print(f"  - {m}: {p} (not created, rows={counts[m]})")
    except Exception:
        pass
    # 写入去重汇总（CSV + JSON），包含每个方法保留数与去重率
    try:
        total_rows = globals().get("_GLOBAL_TEXT_ROWS_PROCESSED", 0)
        if not total_rows:
            # fallback: use no_dedup count if available
            total_rows = counts.get("no_dedup", sum(counts.values()) or 0)

        summary = []
        for m in METHODS:
            kept = counts.get(m, 0)
            dedup_rate = 1.0 - (kept / total_rows) if total_rows else 0.0
            summary.append({
                "method": m,
                "kept": kept,
                "total_input_rows": total_rows,
                "dedup_rate": round(dedup_rate, 6),
                "output_file": out_paths.get(m),
            })

        # CSV
        summary_csv = os.path.join(OUTPUT_DIR, "dedup_summary.csv")
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "kept", "total_input_rows", "dedup_rate", "output_file"])
            writer.writeheader()
            for row in summary:
                writer.writerow(row)

        # JSON
        import json as _json
        summary_json = os.path.join(OUTPUT_DIR, "dedup_summary.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            _json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[Summary] wrote dedup summary CSV: {summary_csv}")
        print(f"[Summary] wrote dedup summary JSON: {summary_json}")
    except Exception as exc:
        print(f"[WARN] Failed to write summary: {exc}")


if __name__ == "__main__":
    run()

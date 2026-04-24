import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path(r"D:\Deduplication_framework\2026_new_experiment\pipeline_ablation")
CODE_DIR = ROOT / "code"
CONFIG_DIR = CODE_DIR / "configs"
RESULT_DIR = ROOT / "result"
ARTIFACTS_DIR = ROOT / "artifacts"

DEFAULT_INPUT_ROOT = Path(r"D:\Deduplication_framework\test_dataset")
DEFAULT_BASE_CONFIG = Path(r"D:\Deduplication_framework\configs\my_pipeline_smoke.yaml")

PIPELINE_ENTRY = ["python", "-m", "pipelines"]

DEFAULT_ENVS = {
    "sorter": "Deplication_Framework",
    "image": "image",
    "audio": "audio",
    "text": "text-dedup",
}


def load_executor_defaults() -> Dict[str, Any]:
    base_config = Path(os.environ.get("ABLATION_BASE_CONFIG", str(DEFAULT_BASE_CONFIG)))
    if not base_config.exists():
        return {
            "conda_executable": None,
            "envs": dict(DEFAULT_ENVS),
        }
    if yaml is None:
        raise RuntimeError("pyyaml is required to read base config for env defaults")
    with base_config.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    executor = data.get("executor") or {}
    conda_executable = executor.get("conda_executable")
    envs = executor.get("envs") or {}
    merged_envs = dict(DEFAULT_ENVS)
    if isinstance(envs, dict):
        merged_envs.update({k: v for k, v in envs.items() if v})
    return {
        "conda_executable": conda_executable,
        "envs": merged_envs,
    }


def _resolve_input_root() -> Path:
    input_root = os.environ.get("ABLATION_INPUT_ROOT")
    return Path(input_root) if input_root else DEFAULT_INPUT_ROOT


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("pyyaml is required to write config files")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def build_image_config(save_embeddings_dir: Path, eps: float) -> Dict[str, Any]:
    return {
        "embedding": {
            "backend": "open_clip",
            "model_name": "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
            "batch_size": 256,
            "device": "auto",
            "fallback": "average_rgb",
            "save_embeddings_dir": str(save_embeddings_dir),
        },
        "dedup": {
            "method": "semdedup",
            "eps": eps,
            "legacy_keep_indices_file": None,
            "legacy_cluster_dir": None,
            "max_candidates": 50000,
        },
    }


def build_audio_config(method: str) -> Dict[str, Any]:
    return {
        "embedding": {
            "fingerprint_backend": "compute",
            "threshold": 0.75,
        },
        "dedup": {
            "method": method,
            "threshold": 0.75,
            "max_candidates": 50000,
            "mfcc_threshold": 0.95,
            "lsh_b": 20,
            "lsh_r": 10,
            "lsh_collision_threshold": 2,
        },
        "general": {
            "manifest_check": {
                "max_missing_fraction": 0.10,
                "max_missing_count": 1000,
                "sample_limit": 500,
            }
        }
    }


def build_text_config(method: str) -> Dict[str, Any]:
    return {
        "embedding": {
            "ngram_size": 3,
            "lowercase": True,
            "strip_non_alnum": True,
            "collapse_whitespace": True,
        },
        "dedup": {
            "method": method,
            "threshold": 0.8,
            "max_candidates": 200000,
            "num_perm": 128,
            "max_words": 200,
            "max_char_grams": 200,
            "simhash_dist": 10,
            "simhash_window": 1000,
        },
    }


def build_pipeline_config(
    name: str,
    output_root: Path,
    input_root: Path,
    sorter_enabled: bool,
    image_enabled: bool,
    audio_enabled: bool,
    text_enabled: bool,
    near_dedup_enabled: bool,
    conda_executable: str | None,
    envs: Dict[str, str],
) -> Dict[str, Any]:
    image_eps = 0.07 if near_dedup_enabled else 0.0
    audio_method = "lsh" if near_dedup_enabled else "md5"
    text_method = "ours_lsh" if near_dedup_enabled else "md5"

    image_config_path = CONFIG_DIR / f"{name}_image.yaml"
    audio_config_path = CONFIG_DIR / f"{name}_audio.yaml"
    text_config_path = CONFIG_DIR / f"{name}_text.yaml"

    write_yaml(image_config_path, build_image_config(output_root / "embeddings", image_eps))
    write_yaml(audio_config_path, build_audio_config(audio_method))
    write_yaml(text_config_path, build_text_config(text_method))

    return {
        "general": {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "temp_root": str(ARTIFACTS_DIR),
            "resume": False,
            "parallel_modalities": False,
            "parallel_workers": 1,
            "batch_size": 2000,
        },
        "logging": {
            "level": "INFO",
            "file": str(output_root / "pipeline.log"),
        },
        "network": {
            "http_read_timeout_seconds": 30,
            "retries": 5,
            "backoff_factor_seconds": 1,
        },
        "executor": {
            "type": "local",
            "conda_executable": conda_executable,
            "envs": dict(envs),
        },
        "data_quality": {
            "include_unusable_bins": False,
        },
        "sorter": {
            "enabled": sorter_enabled,
            "manifest_name": "manifest.csv",
            "batch_size": 2000,
            "move_files": False,
        },
        "image": {
            "enabled": image_enabled,
            "entrypoint": r"D:/Deduplication_framework/pipelines/modalities/image_runner.py",
            "workdir": ".",
            "output_dir": str(output_root / "image"),
            "args": [],
            "env": {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "PIPELINE_IMAGE_EXPORT_JSON": str(output_root / "image_keep_list.json"),
            },
            "config_file": str(image_config_path),
            "batch_size": 0,
            "max_workers": 4,
            "manifest_subset_count": 0,
            "flush_interval": 500,
        },
        "audio": {
            "enabled": audio_enabled,
            "entrypoint": r"D:/Deduplication_framework/pipelines/modalities/audio_runner.py",
            "workdir": ".",
            "output_dir": str(output_root / "audio"),
            "args": [],
            "env": {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "PIPELINE_FORCE_CONDA_RUN": "1",
            },
            "config_file": str(audio_config_path),
            "batch_size": 500,
            "manifest_subset_count": 5000,
            "max_workers": 4,
            "retry_on_decode_error": True,
            "decode_retry_limit": 3,
        },
        "text": {
            "enabled": text_enabled,
            "entrypoint": r"D:/Deduplication_framework/pipelines/modalities/text_runner.py",
            "workdir": ".",
            "output_dir": str(output_root / "text"),
            "args": [],
            "env": {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            },
            "config_file": str(text_config_path),
            "batch_size": 2000,
            "max_workers": 4,
            "flush_interval": 200,
            "window_size": 500,
        },
        "report": {
            "summary_file": str(output_root / "summary.json"),
            "markdown_file": str(output_root / "report.md"),
            "save_intermediate": True,
            "intermediate_dir": str(output_root / "intermediate"),
        },
    }


def run_pipeline(config_path: Path) -> None:
    cmd = PIPELINE_ENTRY + ["--config", str(config_path)]
    subprocess.check_call(cmd)


def read_summary(summary_path: Path) -> Dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _modality_metrics(summary: Dict[str, Any], modality: str) -> Dict[str, Any]:
    aggregated = summary.get("aggregated") or {}
    mod = (aggregated.get("modalities") or {}).get(modality) or {}
    stats = mod.get("stats") or {}
    return {
        "enabled": mod.get("enabled"),
        "processed": mod.get("processed") or mod.get("files"),
        "total_candidates": stats.get("total_candidates"),
        "duplicates": stats.get("duplicates"),
        "unique": stats.get("unique"),
        "missing": stats.get("missing"),
        "dedup_rate": mod.get("deduplication_rate"),
        "unique_ratio": mod.get("unique_ratio"),
    }


def _weighted_ratio(entries: List[Dict[str, Any]], numerator_key: str, denominator_key: str) -> float | None:
    total_num = 0.0
    total_den = 0.0
    for e in entries:
        num = e.get(numerator_key)
        den = e.get(denominator_key)
        if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den > 0:
            total_num += float(num)
            total_den += float(den)
    if total_den <= 0:
        return None
    return total_num / total_den


def _resolve_manifest_path(summary: Dict[str, Any]) -> Path | None:
    sorter_manifest = summary.get("sorter_manifest") or {}
    manifest_path = sorter_manifest.get("manifest_path")
    if manifest_path:
        return Path(manifest_path)
    run_manifest_path = summary.get("run_manifest_path")
    if run_manifest_path:
        run_dir = Path(run_manifest_path).parent
        candidate = run_dir / "stage1_sorter" / "manifest.csv"
        if candidate.exists():
            return candidate
    return None


def _load_manifest_sizes(manifest_path: Path) -> Dict[str, int]:
    import csv
    sizes: Dict[str, int] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("source_path")
            size = row.get("size_bytes")
            if not path:
                continue
            try:
                sizes[path] = int(size) if size not in (None, "") else 0
            except Exception:
                sizes[path] = 0
    return sizes


def _collect_duplicate_paths(duplicates_files: List[str]) -> List[str]:
    dup_paths: List[str] = []
    for dup_file in duplicates_files:
        if not dup_file:
            continue
        path = Path(dup_file)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            duplicates = item.get("duplicates") if isinstance(item, dict) else None
            if not isinstance(duplicates, list):
                continue
            for dup in duplicates:
                if isinstance(dup, dict) and dup.get("path"):
                    dup_paths.append(dup["path"])
    return dup_paths


def _compute_storage_save_bytes(summary: Dict[str, Any]) -> int | None:
    manifest_path = _resolve_manifest_path(summary)
    if manifest_path is None:
        return None
    size_map = _load_manifest_sizes(manifest_path)
    dup_paths: List[str] = []
    for stage in summary.get("stages", []):
        output_paths = stage.get("output_paths") or {}
        dup_files = output_paths.get("duplicates") or []
        if isinstance(dup_files, list):
            dup_paths.extend(_collect_duplicate_paths(dup_files))
    if not dup_paths:
        return 0
    total = 0
    for p in set(dup_paths):
        if p in size_map:
            total += size_map[p]
        else:
            try:
                total += Path(p).stat().st_size
            except Exception:
                continue
    return total


def build_report(rows: List[Dict[str, Any]]) -> None:
    out_csv = RESULT_DIR / "ablation_summary.csv"
    out_md = RESULT_DIR / "ablation_summary.md"

    headers = [
        "configuration",
        "status",
        "run_id",
        "total_inputs",
        "elapsed_minutes",
        "overall_dedup_rate",
        "overall_unique_ratio",
        "storage_save_bytes",
        "storage_save_mb",
        "image_dedup_rate",
        "audio_dedup_rate",
        "text_dedup_rate",
        "image_processed",
        "audio_processed",
        "text_processed",
        "image_missing",
        "audio_missing",
        "text_missing",
        "missing_total",
        "notes",
    ]

    # CSV
    import csv
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in headers})

    # Markdown
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        line = []
        for k in headers:
            v = row.get(k)
            if isinstance(v, float):
                if k in {"overall_dedup_rate", "overall_unique_ratio", "image_dedup_rate", "audio_dedup_rate", "text_dedup_rate"}:
                    line.append(f"{v:.4f}")
                elif k in {"storage_save_mb"}:
                    line.append(f"{v:.2f}")
                else:
                    line.append(f"{v:.2f}")
            elif v is None:
                line.append("-")
            else:
                line.append(str(v))
        md_lines.append("| " + " | ".join(line) + " |")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline ablation or rebuild summary")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild summary from existing runs")
    args = parser.parse_args()

    ensure_dirs()

    input_root = _resolve_input_root()
    if not input_root.exists():
        raise RuntimeError(f"Input root not found: {input_root}")

    dataset_note = input_root.name

    executor_defaults = load_executor_defaults()
    conda_executable = executor_defaults.get("conda_executable")
    envs = executor_defaults.get("envs", dict(DEFAULT_ENVS))
    if conda_executable:
        conda_path = Path(conda_executable)
        if not conda_path.exists():
            raise RuntimeError(f"Conda executable not found: {conda_path}")

    runs = [
        #{
        #     "name": "full",
        #     "label": "Full Pipeline (sorter+image+audio+text)",
        #     "sorter": True,
        #     "image": True,
        #     "audio": True,
        #     "text": True,
        #     "near": True,
        #     "notes": dataset_note,
        # },
        # {
        #     "name": "w_o_sorter",
        #     "label": "w/o Sorter",
        #     "sorter": False,
        #     "image": True,
        #     "audio": True,
        #     "text": True,
        #     "near": True,
        #     "notes": "reuses previous manifest",
        # },
        # {
        #     "name": "w_o_near_dedup",
        #     "label": "w/o Near-Dedup",
        #     "sorter": True,
        #     "image": True,
        #     "audio": True,
        #     "text": True,
        #     "near": False,
        #     "notes": "image eps=0, audio/text md5",
        # },
        # {
        #     "name": "w_o_image",
        #     "label": "w/o Image Dedup",
        #     "sorter": True,
        #     "image": False,
        #     "audio": True,
        #     "text": True,
        #     "near": True,
        #     "notes": "image disabled",
        # },
        # {
        #     "name": "w_o_audio",
        #     "label": "w/o Audio Dedup",
        #     "sorter": True,
        #     "image": True,
        #     "audio": False,
        #     "text": True,
        #     "near": True,
        #     "notes": "audio disabled",
        # },
        # {
        #     "name": "w_o_text",
        #     "label": "w/o Text Dedup",
        #     "sorter": True,
        #     "image": True,
        #     "audio": True,
        #     "text": False,
        #     "near": True,
        #     "notes": "text disabled",
        # },
        {
            "name": "image_only",
            "label": "Image Dedup Only",
            "sorter": False,   # 建议保持 True，sorter 通常负责整理文件清单(manifest)
            "image": True,    # 开启图像去重
            "audio": False,   # 关闭音频去重
            "text": False,    # 关闭文本去重
            "near": True,     # True 表示使用近似去重 (CLIP + SemDedup)，False 则退化为 MD5/精确去重
            "notes": "Only image dedup enabled",
        },
        {
            "name": "audio_only",
            "label": "Audio Dedup Only",
            "sorter": False,   # 建议保持 True，sorter 通常负责整理文件清单(manifest)
            "image": False,    # 关闭图像去重
            "audio": True,    # 开启音频去重
            "text": False,    # 关闭文本去重
            "near": True,     # True 表示使用近似去重 (CLIP + SemDedup)，False 则退化为 MD5/精确去重
            "notes": "Only audio dedup enabled",
        },
        {
            "name": "text_only",
            "label": "Text Dedup Only",
            "sorter": False,   # 建议保持 True，sorter 通常负责整理文件清单(manifest)
            "image": False,    # 关闭图像去重
            "audio": False,   # 关闭音频去重
            "text": True,    # 开启文本去重
            "near": True,     # True 表示使用近似去重 (CLIP + SemDedup)，False 则退化为 MD5/精确去重
            "notes": "Only text dedup enabled",
        },                      
    ]

    results: List[Dict[str, Any]] = []

    for run in runs:
        output_root = RESULT_DIR / run["name"]
        output_root.mkdir(parents=True, exist_ok=True)
        config = build_pipeline_config(
            name=run["name"],
            output_root=output_root,
            input_root=input_root,
            sorter_enabled=run["sorter"],
            image_enabled=run["image"],
            audio_enabled=run["audio"],
            text_enabled=run["text"],
            near_dedup_enabled=run["near"],
            conda_executable=conda_executable,
            envs=envs,
        )
        config_path = CONFIG_DIR / f"{run['name']}.yaml"
        write_yaml(config_path, config)

        if not args.summary_only:
            print(f"[ablation] Running {run['label']} -> {config_path}", flush=True)
            run_pipeline(config_path)

        summary_path = output_root / "summary.json"
        if summary_path.exists():
            summary = read_summary(summary_path)
            aggregated = summary.get("aggregated") or {}
            overall = aggregated.get("overall") or {}
            elapsed_seconds = overall.get("modalities_elapsed_seconds")
            elapsed_minutes = (float(elapsed_seconds) / 60.0) if elapsed_seconds else None

            storage_save_bytes = _compute_storage_save_bytes(summary)
            storage_save_mb = (storage_save_bytes / (1024 * 1024)) if storage_save_bytes is not None else None

            image_metrics = _modality_metrics(summary, "image")
            audio_metrics = _modality_metrics(summary, "audio")
            text_metrics = _modality_metrics(summary, "text")
            missing_total = 0.0
            for m in (image_metrics, audio_metrics, text_metrics):
                if isinstance(m.get("missing"), (int, float)):
                    missing_total += float(m["missing"])
            if missing_total == 0:
                missing_total = None
            overall_dedup_rate = _weighted_ratio(
                [image_metrics, audio_metrics, text_metrics],
                "duplicates",
                "total_candidates",
            )
            overall_unique_ratio = _weighted_ratio(
                [image_metrics, audio_metrics, text_metrics],
                "unique",
                "total_candidates",
            )

            results.append(
                {
                    "configuration": run["label"],
                    "status": "ok",
                    "run_id": summary.get("run_id"),
                    "total_inputs": overall.get("total_inputs"),
                    "elapsed_minutes": elapsed_minutes,
                    "overall_dedup_rate": overall_dedup_rate,
                    "overall_unique_ratio": overall_unique_ratio,
                    "storage_save_bytes": storage_save_bytes,
                    "storage_save_mb": storage_save_mb,
                    "image_dedup_rate": image_metrics.get("dedup_rate"),
                    "audio_dedup_rate": audio_metrics.get("dedup_rate"),
                    "text_dedup_rate": text_metrics.get("dedup_rate"),
                    "image_processed": image_metrics.get("processed"),
                    "audio_processed": audio_metrics.get("processed"),
                    "text_processed": text_metrics.get("processed"),
                    "image_missing": image_metrics.get("missing"),
                    "audio_missing": audio_metrics.get("missing"),
                    "text_missing": text_metrics.get("missing"),
                    "missing_total": missing_total,
                    "notes": run["notes"],
                }
            )
        else:
            results.append(
                {
                    "configuration": run["label"],
                    "status": "missing_summary",
                    "run_id": None,
                    "total_inputs": None,
                    "elapsed_minutes": None,
                    "overall_dedup_rate": None,
                    "overall_unique_ratio": None,
                    "storage_save_bytes": None,
                    "storage_save_mb": None,
                    "image_dedup_rate": None,
                    "audio_dedup_rate": None,
                    "text_dedup_rate": None,
                    "image_processed": None,
                    "audio_processed": None,
                    "text_processed": None,
                    "image_missing": None,
                    "audio_missing": None,
                    "text_missing": None,
                    "missing_total": None,
                    "notes": run["notes"],
                }
            )

    build_report(results)
    print(f"[ablation] Summary written to {RESULT_DIR / 'ablation_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

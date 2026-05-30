#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Official-library evaluation script for low-light image enhancement.

Metrics:
  Full-reference:
    PSNR  - skimage.metrics.peak_signal_noise_ratio
    SSIM  - skimage.metrics.structural_similarity
    MSE   - skimage.metrics.mean_squared_error
    MAE   - numpy.mean(abs(x-y))
    LPIPS - pyiqa

  No-reference:
    NIQE    - pyiqa
    BRISQUE - pyiqa
    PI      - pyiqa
    MUSIQ   - pyiqa, optional
    NIMA    - pyiqa, optional

This script intentionally does not implement complex IQA metrics manually.
LOE is not included because there is no stable mainstream official-library
implementation in scikit-image / pyiqa. If needed, add LOE separately and
clearly mark it as a lightweight/custom implementation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

# torch / pyiqa are optional — metrics using them will be skipped if unavailable
_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    pass

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def imread_rgb_float(path: Union[str, Path]) -> np.ndarray:
    """Read image as RGB float32 in [0, 1]."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def ensure_same_shape(gt: np.ndarray, pred: np.ndarray, resize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure GT and predicted image have the same shape."""
    if gt.shape == pred.shape:
        return gt, pred
    if not resize:
        raise ValueError(
            f"Shape mismatch: gt={gt.shape}, pred={pred.shape}. "
            f"Use --resize for quick testing only."
        )
    h, w = gt.shape[:2]
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
    return gt, pred


def np_to_tensor(img: np.ndarray, device) -> "torch.Tensor":
    """RGB numpy [H,W,3], [0,1] -> torch [1,3,H,W]."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available on this system")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {img.shape}")
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)


def compute_skimage_fr(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute PSNR / SSIM / MSE / MAE."""
    return {
        "PSNR(dB)": float(peak_signal_noise_ratio(gt, pred, data_range=1.0)),
        "SSIM": float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0)),
        "MSE": float(mean_squared_error(gt, pred)),
        "MAE": float(np.mean(np.abs(gt - pred))),
    }


class PyiqaRunner:
    """Lazy pyiqa metric runner. Gracefully handles missing torch."""

    def __init__(self, device: Optional[str] = None):
        if not _TORCH_AVAILABLE:
            self.device = None
            self.cache = {}
            return
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cache = {}

    @property
    def available(self) -> bool:
        return _TORCH_AVAILABLE

    def metric(self, name: str):
        name = name.lower()
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is not available")
        if name not in self.cache:
            try:
                import pyiqa
            except ImportError as exc:
                raise ImportError(
                    "pyiqa is required. Install with: pip install pyiqa torch torchvision"
                ) from exc
            self.cache[name] = pyiqa.create_metric(name, device=self.device)
        return self.cache[name]

    def compute_fr(self, name: str, pred: np.ndarray, gt: np.ndarray) -> float:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch unavailable")
        model = self.metric(name)
        pred_t = np_to_tensor(pred, self.device)
        gt_t = np_to_tensor(gt, self.device)
        with torch.no_grad():
            score = model(pred_t, gt_t)
        return float(score.detach().cpu().reshape(-1)[0])

    def compute_nr(self, name: str, pred: np.ndarray) -> float:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch unavailable")
        model = self.metric(name)
        pred_t = np_to_tensor(pred, self.device)
        with torch.no_grad():
            score = model(pred_t)
        return float(score.detach().cpu().reshape(-1)[0])


def evaluate_pair(
    enhanced_path: Union[str, Path],
    gt_path: Optional[Union[str, Path]] = None,
    fr_metrics: Iterable[str] = ("lpips",),
    nr_metrics: Iterable[str] = ("niqe", "brisque", "pi"),
    resize: bool = False,
    pyiqa_runner: Optional[PyiqaRunner] = None,
) -> Dict[str, Union[float, str]]:
    """Evaluate one enhanced image."""
    enhanced_path = Path(enhanced_path)
    pred = imread_rgb_float(enhanced_path)
    results: Dict[str, Union[float, str]] = {}

    if pyiqa_runner is None:
        pyiqa_runner = PyiqaRunner()

    if gt_path is not None:
        gt_path = Path(gt_path)
        gt = imread_rgb_float(gt_path)
        gt, pred_fr = ensure_same_shape(gt, pred, resize=resize)
        results.update(compute_skimage_fr(pred_fr, gt))

        if pyiqa_runner.available:
            for m in fr_metrics:
                key = m.upper()
                try:
                    results[key] = pyiqa_runner.compute_fr(m, pred_fr, gt)
                except Exception as exc:
                    results[key] = f"N/A: {type(exc).__name__}"
        else:
            for m in fr_metrics:
                results[m.upper()] = "N/A (torch unavailable)"

    if pyiqa_runner.available:
        for m in nr_metrics:
            key = m.upper()
            try:
                results[key] = pyiqa_runner.compute_nr(m, pred)
            except Exception as exc:
                results[key] = f"N/A: {type(exc).__name__}"
    else:
        for m in nr_metrics:
            results[m.upper()] = "N/A (torch unavailable)"

    return results


def list_images(directory: Union[str, Path]) -> List[Path]:
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def average_results(rows: List[Dict[str, Union[float, str]]], keys: List[str]) -> Dict[str, Union[float, str]]:
    avg = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        avg[k] = float(np.mean(vals)) if vals else ""
    return avg


def evaluate_folder(
    enhanced_dir: Union[str, Path],
    gt_dir: Optional[Union[str, Path]],
    method_name: Optional[str],
    fr_metrics: Iterable[str],
    nr_metrics: Iterable[str],
    resize: bool,
    runner: PyiqaRunner,
) -> Dict[str, Union[str, int, float]]:
    """Evaluate one folder and return one CSV row."""
    enhanced_dir = Path(enhanced_dir)
    gt_dir = Path(gt_dir) if gt_dir else None
    images = list_images(enhanced_dir)
    if not images:
        return {"Method": method_name or enhanced_dir.name, "Count": 0}

    metric_keys = ["PSNR(dB)", "SSIM", "MSE", "MAE"]
    metric_keys += [m.upper() for m in fr_metrics]
    metric_keys += [m.upper() for m in nr_metrics]

    per_image_results = []
    valid_count = 0

    for img_path in images:
        gt_path = gt_dir / img_path.name if gt_dir else None
        if gt_path is not None and not gt_path.exists():
            print(f"[Skip] Missing GT: {gt_path}", file=sys.stderr)
            continue

        try:
            r = evaluate_pair(
                enhanced_path=img_path,
                gt_path=gt_path,
                fr_metrics=fr_metrics,
                nr_metrics=nr_metrics,
                resize=resize,
                pyiqa_runner=runner,
            )
            per_image_results.append(r)
            valid_count += 1
        except Exception as exc:
            print(f"[Skip] {img_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    avg = average_results(per_image_results, metric_keys)
    row: Dict[str, Union[str, int, float]] = {"Method": method_name or enhanced_dir.name, "Count": valid_count}
    row.update(avg)
    return row


def write_csv(rows: List[Dict[str, Union[str, int, float]]], output_csv: Union[str, Path]) -> None:
    """Write rows to CSV with README/MD-friendly column order."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "Method", "Count", "PSNR(dB)", "SSIM", "MSE", "MAE",
        "LPIPS", "NIQE", "BRISQUE", "PI", "MUSIQ", "NIMA",
    ]

    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    fieldnames = [k for k in preferred if k in all_keys] + [k for k in all_keys if k not in preferred]

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            formatted = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, float):
                    if k in {"PSNR(dB)", "SSIM"}:
                        formatted[k] = f"{v:.4f}"
                    elif k in {"MSE", "MAE"}:
                        formatted[k] = f"{v:.6f}"
                    else:
                        formatted[k] = f"{v:.4f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)


def print_rows(rows: List[Dict[str, Union[str, int, float]]]) -> None:
    """Print compact result table to terminal."""
    for r in rows:
        print("\n" + "=" * 60)
        print(f"Method: {r.get('Method')}")
        print(f"Count : {r.get('Count')}")
        print("-" * 60)
        for k, v in r.items():
            if k in {"Method", "Count"}:
                continue
            if isinstance(v, float):
                if k in {"MSE", "MAE"}:
                    print(f"{k:10s}: {v:.6f}")
                else:
                    print(f"{k:10s}: {v:.4f}")
            else:
                print(f"{k:10s}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official-library metrics for low-light image enhancement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python evaluate.py -e test_data/enhanced/method_a/scene1.png -g test_data/gt/scene1.png
  python evaluate.py --enhanced_dir test_data/enhanced/method_a --gt_dir test_data/gt
  python evaluate.py --models_root test_data/enhanced --gt_dir test_data/gt
  python evaluate.py --models_root test_data/enhanced --gt_dir test_data/gt --fr --nr
""",
    )

    parser.add_argument("-e", "--enhanced", help="Enhanced image path for single-image evaluation.")
    parser.add_argument("-g", "--gt", help="GT image path for single-image evaluation.")

    parser.add_argument("--enhanced_dir", help="Enhanced image folder for one-method evaluation.")
    parser.add_argument("--models_root", help="Root folder. Each subfolder is treated as one method/model.")
    parser.add_argument("--gt_dir", help="GT image folder. Filenames should match enhanced images.")

    parser.add_argument(
        "--fr",
        nargs="*",
        default=["lpips"],
        help="Full-reference pyiqa metrics. Default: lpips. Use '--fr' with no values to disable.",
    )
    parser.add_argument(
        "--nr",
        nargs="*",
        default=["niqe", "brisque", "pi"],
        help="No-reference pyiqa metrics. Default: niqe brisque pi. Example: --nr niqe brisque pi musiq nima",
    )

    parser.add_argument("--device", default=None, help="cuda / cpu. Default: auto.")
    parser.add_argument("--resize", action="store_true", help="Resize enhanced image to GT size. Use only for quick checks.")
    parser.add_argument("--output_csv", default="metrics_summary.csv", help="Output CSV path.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not _TORCH_AVAILABLE:
        print("[WARN] torch is not available on this system.", file=sys.stderr)
        print("       skimage metrics (PSNR/SSIM/MSE/MAE) will still work.", file=sys.stderr)
        print("       pyiqa metrics (LPIPS/NIQE/BRISQUE/PI/MUSIQ/NIMA) will be skipped.", file=sys.stderr)
        print("       To fix: install PyTorch from https://pytorch.org/get-started/locally/", file=sys.stderr)
        print(file=sys.stderr)

    runner = PyiqaRunner(device=args.device)

    if args.enhanced:
        result = evaluate_pair(
            enhanced_path=args.enhanced,
            gt_path=args.gt,
            fr_metrics=args.fr,
            nr_metrics=args.nr,
            resize=args.resize,
            pyiqa_runner=runner,
        )
        row = {"Method": Path(args.enhanced).stem, "Count": 1}
        row.update(result)
        print_rows([row])
        write_csv([row], args.output_csv)
        print(f"\nSaved CSV: {args.output_csv}")
        return

    rows = []
    if args.models_root:
        root = Path(args.models_root)
        model_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if not model_dirs:
            raise RuntimeError(f"No model subfolders found in {root}")
        for model_dir in model_dirs:
            print(f"[Eval] {model_dir.name}")
            rows.append(
                evaluate_folder(
                    enhanced_dir=model_dir,
                    gt_dir=args.gt_dir,
                    method_name=model_dir.name,
                    fr_metrics=args.fr,
                    nr_metrics=args.nr,
                    resize=args.resize,
                    runner=runner,
                )
            )
    elif args.enhanced_dir:
        rows.append(
            evaluate_folder(
                enhanced_dir=args.enhanced_dir,
                gt_dir=args.gt_dir,
                method_name=Path(args.enhanced_dir).name,
                fr_metrics=args.fr,
                nr_metrics=args.nr,
                resize=args.resize,
                runner=runner,
            )
        )
    else:
        raise RuntimeError("Please provide --enhanced, --enhanced_dir, or --models_root.")

    print_rows(rows)
    write_csv(rows, args.output_csv)
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

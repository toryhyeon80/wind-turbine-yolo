"""
runs/detect/ 최신 학습 결과를 스캔하여 report.md를 자동 갱신합니다.

갱신 항목:
  - 섹션 1 성능 비교 표 (최종 모델 mAP, Epoch, 성능 향상)
  - 섹션 3 Loss/mAP 그래프·검증 예측 이미지 마크다운 삽입

사용 예:
  python update_report.py
  python update_report.py --run-dir runs/detect/train
  python update_report.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "report.md"
DEFAULT_RUNS_DIR = ROOT / "runs" / "detect"

METRICS_START = "<!-- report:auto:metrics-visuals -->"
METRICS_END = "<!-- /report:auto:metrics-visuals -->"
PREDICTIONS_START = "<!-- report:auto:predictions -->"
PREDICTIONS_END = "<!-- /report:auto:predictions -->"
RUN_SUMMARY_START = "<!-- report:auto:run-summary -->"
RUN_SUMMARY_END = "<!-- /report:auto:run-summary -->"

METRICS_TIP_PATTERN = re.compile(
    r"^\s*_\(\s*💡\s*팁:.*?results\.png.*?\)_\s*$",
    re.MULTILINE,
)
PREDICTIONS_TIP_PATTERN = re.compile(
    r"^- \*\*Dirt\(오염\) 및 Damage\(손상\) 탐지 결과\*\*\s*\n\s*_\(\s*💡\s*팁:.*?val_batch0_pred\.jpg.*?\)_\s*$",
    re.MULTILINE,
)

FINAL_MODEL_ROW_PATTERN = re.compile(
    r"^\|\s*\*\*최종 모델\*\*\s*\|.*\|$",
    re.MULTILINE,
)
BASELINE_ROW_PATTERN = re.compile(
    r"\|\s*\*\*Baseline\*\*\s*\|[^|]+\|\s*\d+\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
)
IMPROVEMENT_ROW_PATTERN = re.compile(
    r"(\|\s*\*\*성능 향상\*\*\s*\|[^|]+\|[^|]+\|)\s*\*\*\+ [0-9.]+%p\*\*(\s*\|)\s*\*\*\+ [0-9.]+%p\*\*(\s*\|)",
)


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    run_name: str
    model: str
    epochs_configured: int
    best_epoch: int
    map50: float
    map50_95: float
    updated_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 학습 결과를 report.md에 반영합니다.")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="갱신할 리포트 경로 (기본: report.md)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="YOLO detect 실행 폴더 (기본: runs/detect)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="특정 run 폴더 지정 (미지정 시 최신 results.csv 기준)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 저장 없이 변경 내용만 출력",
    )
    return parser.parse_args()


def _pick_metric(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return float(row[key])
    return None


def find_latest_run_dir(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = [
        p.parent
        for p in runs_dir.rglob("results.csv")
        if p.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_args_yaml(run_dir: Path) -> dict:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return {}
    with args_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_metrics_from_csv(results_csv: Path) -> tuple[int, float, float]:
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"results.csv가 비어 있습니다: {results_csv}")

    best = max(
        rows,
        key=lambda r: _pick_metric(r, "metrics/mAP50(B)", "mAP50", "map50") or 0.0,
    )
    map50 = _pick_metric(best, "metrics/mAP50(B)", "mAP50", "map50")
    map50_95 = _pick_metric(best, "metrics/mAP50-95(B)", "mAP50-95", "map50-95")
    if map50 is None or map50_95 is None:
        raise ValueError(f"mAP 컬럼을 찾을 수 없습니다: {results_csv}")

    epoch = int(float(best.get("epoch", len(rows))))
    return epoch, map50, map50_95


def format_model_name(model_path: str) -> str:
    name = Path(model_path).stem.lower()
    mapping = {
        "yolo11n": "YOLO11n (Nano)",
        "yolo11s": "YOLO11s (Small)",
        "yolo11m": "YOLO11m (Medium)",
        "yolov8n": "YOLOv8n (Nano)",
        "yolov8s": "YOLOv8s (Small)",
        "yolov8m": "YOLOv8m (Medium)",
    }
    return mapping.get(name, model_path)


def collect_run_result(run_dir: Path) -> RunResult:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"results.csv를 찾을 수 없습니다: {results_csv}\n"
            "학습이 완료된 run 폴더를 지정하거나 학습을 먼저 실행해 주세요."
        )

    args = load_args_yaml(run_dir)
    best_epoch, map50, map50_95 = load_metrics_from_csv(results_csv)
    model = str(args.get("model", "yolo11s.pt"))
    epochs_configured = int(args.get("epochs", best_epoch))

    return RunResult(
        run_dir=run_dir.resolve(),
        run_name=run_dir.name,
        model=model,
        epochs_configured=epochs_configured,
        best_epoch=best_epoch,
        map50=map50,
        map50_95=map50_95,
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def to_repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def build_image_lines(run_dir: Path, files: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    for filename, caption in files:
        image_path = run_dir / filename
        if image_path.exists():
            rel = to_repo_relative(image_path)
            lines.append(f"![{caption}]({rel})")
    return lines


def replace_block(content: str, start: str, end: str, body: str) -> str:
    block = f"{start}\n{body}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(content):
        return pattern.sub(block, content, count=1)

    tip_pattern = METRICS_TIP_PATTERN if start == METRICS_START else PREDICTIONS_TIP_PATTERN
    if tip_pattern.search(content):
        return tip_pattern.sub(block, content, count=1)

    return content


def update_metrics_table(content: str, result: RunResult) -> str:
    model_label = format_model_name(result.model)
    map50_text = f"{result.map50:.3f}"
    map50_95_text = f"{result.map50_95:.3f}"
    note = f"자동 반영 ({result.run_name}, epoch {result.best_epoch})"
    new_row = (
        f"| **최종 모델** | {model_label} | {result.best_epoch} | "
        f"{map50_text} | {map50_95_text} | {note} |"
    )

    updated, count = FINAL_MODEL_ROW_PATTERN.subn(new_row, content, count=1)
    if count == 0:
        print("[경고] '최종 모델' 표 행을 찾지 못했습니다.", file=sys.stderr)
        return content

    baseline_match = BASELINE_ROW_PATTERN.search(updated)
    if baseline_match:
        base_map50 = float(baseline_match.group(1))
        base_map50_95 = float(baseline_match.group(2))
        gain50 = (result.map50 - base_map50) * 100
        gain95 = (result.map50_95 - base_map50_95) * 100
        updated = IMPROVEMENT_ROW_PATTERN.sub(
            rf"\1 **+ {gain50:.1f}%p**\2 **+ {gain95:.1f}%p**\3",
            updated,
            count=1,
        )
    return updated


def update_run_summary(content: str, result: RunResult) -> str:
    body = (
        f"- **최신 학습 실행:** `{result.run_name}` ({result.updated_at})\n"
        f"- **모델:** {format_model_name(result.model)} | "
        f"**Best Epoch:** {result.best_epoch} | "
        f"**mAP50:** {result.map50:.3f} | **mAP50-95:** {result.map50_95:.3f}\n"
        f"- **결과 폴더:** `{to_repo_relative(result.run_dir)}/`"
    )
    block = f"{RUN_SUMMARY_START}\n{body}\n{RUN_SUMMARY_END}"
    pattern = re.compile(
        re.escape(RUN_SUMMARY_START) + r".*?" + re.escape(RUN_SUMMARY_END),
        re.DOTALL,
    )
    if pattern.search(content):
        return pattern.sub(block, content, count=1)

    marker = "### [정성적 분석]\n\n"
    if marker in content:
        return content.replace(marker, f"{marker}{block}\n\n", 1)
    return content


def update_visual_sections(content: str, result: RunResult) -> str:
    metrics_images = build_image_lines(
        result.run_dir,
        [
            ("results.png", "Loss/mAP 학습 곡선"),
            ("confusion_matrix.png", "Confusion Matrix"),
            ("BoxF1_curve.png", "Box F1 Curve"),
        ],
    )
    prediction_images = build_image_lines(
        result.run_dir,
        [
            ("val_batch0_pred.jpg", "검증 예측 결과 1"),
            ("val_batch1_pred.jpg", "검증 예측 결과 2"),
            ("val_batch2_pred.jpg", "검증 예측 결과 3"),
        ],
    )

    if metrics_images:
        metrics_body = (
            f"- **자동 반영:** `{result.run_name}` ({result.updated_at})\n\n"
            + "\n\n".join(metrics_images)
        )
        content = replace_block(content, METRICS_START, METRICS_END, metrics_body)
    else:
        print("[경고] results.png 등 메트릭 이미지를 찾지 못했습니다.", file=sys.stderr)

    if prediction_images:
        predictions_body = (
            f"- **Dirt(오염) 및 Damage(손상) 탐지 결과** — `{result.run_name}`\n\n"
            + "\n\n".join(prediction_images)
        )
        content = replace_block(content, PREDICTIONS_START, PREDICTIONS_END, predictions_body)
    else:
        print("[경고] val_batch*_pred.jpg 예측 이미지를 찾지 못했습니다.", file=sys.stderr)

    return content


def apply_report_updates(content: str, result: RunResult) -> str:
    updated = update_metrics_table(content, result)
    updated = update_run_summary(updated, result)
    updated = update_visual_sections(updated, result)
    return updated


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()

    if not report_path.exists():
        raise SystemExit(f"리포트 파일을 찾을 수 없습니다: {report_path}")

    run_dir = args.run_dir.resolve() if args.run_dir else find_latest_run_dir(args.runs_dir.resolve())
    if run_dir is None:
        raise SystemExit(
            f"학습 결과 폴더를 찾을 수 없습니다: {args.runs_dir}\n"
            "먼저 `python train.py`로 학습을 실행해 주세요."
        )

    result = collect_run_result(run_dir)
    original = report_path.read_text(encoding="utf-8")
    updated = apply_report_updates(original, result)

    print(f"Run 폴더: {result.run_dir}")
    print(f"mAP50={result.map50:.3f}, mAP50-95={result.map50_95:.3f} (epoch {result.best_epoch})")

    if updated == original:
        print("변경 사항이 없습니다.")
        return

    if args.dry_run:
        print("\n--- dry-run: report.md 변경 미리보기 ---\n")
        print(updated)
        return

    report_path.write_text(updated, encoding="utf-8")
    print(f"report.md 갱신 완료: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)

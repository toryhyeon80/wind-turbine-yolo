"""
풍력 발전기 파손 탐지 YOLO11 학습 스크립트.

설정 파일:
  - data/data.yaml      : 데이터셋 경로·클래스
  - configs/train.yaml  : 모델·하이퍼파라미터·도메인 증강

사용 예:
  python train.py              # 전체 학습 (configs/train.yaml 기준)
  python train.py --test       # 1 Epoch 검증 (CLAUDE.md 파이프라인 테스트)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CONFIG = ROOT / "configs" / "train.yaml"
DEFAULT_DATA_CONFIG = ROOT / "data" / "data.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_yaml(data_cfg_path: Path) -> Path:
    """data.yaml 경로 유효성을 검사합니다."""
    data_cfg = load_yaml(data_cfg_path)
    dataset_root = (ROOT / data_cfg.get("path", "data")).resolve()

    for split in ("train", "val"):
        split_path = data_cfg.get(split)
        if not split_path:
            raise ValueError(f"data.yaml에 '{split}' 경로가 없습니다.")
        if not (dataset_root / split_path).exists():
            raise FileNotFoundError(
                f"데이터 분할 폴더가 없습니다: {dataset_root / split_path}\n"
                "먼저 `python split_data.py`를 실행해 주세요."
            )
    return data_cfg_path.resolve()


def build_train_kwargs(train_cfg: dict[str, Any], data_yaml: Path) -> dict[str, Any]:
    hyper = dict(train_cfg.get("hyperparameters", {}))
    aug = train_cfg.get("augmentation", {})

    if "project" in hyper:
        hyper["project"] = str((ROOT / hyper["project"]).resolve())

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "device": train_cfg.get("device", "mps"),
        **hyper,
        **aug,
    }
    return kwargs


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in ("out of memory", "oom", "mps backend"))


def train_with_oom_fallback(
    model: YOLO,
    train_kwargs: dict[str, Any],
    batch_candidates: list[int],
) -> Any:
    last_error: BaseException | None = None

    for batch in batch_candidates:
        kwargs = {**train_kwargs, "batch": batch}
        print(f"\n[학습 시작] batch={batch}, device={kwargs.get('device')}")
        try:
            return model.train(**kwargs)
        except (RuntimeError, Exception) as exc:
            if is_oom_error(exc):
                print(f"[OOM] batch={batch} 실패 → 배치 크기를 줄여 재시도합니다.")
                last_error = exc
                continue
            raise

    raise RuntimeError(
        "모든 배치 크기에서 학습에 실패했습니다. "
        "configs/train.yaml의 imgsz·batch를 낮추거나 data.yaml 경로를 확인하세요."
    ) from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 풍력 터빈 파손 탐지 학습")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="학습 설정 YAML (기본: configs/train.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="데이터셋 YAML (기본: data/data.yaml)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="1 Epoch만 실행하여 파이프라인 검증 (CLAUDE.md)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epoch 수 오버라이드 (미지정 시 configs/train.yaml 값 사용)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(args.config.resolve())
    data_yaml = resolve_data_yaml(args.data.resolve())
    train_kwargs = build_train_kwargs(train_cfg, data_yaml)

    if args.test:
        train_kwargs["epochs"] = 1
        train_kwargs["name"] = "test_run"
        print("[테스트 모드] 1 Epoch 파이프라인 검증을 실행합니다.")
    elif args.epochs is not None:
        train_kwargs["epochs"] = args.epochs

    model_name = train_cfg.get("model", "yolo11s.pt")
    print(f"모델: {model_name}")
    print(f"데이터: {data_yaml}")
    print(f"증강 flipud={train_kwargs.get('flipud')} (상하 반전 금지)")

    model = YOLO(model_name)
    batch_candidates = train_cfg.get("oom_fallback_batches", [train_kwargs.get("batch", 16)])
    if train_kwargs.get("batch") not in batch_candidates:
        batch_candidates = [train_kwargs["batch"], *batch_candidates]

    train_with_oom_fallback(model, train_kwargs, batch_candidates)
    print("\n학습이 완료되었습니다. 결과: runs/detect/ 폴더를 확인하세요.")

    if not args.test:
        print("report.md 자동 갱신을 시도합니다...")
        subprocess.run([sys.executable, str(ROOT / "update_report.py")], check=False)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)

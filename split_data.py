"""
이미지·라벨 데이터를 Train/Val(8:2)로 랜덤 분할하여 하위 폴더로 이동합니다.

- 이미지와 라벨은 stem(확장자 제외 파일명) 기준으로 1:1 짝을 맞춥니다.
- 라벨이 없는 배경 이미지는 이미지만 이동합니다.
- 이미지가 없는 라벨(고아 라벨)은 경고 후 건너뜁니다.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val")


@dataclass(frozen=True)
class Sample:
    """분할 단위: 이미지 1장 + (선택) 라벨 1개."""

    stem: str
    image_path: Path
    label_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="data/images, data/labels를 8:2 비율로 train/val로 분할합니다."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="데이터 루트 디렉터리 (기본: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="학습 데이터 비율 (기본: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 분할 시드 (기본: 42)",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_samples(images_dir: Path, labels_dir: Path) -> tuple[list[Sample], list[Path]]:
    """루트 images/labels 폴더에서 분할 대상 샘플을 수집합니다."""
    image_files = [
        p for p in images_dir.iterdir() if is_image_file(p) and p.parent == images_dir
    ]
    label_files = [
        p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt" and p.parent == labels_dir
    ]

    labels_by_stem = {p.stem: p for p in label_files}
    images_by_stem = {p.stem: p for p in image_files}

    samples: list[Sample] = []
    for stem, image_path in sorted(images_by_stem.items()):
        samples.append(
            Sample(
                stem=stem,
                image_path=image_path,
                label_path=labels_by_stem.get(stem),
            )
        )

    orphan_labels = [
        path for stem, path in labels_by_stem.items() if stem not in images_by_stem
    ]
    return samples, orphan_labels


def split_samples(
    samples: list[Sample], train_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio는 0과 1 사이여야 합니다.")

    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    if train_count == 0 and shuffled:
        train_count = 1
    if train_count == len(shuffled) and len(shuffled) > 1:
        train_count = len(shuffled) - 1

    return shuffled[:train_count], shuffled[train_count:]


def ensure_split_dirs(images_dir: Path, labels_dir: Path) -> None:
    for split in SPLIT_NAMES:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)


def move_sample(sample: Sample, split: str, images_dir: Path, labels_dir: Path) -> None:
    dest_image = images_dir / split / sample.image_path.name
    shutil.move(str(sample.image_path), str(dest_image))

    if sample.label_path is not None:
        dest_label = labels_dir / split / sample.label_path.name
        shutil.move(str(sample.label_path), str(dest_label))


def print_summary(
    train_samples: list[Sample],
    val_samples: list[Sample],
    orphan_labels: list[Path],
) -> None:
    def count_with_label(items: list[Sample]) -> int:
        return sum(1 for s in items if s.label_path is not None)

    def count_background(items: list[Sample]) -> int:
        return sum(1 for s in items if s.label_path is None)

    print("\n=== 분할 결과 ===")
    print(f"Train: {len(train_samples)}장 (라벨 있음 {count_with_label(train_samples)}, 배경 {count_background(train_samples)})")
    print(f"Val  : {len(val_samples)}장 (라벨 있음 {count_with_label(val_samples)}, 배경 {count_background(val_samples)})")

    if orphan_labels:
        print(f"\n[경고] 대응 이미지가 없는 라벨 {len(orphan_labels)}개 — 이동하지 않았습니다.")
        for path in orphan_labels[:5]:
            print(f"  - {path.name}")
        if len(orphan_labels) > 5:
            print(f"  ... 외 {len(orphan_labels) - 5}개")


def main() -> None:
    args = parse_args()
    images_dir = args.data_dir / "images"
    labels_dir = args.data_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"라벨 폴더를 찾을 수 없습니다: {labels_dir}")

    samples, orphan_labels = collect_samples(images_dir, labels_dir)
    if not samples:
        raise RuntimeError("분할할 이미지가 없습니다.")

    train_samples, val_samples = split_samples(samples, args.train_ratio, args.seed)
    ensure_split_dirs(images_dir, labels_dir)

    for sample in train_samples:
        move_sample(sample, "train", images_dir, labels_dir)
    for sample in val_samples:
        move_sample(sample, "val", images_dir, labels_dir)

    print_summary(train_samples, val_samples, orphan_labels)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Reynolds 数ごとに分かれたディレクトリ配下の VTU ファイルを、
case を連番に振り直して1つの出力ディレクトリへコピーするスクリプト。

使用例:
        # case1..10 が複数セットある構成を case1..40 のように連番化してコピー
    python scripts/flatten_vtu_directory.py \
      --input_root data/data_vtu \
            --output_dir data/data_vtu/flat_vtu_cases

        # 実際にはコピーせず、変換計画だけ確認
    python scripts/flatten_vtu_directory.py \
      --input_root data/data_vtu \
            --output_dir data/data_vtu/flat_vtu_cases \
      --dry_run
"""

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import List, Tuple


def parse_case_number(case_name: str, case_prefix: str) -> int:
    """case 名から番号を抽出する。抽出できない場合は -1 を返す。"""
    pattern = rf"^{re.escape(case_prefix)}(\d+)$"
    m = re.match(pattern, case_name)
    if not m:
        return -1
    return int(m.group(1))


def remap_case_prefix(rel_path: Path, src_case_name: str, dst_case_name: str) -> Path:
    """相対パス先頭の case 名を新しい番号へ置換する。"""
    parts = list(rel_path.parts)
    if not parts:
        return rel_path

    first = parts[0]
    if first == src_case_name:
        parts[0] = dst_case_name
    elif first.startswith(src_case_name):
        parts[0] = dst_case_name + first[len(src_case_name):]

    return Path(*parts)


def build_case_renumber_plan(
    input_root: Path,
    reynolds_glob: str,
    case_glob: str,
    case_prefix: str,
    vtu_pattern: str,
    start_index: int,
    output_dir: Path,
    cases_per_directory: int,
    shuffle: bool,
    seed: int,
    strict_cases: bool,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """複数セットの case を連番へ振り直すコピー計画を作成する。"""
    if shuffle and seed is not None:
        random.seed(seed)
    reynolds_dirs = sorted(p for p in input_root.glob(reynolds_glob) if p.is_dir())
    if not reynolds_dirs:
        reynolds_dirs = [input_root]

    case_dir_mapping: List[Tuple[Path, Path]] = []
    global_idx = start_index

    for re_dir in reynolds_dirs:
        numbered_cases: List[Tuple[int, Path]] = []
        for case_dir in sorted(p for p in re_dir.glob(case_glob) if p.is_dir()):
            num = parse_case_number(case_dir.name, case_prefix)
            if num >= 0:
                numbered_cases.append((num, case_dir))

        numbered_cases.sort(key=lambda x: x[0])
        if cases_per_directory > 0 and len(numbered_cases) < cases_per_directory:
            msg = (
                f"[WARN] {re_dir}: requested {cases_per_directory} cases, "
                f"but only {len(numbered_cases)} found"
            )
            print(msg)
            if strict_cases:
                raise ValueError(
                    f"strict_cases=True: insufficient cases in {re_dir}"
                )

        if cases_per_directory > 0:
            numbered_cases = numbered_cases[:cases_per_directory]

        for _, case_dir in numbered_cases:
            dst_case_dir = output_dir / f"{case_prefix}{global_idx}"
            case_dir_mapping.append((case_dir, dst_case_dir))
            global_idx += 1

    if shuffle:
        random.shuffle(case_dir_mapping)
        for idx, (src_case_dir, _) in enumerate(case_dir_mapping):
            case_dir_mapping[idx] = (src_case_dir, output_dir / f"{case_prefix}{start_index + idx}")

    file_plan: List[Tuple[Path, Path]] = []
    for src_case_dir, dst_case_dir in case_dir_mapping:
        vtu_files = sorted(p for p in src_case_dir.glob(vtu_pattern) if p.is_file())
        for src in vtu_files:
            rel = src.relative_to(src_case_dir)
            remapped_rel = remap_case_prefix(rel, src_case_dir.name, dst_case_dir.name)
            dst = dst_case_dir / remapped_rel
            file_plan.append((src, dst))

    return file_plan, case_dir_mapping


def execute_plan(
    plan: List[Tuple[Path, Path]],
    dry_run: bool,
) -> None:
    """計画に従ってコピーを実行する。"""

    for i, (src, dst) in enumerate(plan, start=1):
        print(f"[{i}/{len(plan)}] COPY: {src} -> {dst}")
        if dry_run:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy VTU files while renumbering case directories into one output directory"
    )
    parser.add_argument("--input_root", type=str, required=True, help="Input root directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--reynolds_glob",
        type=str,
        default="re_*",
        help="Glob pattern for Reynolds directories under input_root (default: re_*)",
    )
    parser.add_argument(
        "--vtu_pattern",
        type=str,
        default="**/*.vtu",
        help="VTU search pattern under each Reynolds directory (default: **/*.vtu)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show planned operations without file changes",
    )
    parser.add_argument(
        "--case_glob",
        type=str,
        default="case*",
        help="Glob pattern for case directories under each Reynolds directory (default: case*)",
    )
    parser.add_argument(
        "--case_prefix",
        type=str,
        default="case",
        help="Case directory prefix for numbering (default: case)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Start index for renumbered cases (default: 1)",
    )
    parser.add_argument(
        "--cases_per_directory",
        type=int,
        default=0,
        help="Number of cases to take from each Reynolds directory (0 means all)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle case directories before renumbering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling (optional, for reproducibility)",
    )
    parser.add_argument(
        "--strict_cases",
        action="store_true",
        help="Fail if any Reynolds directory has fewer cases than --cases_per_directory",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"input_root is not a directory: {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    plan, case_dir_mapping = build_case_renumber_plan(
        input_root=input_root,
        reynolds_glob=args.reynolds_glob,
        case_glob=args.case_glob,
        case_prefix=args.case_prefix,
        vtu_pattern=args.vtu_pattern,
        start_index=args.start_index,
        output_dir=output_dir,
        cases_per_directory=args.cases_per_directory,
        shuffle=args.shuffle,
        seed=args.seed,
        strict_cases=args.strict_cases,
    )
    if not plan:
        print("No VTU files found in case directories. Nothing to do.")
        return

    print(f"Found {len(case_dir_mapping)} case directories")
    print(f"Found {len(plan)} VTU files")
    print(f"Output directory: {output_dir}")
    if args.cases_per_directory > 0:
        print(f"Cases per Reynolds directory: {args.cases_per_directory}")
    if args.shuffle:
        print(f"Shuffled: True (seed: {args.seed if args.seed is not None else 'random'})")
    if args.dry_run:
        print("Dry-run mode: no files will be changed")

    preview_count = min(10, len(case_dir_mapping))
    print(f"Case renumber preview (first {preview_count}):")
    for src_case, dst_case in case_dir_mapping[:preview_count]:
        print(f"  {src_case} -> {dst_case}")

    execute_plan(plan, args.dry_run)

    print("Done")


if __name__ == "__main__":
    main()

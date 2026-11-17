#!/usr/bin/env python
"""
VTU ファイルを XDMF/HDF5 形式に変換するスクリプト。
設定セクションで入力パスとオプションを指定してください。
"""

import glob
import os
import shutil
import sys
from pathlib import Path
from typing import List

import meshio

# ============================================================================
# 設定セクション - ここを編集してください
# ============================================================================

# 入力 VTU ファイルパス（ワイルドカード対応）
# 例: "predictions/cylinder/step000.vtu"
#     "predictions/cylinder/step*.vtu"
# 複数フォルダを指定したい場合は INPUT_PATTERNS を使う
INPUT_PATTERN = "predictions/cylinder/step*.vtu"

# 複数フォルダ/複数パターン指定（INPUT_PATTERN より優先）
# 例:
# INPUT_PATTERNS = [
#     "predictions/cylinder/*.vtu",
#     "predictions/another_case/*.vtu",
# ]
INPUT_PATTERNS = []

# ケース単位の一括変換モード（各ケースごとに1つのXDMF/H5を作る）
# True の場合は CASE_DIRS を使用し、OUTPUT_FILE は無視される
PROCESS_EACH_CASE = True

# ケースディレクトリ一覧（PROCESS_EACH_CASE=True のとき使用）
# 例:
# CASE_DIRS = [
#     "predictions/case_001",
#     "predictions/case_002",
# ]

# CASE_DIRS を空にしておくと、CASE_ROOT_DIR と CASE_NUMBERS から自動生成する
CASE_DIRS = []

# ケースの親ディレクトリ
CASE_ROOT_DIR = "data/data_vtu/re_50-100"

# 処理したいケース番号
# 例: range(1, 6) -> case1, case2, case3, case4, case5
CASE_NUMBERS = range(1, 26)

# ケースディレクトリ名の形式
CASE_NAME_TEMPLATE = "case{}"

# ケースディレクトリ配下で収集する VTU のパターン（vtm は対象外）
# このデータ構造では caseX/caseX_****/internal.vtu のため再帰指定を推奨
# 例: "*.vtu", "step*.vtu", "**/*.vtu", "**/internal.vtu"
VTU_GLOB_IN_CASE = "**/*.vtu"

# ケースごとの出力ファイル名（拡張子なし）
# 例: "timeseries" -> <case_dir>/timeseries.xdmf, <case_dir>/timeseries.h5
OUTPUT_BASENAME_PER_CASE = "timeseries"

# 出力ファイル名（拡張子なし）
# 例: "output/result" → output/result.xdmf と output/result.h5 が生成される
OUTPUT_FILE = "predictions/cylinder/series"

# タイムステップ値（連続するタイムフレーム間の時間間隔）
TIMESTEP = 1.0

# 変換後に元の VTU ファイルを削除するか（True / False）
REMOVE_VTUS = False

# ============================================================================


def vtu_to_xdmf(
    filename: str, files_list: List[str], timestep=1, remove_vtus: bool = True
) -> None:
    """
    複数のメッシュ（同一の頂点と要素を持つ）の時系列をXDMF/HDF5形式で保存。

    Args:
        filename (str): 出力するXDMF/HDF5ファイルの名前（拡張子なし）
        files_list (List[str]): 圧縮対象のVTUファイルパスのリスト
        timestep (float, optional): 連続するタイムフレーム間のタイムステップ値
        remove_vtus (bool, optional): Trueの場合、変換後に元のVTUファイルを削除

    Returns:
        None: ファイル名で指定されたパスにXDMF/HDF5ファイルが保存される
    """
    # 出力ファイルのパスを定義
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    # 最初のVTUファイルから頂点と要素データを読み込み
    init_vtu = meshio.read(files_list[0])
    points = init_vtu.points
    cells = init_vtu.cells

    # HDF5フォーマットの時系列ライターを開く
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # メッシュ構造（頂点と要素）を1度だけ書き込み
        writer.write_points_cells(points, cells)

        # 各タイムステップのデータを順序に書き込み
        t = 0
        for file in files_list:
            mesh = meshio.read(file)
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            writer.write_data(t, point_data=point_data, cell_data=cell_data)
            t += timestep

    # H5アーカイブはカレントディレクトリに作成されるため、目的地に移動
    shutil.move(
        src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]), dst=h5_filename
    )

    # 元のVTUファイルを削除（オプション）
    if remove_vtus:
        for file in files_list:
            os.remove(file)


def main():
    global INPUT_PATTERN, INPUT_PATTERNS, PROCESS_EACH_CASE
    global CASE_DIRS, CASE_ROOT_DIR, CASE_NUMBERS, CASE_NAME_TEMPLATE
    global VTU_GLOB_IN_CASE, OUTPUT_BASENAME_PER_CASE
    global OUTPUT_FILE, TIMESTEP, REMOVE_VTUS

    if PROCESS_EACH_CASE:
        if CASE_DIRS:
            case_dirs = CASE_DIRS
        else:
            case_dirs = [
                str(Path(CASE_ROOT_DIR) / CASE_NAME_TEMPLATE.format(case_number))
                for case_number in CASE_NUMBERS
            ]

        if not case_dirs:
            print("エラー: PROCESS_EACH_CASE=True の場合は CASE_DIRS を設定してください", file=sys.stderr)
            sys.exit(1)

        print("=" * 70)
        print("ケース単位 VTU → XDMF/HDF5 変換")
        print("=" * 70)
        print("ケースディレクトリ:")
        for case_dir in case_dirs:
            print(f"  - {case_dir}")
        print(f"ケース内 VTU パターン: {VTU_GLOB_IN_CASE}")
        print(f"ケースごとの出力名: {OUTPUT_BASENAME_PER_CASE}")
        print(f"タイムステップ: {TIMESTEP}")
        print(f"VTU 削除オプション: {REMOVE_VTUS}")
        print()

        converted_cases = 0
        for case_dir in case_dirs:
            case_path = Path(case_dir)
            if not case_path.exists() or not case_path.is_dir():
                print(f"スキップ: ディレクトリが存在しません: {case_dir}")
                continue

            pattern = str(case_path / VTU_GLOB_IN_CASE)
            files_list = sorted(set(glob.glob(pattern, recursive=True)))
            if not files_list:
                print(f"スキップ: VTU が見つかりません: {pattern}")
                continue

            output_base = str(case_path / OUTPUT_BASENAME_PER_CASE)
            print(f"変換中: {case_dir} ({len(files_list)} files)")
            try:
                vtu_to_xdmf(
                    filename=output_base,
                    files_list=files_list,
                    timestep=TIMESTEP,
                    remove_vtus=REMOVE_VTUS,
                )
                print(f"  ✓ 出力: {output_base}.xdmf / {output_base}.h5")
                converted_cases += 1
            except Exception as e:
                print(f"  ✗ エラー: {e}", file=sys.stderr)

        if converted_cases == 0:
            print("エラー: 変換できたケースがありません", file=sys.stderr)
            sys.exit(1)

        print()
        print(f"✓ 完了: {converted_cases} ケースを変換")
        print("=" * 70)
        return

    # ワイルドカードを展開（複数パターン対応）
    patterns = INPUT_PATTERNS if INPUT_PATTERNS else [INPUT_PATTERN]
    files_set = set()
    for pattern in patterns:
        files_set.update(glob.glob(pattern))
    files_list = sorted(files_set)

    if not files_list:
        print(f"エラー: 入力ファイルが見つかりません: {patterns}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("VTU → XDMF/HDF5 変換")
    print("=" * 70)
    print("入力パターン:")
    for p in patterns:
        print(f"  - {p}")
    print(f"入力ファイル数: {len(files_list)}")
    for f in files_list[:5]:  # 最初の5つを表示
        print(f"  - {f}")
    if len(files_list) > 5:
        print(f"  ... 他 {len(files_list) - 5} ファイル")

    # 出力ファイル名のディレクトリを作成
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n出力ファイル:")
    print(f"  - {OUTPUT_FILE}.xdmf")
    print(f"  - {OUTPUT_FILE}.h5")
    print(f"タイムステップ: {TIMESTEP}")
    print(f"VTU 削除オプション: {REMOVE_VTUS}")
    print()

    try:
        vtu_to_xdmf(
            filename=OUTPUT_FILE,
            files_list=files_list,
            timestep=TIMESTEP,
            remove_vtus=REMOVE_VTUS,
        )
        print("✓ 変換完了")
        print("=" * 70)
    except Exception as e:
        print(f"✗ エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

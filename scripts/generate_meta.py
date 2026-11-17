import json
import h5py
from pathlib import Path

# ========== 設定 ==========
H5_PATH = Path("data/data_h5/re_100-300_100_train.h5") # 分割した train.h5 などを指定
OUTPUT_META_PATH = Path("data/data_h5/openfoam_meta.json")
DT_VALUE = 1.0  # タイムステップの間隔
# ==========================

def main():
    if not H5_PATH.exists():
        print(f"❌ エラー: HDF5ファイルが見つかりません: {H5_PATH}")
        return

    print(f"▶ HDF5ファイルの構造を詳細解析します: {H5_PATH}")
    
    with h5py.File(H5_PATH, "r") as f:
        sim_cases = sorted(list(f.keys()))
        if not sim_cases:
            print("❌ エラー: HDF5ファイル内にケースグループ（sim_XXXXX）が存在しません。")
            return
        
        # 最初のケース（sim_00000）をマスターテンプレートとして読み込む
        sample_case = sim_cases[0]
        group = f[sample_case]
        
        features_dict = {}
        field_names = []
        trajectory_length = None

        # 固定で並び順を cylinder_meta.json の標準に合わせる、または検出された順にする
        # 確実に必要なキーを優先して走査
        target_keys = ["cells", "mesh_pos", "node_type", "U", "p"]
        available_keys = [k for k in target_keys if k in group] + [k for k in group.keys() if k not in target_keys]

        for key in available_keys:
            dataset = group[key]
            shape_info = list(dataset.shape)
            
            # --- 1. 動的特徴量 (U, p など) のパース ---
            if len(shape_info) == 3:
                t_len, num_nodes, dim = shape_info
                if trajectory_length is None:
                    trajectory_length = t_len
                
                features_dict[key] = {
                    "type": "dynamic",
                    "shape": [t_len, -1, dim],  # [Time, -1, Dim] 構造に落とし込む
                    "dtype": str(dataset.dtype)
                }
                field_names.append(key)

            # --- 2. 静的特徴量 (mesh_pos, cells, node_type) のパース ---
            elif len(shape_info) == 2:
                # cells や mesh_pos などの静的データ
                # cylinder_meta.jsonの規則に従い、Time次元位置には trajectory_length を、
                # ノード数/エッジ数の位置には可変を示す -1 を割り当てます
                dim = shape_info[1]
                
                features_dict[key] = {
                    "type": "static",
                    "shape": ["TMP_LEN", -1, dim], # 後で確定したtrajectory_lengthに置換
                    "dtype": str(dataset.dtype)
                }
                field_names.append(key)
                
            elif len(shape_info) == 1:
                # cells が1次元フラット配列で保存されている場合のフォールバック
                features_dict[key] = {
                    "type": "static",
                    "shape": ["TMP_LEN", -1, 1],
                    "dtype": str(dataset.dtype)
                }
                field_names.append(key)

        # 全ての静的特徴量の shape に、確定した trajectory_length を安全にマッピング
        if trajectory_length is None:
            print("⚠️ 警告: 動的特徴量が検出されなかったため、デフォルトの長さを7に設定します。")
            trajectory_length = 7

        for key, value in features_dict.items():
            if value["shape"][0] == "TMP_LEN":
                value["shape"][0] = trajectory_length

        # numpyの型文字列（float32やint32など）をピュアなJSON規格に微調整
        for key, value in features_dict.items():
            if "float" in value["dtype"]:
                value["dtype"] = "float32"
            elif "int" in value["dtype"]:
                value["dtype"] = "int32"

        # cylinder_meta.json と一言一句同じ構造をディクショナリでビルド
        meta_data = {
            "dt": DT_VALUE,
            "features": features_dict,
            "field_names": field_names,
            "trajectory_length": trajectory_length
        }

        # 保存先ディレクトリの作成
        OUTPUT_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存
        with open(OUTPUT_META_PATH, "w") as json_file:
            json.dump(meta_data, json_file, indent=2)

    print("\n✨ 完璧な形式のメタデータファイルが作成されました！")
    print(f"📂 出力先パス: {OUTPUT_META_PATH}")
    print("📋 生成された内容プレビュー:")
    print(json.dumps(meta_data, indent=2))

if __name__ == "__main__":
    main()
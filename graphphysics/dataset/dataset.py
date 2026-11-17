import json
import math
from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch_geometric.transforms as T
from loguru import logger
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_random_edge, subgraph

from graphphysics.utils.torch_graph import (
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    create_subgraphs,
    get_masked_indexes,
)


class BaseDataset(Dataset, ABC):
    """
    メッシュベースの物理シミュレーションデータを扱うための基底データセットクラス。
    HDF5形式やXDMF形式などの具体的なデータ読み込みクラスは、このクラスを継承して実装される。
    """
    def __init__(
        self,
        meta_path: str,
        targets: list[str],
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        world_pos_parameters: Optional[dict] = None,
        use_partitioning: bool = False,
        num_partitions: Optional[int] = None,
        max_nodes_per_partition: Optional[int] = None,
    ):
        # シミュレーションの詳細情報（物理量の名前や時間ステップなど）を含むJSONを読み込む
        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        # データの前処理はCPU上で行い、デバイス（GPU等）への転送はDataLoaderに遅延評価させる
        self.device = torch.device("cpu")

        self.meta: Dict[str, Any] = meta

        # ターゲット（モデルが予測する物理量。例: 'velocity'）が正しく定義されているか検証
        if targets is None or len(targets) == 0:
            raise ValueError("少なくとも1つのターゲット（予測対象）を指定する必要があります。")
        for target in targets:
            if target not in self.meta["features"]:
                raise ValueError(f"ターゲット {target} が利用可能なフィールド（特徴量）に見つかりません。")
            if self.meta["features"][target]["type"] != "dynamic":
                raise ValueError(f"ターゲット {target} は時間変化する動的（dynamic）フィールドではありません。")
        self.targets = targets

        # 1つのシミュレーションケース（軌跡）に含まれる総ステップ数
        self.trajectory_length: int = self.meta["trajectory_length"]
        self.num_trajectories: Optional[int] = None
        
        # K-hop（数手先の隣接ノード）のエッジ情報を軌跡ごとにキャッシュして計算を省略するための辞書
        self.khop_edge_index_cache: Dict[int, torch.Tensor] = {}
        self.khop_edge_attr_cache: Dict[int, torch.Tensor] = {}

        self.preprocessing = preprocessing
        self.masking_ratio = masking_ratio
        self.khop = khop
        self.new_edges_ratio = new_edges_ratio
        self.add_edge_features = add_edge_features
        self.use_previous_data = use_previous_data
        
        # 巨大なメッシュをサブグラフに分割する（Partitioning）設定
        self.use_partitioning = use_partitioning

        if use_partitioning:
            # 分割数指定と、1分割あたりの最大ノード数指定は排他利用（どちらか一方のみ）
            if num_partitions is not None and max_nodes_per_partition is not None:
                raise ValueError(
                    "'num_partitions'（分割数）か 'max_nodes_per_partition'（最大ノード数）のどちらか一方のみを指定してください。"
                )
            if num_partitions is None and max_nodes_per_partition is None:
                raise ValueError(
                    "'use_partitioning'がTrueの場合、'num_partitions' か 'max_nodes_per_partition' を指定する必要があります。"
                )

        self.num_partitions = num_partitions
        self.max_nodes_per_partition = max_nodes_per_partition
        
        # 分割されたサブグラフのノードID情報を軌跡ごとにキャッシュする辞書
        self.partitions_node_ids_cache: Dict[int, List[torch.Tensor]] = {}
        self.partitions_per_trajectory: Dict[int, int] = {}
        
        # データセット全体でのサンプルの通し番号（グローバルインデックス）を管理する累積リスト
        self.cumulative_samples: List[int] = [0]
        self._size_dataset = 0
        self._len_dataset = 0

        # （オプション）メッシュ間のグローバルな位置情報エッジ（world edges）の設定
        self.world_pos_index_start = None
        self.world_pos_index_end = None
        if world_pos_parameters is not None:
            self.world_pos_index_start = world_pos_parameters.get("world_pos_index_start")
            self.world_pos_index_end = world_pos_parameters.get("world_pos_index_end")

    @property
    def size_dataset(self) -> int:
        """データセット内の総軌跡数（シミュレーションケース数）を返す"""
        return self._size_dataset

    @abstractmethod
    def _build_index_map(self):
        """データセットのインデックスマップを構築するための抽象メソッド（子クラスで実装）"""
        raise NotImplementedError

    def _get_indices(self, index: int) -> Tuple[int, int, int]:
        """
        DataLoaderから渡される1次元のグローバルなインデックス（例: 135番目のサンプル）から、
        対応する軌跡番号(traj_index)、フレーム番号(frame)、サブグラフ番号(subgraph_idx)を逆算する。
        """
        # 二分探索を用いて、このインデックスがどの軌跡（シミュレーションケース）に属するかを高速に特定
        traj_index = bisect_right(self.cumulative_samples, index) - 1

        # その軌跡の中でのローカルインデックス
        local_index = index - self.cumulative_samples[traj_index]
        num_partitions = self.partitions_per_trajectory[traj_index]

        # 軌跡内の分割数を利用して、何フレーム目の、どのサブグラフ領域かを特定
        frame_in_traj = local_index // num_partitions
        subgraph_idx = local_index % num_partitions
        
        # 過去データ（t-1等）を入力に用いる場合、取得開始フレームを1つずらす
        frame = frame_in_traj + int(self.use_previous_data)
        
        return traj_index, frame, subgraph_idx

    def _add_traj_to_index_map(self, traj_index: int, num_nodes: int):
        """
        1つの軌跡（シミュレーション動画）をインデックスマップに追加し、
        総サンプル数（長さ）や分割情報を更新する。

        パラメータ:
            traj_index (int): 追加する軌跡のインデックス
            num_nodes (int): その軌跡のメッシュを構成するノード数
        """
        # パーティショニング（メッシュ分割）が有効な場合、分割数を計算
        if self.use_partitioning:
            if self.num_partitions is not None:
                num_partitions = self.num_partitions
            else:
                num_partitions = math.ceil(num_nodes / self.max_nodes_per_partition)
        else:
            num_partitions = 1

        self.partitions_per_trajectory[traj_index] = num_partitions
        
        # 予測に利用可能な有効フレーム数を計算
        # 教師ラベル(t+1)用に1ステップ分を引き、さらに過去データ(t-1)を利用する場合はさらに1ステップ引く
        num_valid_frames = self.trajectory_length - 1 - int(self.use_previous_data)

        # この軌跡から得られる合計サンプル数 = 有効フレーム数 × メッシュ分割数
        total_samples_in_traj = num_valid_frames * num_partitions
        
        # データセットの全体の長さを更新
        self._len_dataset += total_samples_in_traj
        self.cumulative_samples.append(self._len_dataset)

    def __len__(self) -> int:
        return self._len_dataset

    @abstractmethod
    def __getitem__(self, index: int) -> Data:
        """1つのデータサンプル（グラフ）を取得する抽象メソッド（子クラスで実装）"""
        raise NotImplementedError

    def _apply_preprocessing(self, graph: Data) -> Data:
        """
        グラフに対して前処理（ガウスノイズの付加など）を適用する。
        """
        if self.preprocessing is not None:
            graph = self.preprocessing(graph)
        return graph

    def _add_random_edges(self, graph: Data) -> Data:
        """
        メッシュ上の情報の長距離伝播（ランダムジャンプ）をシミュレートするため、
        隣接行列にランダムなエッジを一定割合(new_edges_ratio)で追加する。
        """
        new_edges_ratio = self.new_edges_ratio
        if new_edges_ratio <= 0.0 or new_edges_ratio > 1.0:
            return graph
            
        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None:
            logger.warning(
                "ランダムエッジを追加しようとしていますが、グラフにエッジ(edge_index)が存在しません。"
            )
            return graph

        # PyGの機能を用いてランダムな無向エッジを追加
        new_edge_index, _ = add_random_edge(
            edge_index, p=new_edges_ratio, force_undirected=True
        )
        graph.edge_index = new_edge_index
        
        # エッジが変化したため、デカルト座標や距離に基づくエッジ特徴量(edge_attr)を再計算する
        if self.add_edge_features:
            graph.edge_attr = None
            edge_feature_computer = T.Compose(
                [
                    T.Cartesian(norm=False),
                    T.Distance(norm=False),
                ]
            )
            graph = edge_feature_computer(graph).to(self.device)

        return graph

    def _apply_k_hop(self, graph: Data, traj_index: int) -> Data:
        """
        K-hop展開（直接繋がっていない数手先のノード同士もエッジで結ぶ）をグラフに適用する。
        計算コストを削減するため、一度計算したトポロジーはキャッシュに保存される。
        """
        if self.khop > 1:
            if traj_index in self.khop_edge_index_cache:
                # キャッシュが存在する場合はそれを再利用
                khop_edge_index = self.khop_edge_index_cache[traj_index]
                graph.edge_index = khop_edge_index.to(graph.edge_index.device)
                if self.add_edge_features:
                    khop_edge_attr = self.khop_edge_attr_cache[traj_index]
                    graph.edge_attr = khop_edge_attr.to(graph.edge_attr.device)
            else:
                # K-hopエッジインデックスを計算し、キャッシュに保存
                if self.add_edge_features:
                    # エッジ特徴量（距離など）も含めて計算する場合
                    graph = compute_k_hop_graph(
                        graph,
                        num_hops=self.khop,
                        add_edge_features_to_khop=True,
                        device=self.device,
                        world_pos_index_start=self.world_pos_index_start,
                        world_pos_index_end=self.world_pos_index_end,
                    )
                    self.khop_edge_index_cache[traj_index] = graph.edge_index.cpu()
                    self.khop_edge_attr_cache[traj_index] = graph.edge_attr.cpu()
                else:
                    # トポロジー（繋がり）のみを計算する場合
                    khop_edge_index = compute_k_hop_edge_index(
                        graph.edge_index, self.khop, graph.num_nodes
                    )
                    self.khop_edge_index_cache[traj_index] = khop_edge_index.cpu()
                    graph.edge_index = khop_edge_index.to(graph.edge_index.device)
        return graph

    def _apply_partition(self, graph: Data, node_ids: torch.Tensor):
        """
        提供されたノードID群（node_ids）に基づいて、巨大なグラフから部分グラフ（サブグラフ）を切り出す。
        メモリ節約（VRAMオプティマイズ）のために利用される。
        """
        node_ids, _ = node_ids.sort()
        # 選択されたノードの特徴量と座標のみを抽出
        x = graph.x[node_ids]
        y = graph.y[node_ids]
        pos = graph.pos[node_ids]

        # サブグラフ内のノード同士を結ぶエッジのみを抽出し、インデックスを 0 から振り直す(relabel_nodes=True)
        edge_index, edge_attr = subgraph(
            subset=node_ids,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            relabel_nodes=True,
            num_nodes=graph.num_nodes,
        )

        # 抽出されたデータから新しい小規模なDataオブジェクト（グラフ）を構築
        sub_graph = Data(
            x=x,
            y=y,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=x.size(0),
            traj_index=graph.traj_index,
        )
        return sub_graph

    def _get_partition(self, graph, traj_index, subgraph_idx):
        """
        特定の軌跡とサブグラフインデックスに対する、分割された部分グラフを取得する。
        （初回アクセス時にMETISアルゴリズム等を用いてグラフをクラスタリング・分割する）
        """
        num_partitions = self.partitions_per_trajectory[traj_index]
        if num_partitions != 1:
            if traj_index not in self.partitions_node_ids_cache:
                # キャッシュが無い場合はメッシュを分割（クラスタリング）し、所属ノードIDリストを保存
                loader, node_ids = create_subgraphs(graph, num_partitions)
                self.partitions_node_ids_cache[traj_index] = node_ids

            # 指定されたサブグラフ領域に対応するノードID群を取得し、切り出しを適用
            partitioned_node_ids = self.partitions_node_ids_cache[traj_index][
                subgraph_idx
            ].to(self.device)
            graph = self._apply_partition(graph, partitioned_node_ids)

        return graph

    def _may_remove_edges_attr(self, graph: Data) -> Data:
        """エッジ特徴量が不要な設定(add_edge_features=False)の場合、データから削除してメモリを節約する"""
        if not self.add_edge_features:
            graph.edge_attr = None
        return graph

    def _get_masked_indexes(self, graph: Data) -> Optional[torch.Tensor]:
        """
        マスキング（MeshMask手法など）が有効な場合、隠蔽するノードのインデックスを取得する。
        """
        if self.masking_ratio is not None:
            selected_indices = get_masked_indexes(graph, self.masking_ratio)
            return selected_indices
        else:
            return None

class MultiStepDataset(TorchDataset):
    """同一trajectoryの連続サンプルをmulti-step学習用に返す。"""

    def __init__(self, dataset: BaseDataset, steps: int):
        if steps < 2:
            raise ValueError("multi-step学習のstepsは2以上である必要があります。")
        if dataset.masking_ratio is not None:
            raise ValueError("multi-step学習とノードマスキングの併用は未対応です。")
        self.dataset = dataset
        self.steps = int(steps)
        self._start_indices: List[Tuple[int, int]] = []
        for traj_index in range(dataset.size_dataset):
            start = dataset.cumulative_samples[traj_index]
            end = dataset.cumulative_samples[traj_index + 1]
            num_partitions = dataset.partitions_per_trajectory[traj_index]
            valid_frames = (end - start) // num_partitions
            num_starts = valid_frames - self.steps + 1
            for frame in range(max(0, num_starts)):
                for partition in range(num_partitions):
                    sample_index = start + frame * num_partitions + partition
                    self._start_indices.append((sample_index, num_partitions))
        if not self._start_indices:
            raise ValueError(
                f"steps={self.steps}を満たす連続フレームがデータセットにありません。"
            )

    def __len__(self) -> int:
        return len(self._start_indices)

    def __getitem__(self, index: int) -> List[Data]:
        start, stride = self._start_indices[index]
        samples = [self.dataset[start + step * stride] for step in range(self.steps)]
        if any(not isinstance(sample, Data) for sample in samples):
            raise TypeError("multi-step学習ではDatasetがDataを返す必要があります。")
        return samples

    @property
    def type(self) -> str:
        return self.dataset.type

    @property
    def trajectory_length(self) -> int:
        return self.dataset.trajectory_length

    @property
    def dt(self) -> float:
        return self.dataset.dt

    @property
    def size_dataset(self) -> int:
        return self.dataset.size_dataset

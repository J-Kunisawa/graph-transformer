import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

import graphphysics.models.transolver as Transolver
from graphphysics.models.eagle import EagleMeshTransformer, clusters_from_labels
from graphphysics.models.layers import (
    GraphNetBlock,
    TemporalAttention,
    Transformer,
    build_mlp,
)
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.torch_graph import (
    compute_exact_k_hop_edge_index,
    generate_global_attention_edges,
    generate_random_edges,
    merge_edge_indices,
)

# --- 外部高速化ライブラリ (DGL Sparse) のロードと動的フォールバック制御 ---
try:
    import dgl.sparse as dglsp
    HAS_DGL_SPARSE = True
except ImportError as e:
    HAS_DGL_SPARSE = False
    dglsp = None
    logger.critical(
        f"DGLのインポートに失敗しました。Transformerアーキテクチャは自動的に torch_geometric.TransformerConv にフォールバックします。理由: {e}"
    )
    
    # 環境変数またはユーザー入力により、DGLなしでの訓練続行を承認するか確認
    assume_no_dgl = os.getenv("GRAPH_PHYSICS_ASSUME_NO_DGL", "").strip().lower()
    if assume_no_dgl in {"1", "true", "yes", "y"}:
        logger.warning(
            "GRAPH_PHYSICS_ASSUME_NO_DGL が設定されています。torch_geometric.TransformerConv を使用して処理を続行します。"
        )
    else:
        prompt = (
            "DGLなしで学習を起動しようとしています。Transformerモデルは代替として "
            "torch_geometric.TransformerConv を使用します。よろしいですか？ "
            "続行する場合は 'Y'、学習をキャンセルする場合は 'N' を入力してください: "
        )
        while True:
            try:
                response = input(prompt)
            except EOFError:
                logger.critical(
                    "ユーザー入力を受け取れませんでした。対話をスキップするには環境変数 GRAPH_PHYSICS_ASSUME_NO_DGL=1 を設定してください。"
                )
                sys.exit(1)
            response = response.strip().lower()
            if response == "y":
                break
            if response == "n":
                sys.exit(1)
            print("Y または N でお答えください。")


class EncodeProcessDecode(nn.Module):
    """
    Encode-Process-Decode (EPD) モデル (MeshGraphNetベース)。
    
    1. Encoder: ノード場、エッジ場の生特徴量を独立した高次元潜在空間に写像する。
    2. Processor: グラフネットワークブロックを複数回（message_passing_num回）重ね、周辺メッシュとの物理通信を行う。
    3. Decoder: 最終的な潜在特徴量から、目的の物理変化量（時間微分値等）を解き放つ。
    """
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        only_processor: bool = False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        use_gated_mlp: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
    ):
        super().__init__()
        self.only_processor = only_processor
        self.hidden_size = hidden_size
        self.d = output_size
        self.use_temporal_block = use_temporal_block
        self.use_gated_mlp = use_gated_mlp
        self.use_rope = use_rope_embeddings
        self.use_gate = use_gated_attention
        self.rope_axes = rope_pos_dimension
        self.rope_base = rope_base
        
        # 空間回転位置埋め込み（RoPE）の次元チェック
        if self.use_rope and self.rope_axes not in (2, 3):
            raise ValueError(
                "use_rope_embeddings=True の場合、rope_pos_dimension は 2 または 3 である必要があります。"
            )
            
        # 時間補正アテンションブロックの設定とバックエンド警告
        if self.use_temporal_block and not HAS_DGL_SPARSE:
            logger.warning(
                "use_temporal_block=True ですが DGL sparse バックエンドが利用できません。 "
                "時間アテンションは疎行列隣接関係なし（全結合的）で実行されます。"
            )
        self.temporal_block = (
            TemporalAttention(hidden_size=hidden_size)
            if self.use_temporal_block
            else None
        )

        # 特徴量のエンコーダ・デコーダMLPの構築（only_processor=True の場合はスキップ）
        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=node_input_size, hidden_size=hidden_size, out_size=hidden_size,
            )
            self.edges_encoder = build_mlp(
                in_size=edge_input_size, hidden_size=hidden_size, out_size=hidden_size,
            )
            # デコーダの最終層には物理量を出力するため正規化層（layer_norm）をかけない
            self.decode_module = build_mlp(
                in_size=hidden_size, hidden_size=hidden_size, out_size=output_size, layer_norm=False,
            )

        # プロセッサコア（Message Passing レイヤー群）のディープスタック構築
        self.processor_list = nn.ModuleList(
            [
                GraphNetBlock(
                    hidden_size=hidden_size,
                    use_gated_mlp=use_gated_mlp,
                    use_rope=use_rope_embeddings,
                    rope_axes=rope_pos_dimension,
                    rope_base=rope_base,
                    use_gate=use_gated_attention,
                )
                for _ in range(message_passing_num)
            ]
        )

    def forward(self, graph: Data) -> torch.Tensor:
        edge_index = graph.edge_index
        adj = None

        # 1. 前置エンコード処理
        if self.only_processor:
            x, edge_attr = graph.x, graph.edge_attr
        else:
            x = self.nodes_encoder(graph.x)
            edge_attr = self.edges_encoder(graph.edge_attr)

        # DGL Sparse用の疎行列隣接関係オブジェクトの動的生成
        if self.use_temporal_block and HAS_DGL_SPARSE:
            val = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)
            adj = dglsp.spmatrix(indices=edge_index, val=val, shape=(x.size(0), x.size(0)))

        prev_x = x
        last_x = x
        
        # 空間RoPEおよびGatingに必要なオプション属性をグラフから抽出
        pos = getattr(graph, "pos", None) if self.use_rope else None
        if self.use_rope and pos is None:
            raise ValueError(
                "use_rope_embeddings=True の場合、入力グラフデータに座標属性 `pos` が含まれている必要があります。"
            )
        phi = getattr(graph, "phi", None) if self.use_gate else None
        
        # 2. メッセージパッシング・ループの実行
        for block in self.processor_list:
            prev_x = x # 1つ前のブロック出力を記憶（時間アテンション用）
            x, edge_attr = block(
                x,
                edge_index,
                edge_attr,
                pos=pos,
                phi=phi,
            )
            last_x = x # 最新のブロック出力を更新

        # 3. 必要に応じて時間補正クロスアテンションを適用
        if self.use_temporal_block and self.temporal_block is not None:
            x = self.temporal_block(
                prev_x,
                last_x,
                adj if HAS_DGL_SPARSE else None,
            )

        # 4. 後置デコード処理（最終物理量の出力）
        if self.only_processor:
            return x
        else:
            x_decoded = self.decode_module(x)
            return x_decoded


class EncodeTransformDecode(nn.Module):
    """
    Encode-Transform-Decode モデル。
    
    メッセージパッシングの代わりに Transformer (自己アテンション) ブロックを用いて、
    メッシュ内の長距離・大域的な空間相互作用を直接計算するアーキテクチャ。
    """
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        only_processor: bool = False,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        rope_length_scale: float = 1.0,
        use_temporal_block: bool = False,
        adjacency_augmentation: Optional[dict] = None,
    ):
        super(EncodeTransformDecode, self).__init__()
        self.hidden_size = hidden_size
        self.only_processor = only_processor
        self.d = output_size
        
        # TransformerでのRoPEは、DGL Sparseによる隣接マスク処理が走る場合のみ有効化可能
        self.use_rope_embeddings = use_rope_embeddings and HAS_DGL_SPARSE
        self.use_gated_attention = use_gated_attention
        self._requested_rope = use_rope_embeddings
        self.use_temporal_block = use_temporal_block
        self.num_heads = num_heads
        self.num_layers = message_passing_num
        self.rope_pos_dimension = rope_pos_dimension
        self.rope_length_scale = float(rope_length_scale)
        if self.rope_length_scale <= 0:
            raise ValueError("rope_length_scale は正の値である必要があります。")
        self.adjacency_augmentation = dict(adjacency_augmentation or {})
        if not self.adjacency_augmentation.get("enabled", True):
            self.adjacency_augmentation = {}
        self.random_edges_ratio = float(
            self.adjacency_augmentation.get("random_edges_ratio", 0.0)
        )
        if not 0.0 <= self.random_edges_ratio <= 1.0:
            raise ValueError("random_edges_ratio は0以上1以下である必要があります。")

        global_config = self.adjacency_augmentation.get("global_attention", {}) or {}
        if not global_config.get("enabled", True):
            global_config = {}
        raw_node_types = global_config.get("node_types", [])
        self.global_node_types = tuple(
            int(NodeType[node_type]) if isinstance(node_type, str) else int(node_type)
            for node_type in raw_node_types
        )
        self.global_attention_ratio = float(global_config.get("sample_ratio", 0.0))
        if not 0.0 <= self.global_attention_ratio <= 1.0:
            raise ValueError(
                "global_attention.sample_ratio は0以上1以下である必要があります。"
            )

        self.layer_head_hops = self._parse_dilation_config(
            self.adjacency_augmentation.get("dilation")
        )

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=node_input_size, hidden_size=hidden_size, out_size=hidden_size,
            )
            self.decode_module = build_mlp(
                in_size=hidden_size, hidden_size=hidden_size, out_size=output_size, layer_norm=False,
            )

        # バックエンドの有無によって、生成するアテンションレイヤーを根底から分岐させる
        if HAS_DGL_SPARSE:
            # DGL環境：高性能なカスタムSparse Transformerブロックを配置
            self.processor_list = nn.ModuleList(
                [
                    Transformer(
                        input_dim=hidden_size,
                        output_dim=hidden_size,
                        num_heads=num_heads,
                        use_proj_bias=use_proj_bias,
                        use_separate_proj_weight=use_separate_proj_weight,
                        use_rope_embeddings=self.use_rope_embeddings,
                        use_gated_attention=use_gated_attention,
                        pos_dimension=rope_pos_dimension,
                        rope_base=rope_base,
                    )
                    for _ in range(message_passing_num)
                ]
            )
        else:
            # 非DGL環境：PyTorch Geometric標準の MessagePassing型グラフアテンション（TransformerConv）で代用
            self.processor_list = nn.ModuleList(
                [
                    TransformerConv(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        heads=num_heads,
                        concat=False, # 各ヘッド出力を結合せず平均化する設定
                        beta=True,   # 残差接続用のスケーリング因数を有効化
                    )
                    for _ in range(message_passing_num)
                ]
            )

        # DGLがないために機能がオミットされる場合の警告ログ群
        if self._requested_rope and not HAS_DGL_SPARSE:
            logger.warning("use_rope_embeddings=True ですが DGL sparse バックエンドがないため、RoPEは無視されます。")
        if use_gated_attention and not HAS_DGL_SPARSE:
            logger.warning("use_gated_attention=True ですが DGL sparse バックエンドがないため、ゲートアテンションは無視されます。")
        if use_temporal_block and not HAS_DGL_SPARSE:
            logger.warning("use_temporal_block=True ですが DGL sparse バックエンドがないため、時間アテンションは隣接関係なしで動作します。")
            
        self.temporal_block = (
            TemporalAttention(hidden_size=hidden_size, num_heads=num_heads)
            if self.use_temporal_block
            else None
        )

    def _parse_dilation_config(
        self, raw_config
    ) -> Dict[int, Dict[int, int]]:
        """Dilation設定を layer -> head -> hop の対応へ正規化する。"""
        if raw_config is None:
            return {}
        configs = raw_config if isinstance(raw_config, list) else [raw_config]
        layer_head_hops: Dict[int, Dict[int, int]] = {}

        for config in configs:
            if not isinstance(config, dict):
                raise ValueError("dilation は辞書または辞書のリストで指定してください。")
            if not config.get("enabled", True):
                continue

            hop = int(config.get("hop", 2))
            if hop < 2:
                raise ValueError("dilation.hop は2以上である必要があります。")

            layers = config.get("layers")
            if layers is None and "start_layer" in config:
                layers = range(int(config["start_layer"]), self.num_layers)
            if layers is None:
                raise ValueError("dilation.layers または dilation.start_layer が必要です。")
            layers = [int(layer) for layer in layers]
            if any(layer < 0 or layer >= self.num_layers for layer in layers):
                raise ValueError(
                    f"dilation.layers は0から{self.num_layers - 1}の範囲で指定してください。"
                )

            head_indices = config.get("head_indices")
            if head_indices is None:
                head_fraction = float(config.get("head_fraction", 0.5))
                if not 0.0 < head_fraction <= 1.0:
                    raise ValueError("dilation.head_fraction は0より大きく1以下にしてください。")
                dilated_head_count = max(
                    1, min(self.num_heads, int(round(self.num_heads * head_fraction)))
                )
                head_indices = range(dilated_head_count)
            head_indices = [int(head) for head in head_indices]
            if any(head < 0 or head >= self.num_heads for head in head_indices):
                raise ValueError(
                    f"dilation.head_indices は0から{self.num_heads - 1}の範囲で指定してください。"
                )

            for layer in layers:
                assignments = layer_head_hops.setdefault(layer, {})
                for head in head_indices:
                    if head in assignments and assignments[head] != hop:
                        raise ValueError(
                            f"layer {layer}, head {head} に複数のdilation hopが指定されています。"
                        )
                    assignments[head] = hop

        return layer_head_hops

    def _build_attention_edge_indices(
        self, graph: Data, num_nodes: int
    ) -> tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """通常head用と各Dilation hop用の疎なedge_indexをforwardごとに構築する。"""
        base_edges = graph.edge_index.to(dtype=torch.long)
        batch = getattr(graph, "batch", None)
        random_edges = generate_random_edges(
            base_edges,
            num_nodes=num_nodes,
            ratio=self.random_edges_ratio,
            batch=batch,
        )

        global_edges = base_edges.new_empty((2, 0))
        if self.global_attention_ratio > 0.0 and self.global_node_types:
            node_type = getattr(graph, "node_type", None)
            if node_type is None:
                raise ValueError(
                    "Global Attentionを使用するには入力グラフの node_type が必要です。"
                )
            global_edges = generate_global_attention_edges(
                node_type=node_type,
                num_nodes=num_nodes,
                global_node_types=self.global_node_types,
                sample_ratio=self.global_attention_ratio,
                batch=batch,
            )

        common_edges = merge_edge_indices(
            [random_edges, global_edges], num_nodes=num_nodes
        ).to(base_edges.device)
        regular_edges = merge_edge_indices(
            [base_edges, common_edges], num_nodes=num_nodes
        ).to(base_edges.device)

        dilated_edges: Dict[int, torch.Tensor] = {}
        required_hops = {
            hop for assignments in self.layer_head_hops.values() for hop in assignments.values()
        }
        for hop in required_hops:
            exact_hop_edges = compute_exact_k_hop_edge_index(
                base_edges, num_hops=hop, num_nodes=num_nodes
            )
            dilated_edges[hop] = merge_edge_indices(
                [exact_hop_edges, common_edges], num_nodes=num_nodes
            ).to(base_edges.device)

        return regular_edges, dilated_edges

    @staticmethod
    def _to_dgl_adjacency(edge_index: torch.Tensor, x: torch.Tensor):
        values = torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device)
        return dglsp.spmatrix(
            indices=edge_index, val=values, shape=(x.shape[0], x.shape[0])
        )

    def forward(self, graph: Data) -> torch.Tensor:
        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None:
            raise ValueError("入力グラフに edge_index が必要です。")

        if self.only_processor:
            x = graph.x
        else:
            x = self.nodes_encoder(graph.x)

        pos = getattr(graph, "pos", None)
        if self.use_rope_embeddings and pos is None:
            raise ValueError(
                "use_rope_embeddings=True の場合、入力グラフに `pos` 属性が必要です。"
            )
        if self.use_rope_embeddings:
            if pos.size(-1) < self.rope_pos_dimension:
                raise ValueError(
                    f"要求されたRoPE軸数 {self.rope_pos_dimension} に対し、"
                    f"入力の座標次元数が {pos.size(-1)} と不足しています。"
                )
            # 元の物理座標は変更せず、RoPEに使う軸だけを代表長さで無次元化する。
            pos = pos[..., : self.rope_pos_dimension] / self.rope_length_scale

        prev_x = x
        last_x = x
        adj = None
        regular_edges, dilated_edges = self._build_attention_edge_indices(
            graph, num_nodes=x.shape[0]
        )

        # バックエンドに応じた条件分岐型のレイヤー順伝播
        if HAS_DGL_SPARSE:
            # DGL Sparseによる疎行列形式でのアテンションループ
            regular_adj = self._to_dgl_adjacency(regular_edges, x)
            dilated_adjs = {
                hop: self._to_dgl_adjacency(edges, x)
                for hop, edges in dilated_edges.items()
            }
            adj = regular_adj
            for layer_index, block in enumerate(self.processor_list):
                head_hops = self.layer_head_hops.get(layer_index)
                layer_adj = regular_adj
                if head_hops:
                    layer_adj = [regular_adj] * self.num_heads
                    for head_index, hop in head_hops.items():
                        layer_adj[head_index] = dilated_adjs[hop]
                prev_x = x
                last_x = block(prev_x, layer_adj, pos=pos)
                x = last_x
        else:
            # PyGによるトポロジーエッジインデックス形式でのメッセージ集約型アテンションループ
            for layer_index, block in enumerate(self.processor_list):
                layer_edges = regular_edges
                head_hops = self.layer_head_hops.get(layer_index)
                if head_hops:
                    unique_hops = set(head_hops.values())
                    if len(head_hops) != self.num_heads or len(unique_hops) != 1:
                        raise RuntimeError(
                            "head別DilationにはDGL Sparseバックエンドが必要です。"
                        )
                    layer_edges = dilated_edges[next(iter(unique_hops))]
                prev_x = x
                last_x = block(prev_x, layer_edges)
                x = last_x

        if self.use_temporal_block and self.temporal_block is not None:
            x = self.temporal_block(prev_x, last_x, adj)

        if self.only_processor:
            return x
        else:
            x_decoded = self.decode_module(x)
            return x_decoded


class TransolverProcessor(nn.Module):
    """
    外部先進モデル「Transolver++」をリポジトリのパイプラインに適応させるためのラッパークラス。
    
    Transolverは、メッシュ全体の物理場をスライス（量子化）して、大規模な3D幾何でも
    高精度かつ高速に解く物理特化型Transformer。
    """
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_heads: int = 2,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_rope_embeddings = use_rope_embeddings

        # 内部でオリジナルの Transolver モデル構造を組み上げ、全設定を転送
        self.model = Transolver.Model(
            space_dim=0, # 入力特徴量内で位置と場が融合されているため0に初期化
            n_layers=message_passing_num,
            n_hidden=hidden_size,
            dropout=dropout,
            n_head=num_heads,
            act="gelu",
            mlp_ratio=mlp_ratio,
            fun_dim=node_input_size,
            out_dim=output_size,
            slice_num=slice_num,
            ref=ref,
            unified_pos=unified_pos,
            use_rope_embeddings=use_rope_embeddings,
            use_gated_attention=use_gated_attention,
            rope_pos_dimension=rope_pos_dimension,
            rope_base=rope_base,
            use_temporal_block=use_temporal_block,
        )

    def forward(self, graph: Data) -> torch.Tensor:
        """
        PyGの Data 構造（2次元フラット）を、Transolverが要求する
        バッチ次元（B=1）を含んだ3次元テンソルに拡張して推論を通す。
        """
        # (NumNodes, FeatureDim) -> (1, NumNodes, FeatureDim) に次元拡張
        x_batched = graph.x.unsqueeze(0)  
        pos_batched = (
            graph.pos.unsqueeze(0) if graph.pos is not None else None
        )  # (1, NumNodes, 3)
        
        condition = None  # グローバル条件ベクトル（現状は未使用のプレースホルダー）
        
        if self.use_rope_embeddings and pos_batched is None:
            raise ValueError(
                "use_rope_embeddings=True の場合、入力グラフに `pos` 属性が必要です。"
            )

        # オリジナルモデルの順伝播を実行
        out = self.model.forward(x_batched, pos_batched, condition)
        
        # 出力を元のフラットな形状に戻す (1, NumNodes, OutDim) -> (NumNodes, OutDim)
        out = out.squeeze(0)  
        return out


class EagleProcessor(nn.Module):
    """PyG adapter for the EAGLE mesh transformer.

    Each graph in a PyG batch is processed independently so global cluster
    attention never mixes nodes from different simulations. If ``graph``
    contains a one-dimensional ``cluster`` attribute, those labels are used as
    the precomputed partition. Otherwise deterministic spatial clusters are
    built from ``graph.pos``.
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        token_size: Optional[int] = None,
        cluster_size: int = 20,
        space_dim: int = 2,
        eagle_gnn_layers: int = 4,
        pos_frequencies: int = 8,
        pos_frequency_start: int = -3,
    ):
        super().__init__()
        self.output_size = output_size
        self.model = EagleMeshTransformer(
            node_input_size=node_input_size,
            output_size=output_size,
            space_dim=space_dim,
            latent_size=hidden_size,
            token_size=token_size or hidden_size * 4,
            num_attention_layers=message_passing_num,
            num_gnn_layers=eagle_gnn_layers,
            num_heads=num_heads,
            cluster_size=cluster_size,
            num_position_frequencies=pos_frequencies,
            position_frequency_start=pos_frequency_start,
            dropout=dropout,
        )

    @property
    def nodes_encoder(self) -> nn.Module:
        """Expose the encoder under the standard GraphPhysics hook name."""

        return self.model.encoder.node_encoder

    @property
    def decode_module(self) -> nn.Module:
        """Expose the output head under the standard GraphPhysics hook name."""

        return self.model.retriever.output_mlp

    def forward(self, graph: Data) -> torch.Tensor:
        if getattr(graph, "pos", None) is None:
            raise ValueError("EAGLEモデルには入力グラフの `pos` 属性が必要です。")
        if getattr(graph, "edge_index", None) is None:
            raise ValueError("EAGLEモデルには入力グラフの `edge_index` が必要です。")

        num_nodes = graph.x.shape[0]
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=graph.x.device)
        if batch.ndim != 1 or batch.shape[0] != num_nodes:
            raise ValueError("graph.batch は [num_nodes] である必要があります。")

        source, target = graph.edge_index
        if graph.edge_index.numel() and torch.any(batch[source] != batch[target]):
            raise ValueError("異なるPyGバッチ要素を結ぶedgeは使用できません。")

        cluster_labels = getattr(graph, "cluster", None)
        if cluster_labels is not None and (
            cluster_labels.ndim != 1 or cluster_labels.shape[0] != num_nodes
        ):
            raise ValueError(
                "graph.cluster は各ノードに1つのラベルを持つ [num_nodes] "
                "テンソルである必要があります。"
            )

        output = graph.x.new_empty((num_nodes, self.output_size))
        global_to_local = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=graph.x.device
        )

        for batch_id in torch.unique(batch, sorted=True):
            node_indices = torch.nonzero(batch == batch_id, as_tuple=False).flatten()
            global_to_local.fill_(-1)
            global_to_local[node_indices] = torch.arange(
                node_indices.numel(), device=graph.x.device
            )

            edge_mask = (batch[source] == batch_id) & (batch[target] == batch_id)
            local_edge_index = global_to_local[graph.edge_index[:, edge_mask]]

            clusters = None
            cluster_mask = None
            if cluster_labels is not None:
                clusters, cluster_mask = clusters_from_labels(
                    cluster_labels[node_indices]
                )

            output[node_indices] = self.model(
                graph.x[node_indices],
                graph.pos[node_indices],
                local_edge_index,
                clusters=clusters,
                cluster_mask=cluster_mask,
            )
        return output

# Ubuntu 22.04 + CUDA 12.4.1
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DGLBACKEND=pytorch

# システムパッケージ（PyVistaのレンダリングやHDF5に必要なOSライブラリを網羅）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential libhdf5-dev libgl1-mesa-glx libglib2.0-0 \
    libxrender1 libxtst6 libxi6 \
    python3.10 python3.10-dev python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python 3.10 仮想環境の作成
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1. 【コア】PyTorch公式の指定通りに、PyTorch 2.4.0 安定ペアを完璧にインストール
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 2. 【DGL】公式の PyTorch 2.4 + CUDA 12.4 専用URLからインストール
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# 3. 【PyG】PyTorch 2.4.0 + CUDA 12.4 専用のバイナリを固定
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# 4. 残りの依存関係を requirements.txt からインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

CMD ["/bin/bash"]
# PTCG AI Battle Challenge - 開発/学習用イメージ
# ベースイメージ・バージョンはハードコードせず build ARG で差し替え可能にする。
#   既定値: PyTorch 公式 CUDA ランタイム。
#   GPU: GeForce GTX 1060 (Pascal / sm_61)、Driver 581.57 (CUDA 13 互換) を想定し、
#        Pascal でも安定動作する CUDA 12.4 ビルドを既定とする。
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 作業ディレクトリも ARG で変更可能に
ARG WORKDIR=/workspace
WORKDIR ${WORKDIR}

# 最低限の OS パッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依存関係（PyTorch 本体はベースイメージに同梱されているため requirements には含めない）
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ソースは compose の bind mount で開発時に同期する想定。
# 本番ビルドで固める場合は下行を有効化:
# COPY . ${WORKDIR}

CMD ["bash"]

#!/bin/bash

# 0. 升级 pip 以避免安装时的构建错误
echo "[0/4] Upgrading pip..."
pip install --upgrade pip

# 1. 安装基础依赖
echo "[1/4] Installing base requirements..."
pip install -r requirements.txt


# 2. 安装 PyTorch 2.3.0 (GPU版 - CUDA 12.1)
echo "[2/4] Installing PyTorch 2.3.0 (CUDA 12.1)..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 PyG (Graph) 依赖 (适配 CUDA 12.1)
echo "[3/4] Installing PyG dependencies..."
pip install torch-scatter==2.1.2 \
            torch-sparse==0.6.18 \
            -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 安装 PyG 主库 
pip install torch-geometric==2.5.3 torchdata==0.7.1

# 4. 安装 DGL 2.2.1 (适配 CUDA 12.1)
echo "[4/4] Installing DGL 2.2.1 (CUDA 12.1)..."
# DGL 也需要指定 CUDA 12.1 的源
pip install dgl==2.2.1 \
    -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

echo "✅ Environment installation complete (GPU Mode)!"
# 验证安装
python -c "import torch; import dgl; print(f'Torch: {torch.__version__}, DGL: {dgl.__version__}')"
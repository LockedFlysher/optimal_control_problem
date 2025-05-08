#!/bin/bash
# 一键配置阿里云镜像并拉取NVIDIA CUDA 12.8.1镜像

echo "开始配置阿里云Docker镜像源并拉取NVIDIA CUDA 12.8.1镜像..."

# 配置阿里云镜像源
echo "配置阿里云Docker镜像源..."
sudo mkdir -p /etc/docker
cat << EOF | sudo tee /etc/docker/daemon.json
{
  "registry-mirrors": ["https://registry.cn-hangzhou.aliyuncs.com"]
}
EOF

# 重启Docker服务
echo "重启Docker服务..."
sudo systemctl daemon-reload
sudo systemctl restart docker

# 等待Docker重启完成
sleep 3

# 拉取NVIDIA CUDA镜像
echo "开始拉取NVIDIA CUDA 12.8.1镜像..."
sudo docker pull nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# 检查镜像是否拉取成功
if sudo docker images | grep -q "nvidia/cuda.*12.8.1"; then
  echo "✅ NVIDIA CUDA 12.8.1镜像拉取成功！"
  sudo docker images | grep "nvidia/cuda.*12.8.1"
else
  echo "❌ 镜像拉取失败，尝试备用方案..."
  echo "尝试使用其他镜像源..."

  cat << EOF | sudo tee /etc/docker/daemon.json
{
  "registry-mirrors": [
    "https://registry.cn-hangzhou.aliyuncs.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}
EOF

  sudo systemctl daemon-reload
  sudo systemctl restart docker
  sleep 3

  echo "尝试拉取备用版本..."
  sudo docker pull nvidia/cuda:12.8.1-base-ubuntu22.04

  if sudo docker images | grep -q "nvidia/cuda.*12.8.1"; then
    echo "✅ 备用NVIDIA CUDA 12.8.1镜像拉取成功！"
    sudo docker images | grep "nvidia/cuda.*12.8.1"
  else
    echo "❌ 所有尝试均失败，请检查网络连接或手动尝试其他CUDA版本。"
  fi
fi

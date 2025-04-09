#!/bin/bash

# 配置路径
OSQP_SRC_DIR="/home/andew/tool/osqp-v1.0.0.beta1-src"
OSQP_BUILD_DIR="${OSQP_SRC_DIR}/build"
OSQP_EIGEN_SRC_DIR="/home/andew/tool/osqp-eigen-0.9.0"
OSQP_EIGEN_BUILD_DIR="${OSQP_EIGEN_SRC_DIR}/build"

# 安装结果状态
osqp_result="\033[31m未安装\033[0m"
osqp_eigen_result="\033[31m未安装\033[0m"

# 打印分隔线函数
print_separator() {
    echo "=================================================="
}

# 安装OSQP
install_osqp() {
    echo -e "\n\033[34m开始安装OSQP...\033[0m"
    source ~/.bashrc
    
    # 清理构建目录
    echo "正在清理OSQP构建目录..."
    if [ -d "$OSQP_BUILD_DIR" ]; then
        rm -rf "${OSQP_BUILD_DIR:?}/"*
    else
        mkdir -p "$OSQP_BUILD_DIR"
    fi

    cd "$OSQP_BUILD_DIR" || return 1

    # CMake配置
    if ! cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DOSQP_BUILD_SHARED_LIB=ON \
        -DOSQP_BUILD_STATIC_LIB=ON \
        -DOSQP_ALGEBRA_BACKEND=cuda \
        -DOSQP_ENABLE_PRINTING=ON \
        -DOSQP_ENABLE_PROFILING=ON \
        -DOSQP_ENABLE_INTERRUPT=ON \
        -DOSQP_CODEGEN=ON \
        -DOSQP_ENABLE_DERIVATIVES=ON \
        -DOSQP_USE_FLOAT=ON
    then
        echo -e "\033[31mOSQP CMake配置失败！\033[0m"
        return 1
    fi

    # 编译
    if ! make -j$(nproc); then
        echo -e "\033[31mOSQP编译失败！\033[0m"
        return 1
    fi

    # 安装
    if sudo make install; then
        osqp_result="\033[32m安装成功\033[0m"
        echo -e "\033[32mOSQP安装完成！\033[0m"
        return 0
    else
        echo -e "\033[31mOSQP安装失败！\033[0m"
        return 1
    fi
}

# 安装OSQP-Eigen
install_osqp_eigen() {
    echo -e "\n\033[34m开始安装OSQP-Eigen...\033[0m"
    
    # 清理构建目录
    echo "正在清理OSQP-Eigen构建目录..."
    if [ -d "$OSQP_EIGEN_BUILD_DIR" ]; then
        rm -rf "${OSQP_EIGEN_BUILD_DIR:?}/"*
    else
        mkdir -p "$OSQP_EIGEN_BUILD_DIR"
    fi

    cd "$OSQP_EIGEN_BUILD_DIR" || return 1

    # CMake配置
    if ! cmake .. \
        -DOSQP_DIR=/usr/local \
        -DCMAKE_CXX_FLAGS="-L/usr/local/cuda/lib64"
    then
        echo -e "\033[31mOSQP-Eigen CMake配置失败！\033[0m"
        return 1
    fi

    # 编译
    if ! make -j$(nproc); then
        echo -e "\033[31mOSQP-Eigen编译失败！\033[0m"
        return 1
    fi

    # 安装
    if sudo make install; then
        osqp_eigen_result="\033[32m安装成功\033[0m"
        echo -e "\033[32mOSQP-Eigen安装完成！\033[0m"
        return 0
    else
        echo -e "\033[31mOSQP-Eigen安装失败！\033[0m"
        return 1
    fi
}

# 主执行流程
print_separator
if install_osqp; then
    print_separator
    echo -e "\n\033[33m按i键继续安装OSQP-Eigen，其他键退出...\033[0m"
    read -n 1 -s -r key
    if [[ $key == "i" ]]; then
        if install_osqp_eigen; then
            : # 安装成功继续
        fi
    else
        echo -e "\n\033[33m已跳过OSQP-Eigen安装\033[0m"
    fi
fi

# 显示最终结果
print_separator
echo -e "\n\033[36m—— 安装结果汇总 ——\033[0m"
echo -e "OSQP状态:        ${osqp_result}"
echo -e "OSQP-Eigen状态: ${osqp_eigen_result}"
print_separator

# 提示颜色说明
echo -e "\n\033[32m绿色\033[0m: 成功状态 | \033[31m红色\033[0m: 失败/未安装状态"

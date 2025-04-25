#!/bin/bash

# Print separator function
print_separator() {
    echo "=================================================="
}

# Print colored status messages
print_status() {
    local status=$1
    local message=$2

    if [ "$status" -eq 0 ]; then
        echo -e "\033[32m✓ SUCCESS: $message\033[0m"
    else
        echo -e "\033[31m✗ FAILED: $message\033[0m"
    fi
}

# Current directory (should be installation folder)
INSTALL_DIR="$(pwd)"

# Step 1: Install CasADi
install_casadi() {
    print_separator
    echo -e "\033[34mStep 1: Installing CasADi...\033[0m"

    if [ -d "casadi" ]; then
        echo "CasADi directory already exists. Using existing directory..."
    else
        echo "Cloning CasADi repository..."
        git clone https://github.com/casadi/casadi.git -b main casadi
    fi

    cd casadi
    mkdir -p build
    cd build

    echo "Configuring CasADi..."
    cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DWITH_IPOPT=ON -DWITH_QPOASES=ON -DWITH_LAPACK=ON ..

    echo "Building and installing CasADi..."
    make -j$(nproc) && sudo make install
    casadi_status=$?

    cd $INSTALL_DIR
    print_status $casadi_status "CasADi installation"
    return $casadi_status
}

# Step 2: Install libtorch
install_libtorch() {
    print_separator
    echo -e "\033[34mStep 2: Installing libtorch...\033[0m"

    LIBTORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-2.7.0+cu126.zip"

    # Check if libtorch directory already exists
    if [ -d "libtorch" ]; then
        echo "libtorch directory already exists. Using existing directory..."
    else
        # Check if the zip file already exists
        if [ -f "$LIBTORCH_ZIP" ]; then
            echo "libtorch zip file already exists. Extracting..."
        else
            echo "Downloading libtorch..."
            wget -q --show-progress https://download.pytorch.org/libtorch/cu126/$LIBTORCH_ZIP
        fi

        echo "Extracting libtorch..."
        unzip -q $LIBTORCH_ZIP
    fi

    # Apply patch only if the file doesn't exist yet
    if [ ! -f "./libtorch/include/c10/util/logging_is_not_google_glog.h" ]; then
        echo "Applying patch to libtorch..."
        cp ./logging_is_not_google_glog.h ./libtorch/include/c10/util/logging_is_not_google_glog.h
    else
        echo "libtorch patch already applied. Skipping..."
    fi

    echo "Installing libtorch..."
    sudo cp -r ./libtorch/lib/* /usr/local/lib/
    sudo cp -r ./libtorch/include/* /usr/local/include/
    sudo cp -r ./libtorch/share/* /usr/local/share/
    sudo ldconfig

    libtorch_status=0
    print_status $libtorch_status "libtorch installation"
    return $libtorch_status
}

# Step 3: Install OSQP
install_osqp() {
    print_separator
    echo -e "\033[34mStep 3: Installing OSQP...\033[0m"

    # Check for different possible OSQP directory names
    OSQP_POSSIBLE_DIRS=("./OSQP" "./osqp" "./osqp-v1.0.0.beta1-src")
    OSQP_SRC_DIR=""

    for dir in "${OSQP_POSSIBLE_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            OSQP_SRC_DIR="$dir"
            echo "Found OSQP directory at: $OSQP_SRC_DIR"
            break
        fi
    done

    if [ -z "$OSQP_SRC_DIR" ]; then
        echo "OSQP directory not found. Cloning from specified repository..."
        git clone https://github.com/LockedFlysher/OSQP.git -b master OSQP
        OSQP_SRC_DIR="./OSQP"
    fi

    cd $OSQP_SRC_DIR
    mkdir -p build
    cd build

    # Check if CUDA exists in common locations
    CUDA_PATH=""
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    elif [ -d "/usr/cuda" ]; then
        CUDA_PATH="/usr/cuda"
    fi

    # Only use CUDA backend if CUDA is found
    if [ -n "$CUDA_PATH" ] && [ "$ALGEBRA_BACKEND" = "cuda" ]; then
        echo "CUDA found at: $CUDA_PATH"
        echo "Configuring OSQP with CUDA backend..."
        export CUDA_BIN_PATH="$CUDA_PATH/bin"
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DOSQP_BUILD_SHARED_LIB=ON \
            -DOSQP_BUILD_STATIC_LIB=ON \
            -DOSQP_ALGEBRA_BACKEND=cuda \
            -DOSQP_ENABLE_PRINTING=ON \
            -DOSQP_ENABLE_PROFILING=ON \
            -DOSQP_ENABLE_INTERRUPT=ON \
            -DOSQP_CODEGEN=ON \
            -DOSQP_ENABLE_DERIVATIVES=ON \
            -DOSQP_USE_FLOAT=ON \
            -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc"
    else
        if [ "$ALGEBRA_BACKEND" = "cuda" ]; then
            echo "CUDA not found. Falling back to CPU-only mode."
            ALGEBRA_BACKEND="builtin"
        fi

        echo "Configuring OSQP with CPU backend..."
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DOSQP_BUILD_SHARED_LIB=ON \
            -DOSQP_BUILD_STATIC_LIB=ON \
            -DOSQP_ALGEBRA_BACKEND=builtin \
            -DOSQP_ENABLE_PRINTING=ON \
            -DOSQP_ENABLE_PROFILING=ON \
            -DOSQP_ENABLE_INTERRUPT=ON \
            -DOSQP_CODEGEN=ON \
            -DOSQP_ENABLE_DERIVATIVES=ON \
            -DOSQP_USE_FLOAT=ON
    fi

    echo "Building and installing OSQP..."
    make -j$(nproc) && sudo make install
    osqp_status=$?

    cd $INSTALL_DIR
    print_status $osqp_status "OSQP installation"
    return $osqp_status
}

# Step 4: Install OSQP_Eigen
install_osqp_eigen() {
    print_separator
    echo -e "\033[34mStep 4: Installing OSQP_Eigen...\033[0m"

    # Check for different possible OSQP_Eigen directory names
    OSQP_EIGEN_POSSIBLE_DIRS=("./OSQP_Eigen" "./osqp-eigen" "./osqp_eigen" "./osqp-eigen-0.9.0")
    OSQP_EIGEN_SRC_DIR=""

    for dir in "${OSQP_EIGEN_POSSIBLE_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            OSQP_EIGEN_SRC_DIR="$dir"
            echo "Found OSQP_Eigen directory at: $OSQP_EIGEN_SRC_DIR"
            break
        fi
    done

    if [ -z "$OSQP_EIGEN_SRC_DIR" ]; then
        echo "OSQP_Eigen directory not found. Cloning from specified repository..."
        git clone https://github.com/LockedFlysher/OSQP_Eigen.git -b master OSQP_Eigen
        OSQP_EIGEN_SRC_DIR="./OSQP_Eigen"
    fi

    cd $OSQP_EIGEN_SRC_DIR
    mkdir -p build
    cd build

    # Check if CUDA exists in common locations
    CUDA_PATH=""
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    elif [ -d "/usr/cuda" ]; then
        CUDA_PATH="/usr/cuda"
    fi

    if [ -n "$CUDA_PATH" ] && [ "$ALGEBRA_BACKEND" = "cuda" ]; then
        echo "Configuring OSQP_Eigen with CUDA support..."
        # 使用与您之前成功脚本相同的配置
        cmake .. \
            -DOSQP_DIR=/usr/local \
            -DCMAKE_CXX_FLAGS="-L$CUDA_PATH/lib64"
    else
        echo "Configuring OSQP_Eigen without CUDA support..."
        cmake .. -DOSQP_DIR=/usr/local
    fi

    echo "Building and installing OSQP_Eigen..."
    make -j$(nproc) && sudo make install
    osqp_eigen_status=$?

    cd $INSTALL_DIR
    print_status $osqp_eigen_status "OSQP_Eigen installation"
    return $osqp_eigen_status
}

# Main installation process
main() {
    echo -e "\033[1;36mOptimalControlProblem Installation Script\033[0m"
    echo "This script will install all required dependencies."
    print_separator

    # Ask user to choose between CUDA and CPU
    echo -e "\033[1;33mPlease select installation mode:\033[0m"
    echo "1) CUDA (Requires NVIDIA GPU and CUDA toolkit)"
    echo "2) CPU only"
    read -p "Enter your choice (1/2): " choice

    case $choice in
        1)
            echo "Selected CUDA installation mode."
            ALGEBRA_BACKEND="cuda"
            ;;
        2)
            echo "Selected CPU-only installation mode."
            ALGEBRA_BACKEND="builtin"
            ;;
        *)
            echo "Invalid choice. Defaulting to CPU-only installation."
            ALGEBRA_BACKEND="builtin"
            ;;
    esac

    # Install CasADi
    install_casadi
    casadi_result=$?

    # Install libtorch
    install_libtorch
    libtorch_result=$?

    # Install OSQP
    install_osqp
    osqp_result=$?

    # Install OSQP_Eigen
    install_osqp_eigen
    osqp_eigen_result=$?

    # Installation summary
    print_separator
    echo -e "\033[1;36mInstallation Summary\033[0m"
    echo "Installation mode: $([ "$ALGEBRA_BACKEND" = "cuda" ] && echo "CUDA" || echo "CPU-only")"
    print_status $casadi_result "CasADi"
    print_status $libtorch_result "libtorch"
    print_status $osqp_result "OSQP"
    print_status $osqp_eigen_result "OSQP_Eigen"

    # Final message
    if [ $casadi_result -eq 0 ] && [ $libtorch_result -eq 0 ] && [ $osqp_result -eq 0 ] && [ $osqp_eigen_result -eq 0 ]; then
        echo -e "\n\033[32mAll components were installed successfully!\033[0m"
        echo "You can now use the OptimalControlProblem library."
    else
        echo -e "\n\033[31mSome components failed to install. Please check the errors above.\033[0m"
    fi

    print_separator
}

# Run the main function
main

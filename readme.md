# OptimalControlProblem 类说明文档

## 简介

OptimalControlProblem 是一个用于构建和求解最优控制问题的基类，它提供了：
- 优化变量的管理
- 约束条件管理
- 代价函数的构建
- 变量边界条件的设置

目前可以选择的求解器有casadi封装的IPOPT、SQPMethod求解器，以及单独兼容扩展的SQP+CUDAADMM的基于OSQP的求解器

## 安装

使用脚本编译并安装libtorch、OSQP、OSQP-Eigen，根据需求选择使用OSQP的UDA版本或者CPU版本
基础安装
克隆本仓库
```shell
git clone https://github.com/LockedFlysher/optimal_control_problem.git -b master
cd optimal_control_problem
```
```shell
cd installation
git clone https://github.com/casadi/casadi.git -b main casadi
cd casadi
mkdir build
cd build
cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DWITH_IPOPT=ON -DWITH_QPOASES=ON -DWITH_LAPACK=ON .. 
make -j8 && sudo make install
```
libtorch用于系统矩阵的更新，但是要做代码修改
```shell
cd installation
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu126.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu126.zip
cp $(pwd)/logging_is_not_google_glog.h $(pwd)/libtorch/include/c10/utils/logging_is_not_google_glog.h
sudo cp -r $(pwd)/libtorch/lib/* /usr/local/lib/
sudo cp -r $(pwd)/libtorch/include/* /usr/local/include/
sudo cp -r $(pwd)/libtorch/share/* /usr/local/share/
sudo ldconfig
# libtorch完成安装
sudo rm -r libtorch
```
# OSQP 
```shell
cd installation
git clone https://github.com/LockedFlysher/OSQP.git -b master
cd OSQP 
mkdir build
cd build
cmake ..   -DCMAKE_BUILD_TYPE=Release   -DOSQP_BUILD_SHARED_LIB=ON   -DOSQP_BUILD_STATIC_LIB=ON   -DOSQP_ALGEBRA_BACKEND=cuda   -DOSQP_ENABLE_PRINTING=ON   -DOSQP_ENABLE_PROFILING=ON   -DOSQP_ENABLE_INTERRUPT=ON   -DOSQP_CODEGEN=ON   -DOSQP_ENABLE_DERIVATIVES=ON   -DOSQP_USE_FLOAT=ON
sudo make -j8 && sudo make install
```
# OSQP_Eigen
```shell
cd installation
git clone https://github.com/LockedFlysher/OSQP_Eigen.git -b master
cd OSQP_Eigen
mkdir build
cd build
cmake .. -DCMAKE_CXX_FLAGS="-L/usr/local/cuda/lib64" && make -j8 && sudo make install
```
到包目录下执行两种不同的脚本即可完成切换
```shell
sh cpu_install.sh
```
```shell
sh cuda_install.sh
```

## 配置示例（YAML）
optimal_control_problem的构造函数参数为一个YAML::Node，词条示例如下
```yaml
optimal_control_problem:
  discretization_settings:
    # 预测时间长度为dt*horizon
    dt: 0.005
    horizon: 20
  solver_settings:
    verbose: false
    gen_code: false
    #  如果是recompile则会在运行时重新生成.so文件
    recompile: false
    load_lib: false
    #  求解器的方案有很多种： IPOPT, SQP, CUDA_SQP, MIXED,对应不同的求解方式
    solve_method: CUDA_SQP
    #  如果是CUDA_SQP，需要在下面进行设置，仅有CUDA_SQP是需要设置的
    SQP_step: 0.1
    ADMM_step: 10
```

## 主要贡献：
- 实现SQP系统矩阵计算与更新，将结果赋给Eigen-OSQP，再利用OSQP的CUDA backend做显卡上的优化求解
---
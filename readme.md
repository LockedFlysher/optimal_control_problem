# OptimalControlProblem 类说明文档

## 简介

OptimalControlProblem 是一个用于构建和求解最优控制问题的基类，它提供了：
- 优化变量的管理
- 约束条件管理
- 代价函数的构建
- 变量边界条件的设置

目前可以选择的求解器有casadi封装的IPOPT、SQPMethod求解器，以及单独兼容扩展的SQP+CUDAADMM的基于OSQP的求解器

## 安装
CasADi支持
```shell
git clone https://github.com/casadi/casadi.git -b main casadi
cd casadi
mkdir build && cd build
cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DWITH_IPOPT=ON -DWITH_QPOASES=ON -DWITH_LAPACK=ON .. 
make -j8
sudo make install
```

下载OSQP与OSQP-Eigen依赖
```shell
git clone https://github.com/osqp/osqp.git
git clone https://github.com/robotology/osqp-eigen.git
```
修改cpu_install.sh和cuda_install.sh的路径为上述包的安装路径

## 配置示例（YAML）
optimal_control_problem的构造函数参数为一个YAML::Node，示例如下
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

- 使用新版的CUDA版本OSQP集成CasADi
- 实现SQP系统矩阵更新

---

# OptimalControlProblem Class Documentation

## Introduction

OptimalControlProblem is a base class for building and solving optimal control problems, providing:
- Management of optimization variables
- Management of constraints
- Construction of cost functions
- Setting boundary conditions for variables

Currently available solvers include IPOPT and SQPMethod wrapped by casadi, as well as a separately compatible extension of SQP+CUDAADMM based on the OSQP solver.

## Installation

CasADi Support
```shell
git clone https://github.com/casadi/casadi.git -b main casadi
cd casadi
mkdir build && cd build
cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DWITH_IPOPT=ON -DWITH_QPOASES=ON -DWITH_LAPACK=ON .. 
make -j8
sudo make install
```

Download OSQP and OSQP-Eigen dependencies
```shell
git clone https://github.com/osqp/osqp.git
git clone https://github.com/robotology/osqp-eigen.git
```
Modify the paths in cpu_install.sh and cuda_install.sh to match the installation paths of the packages above.

## Configuration Example (YAML)

The constructor parameter for optimal_control_problem is a YAML::Node, as shown in the example below:
```yaml
optimal_control_problem:
  discretization_settings:
    # Prediction time length is dt*horizon
    dt: 0.005
    horizon: 20
  solver_settings:
    verbose: false
    gen_code: false
    #  If recompile is true, .so files will be regenerated at runtime
    recompile: false
    load_lib: false
    #  There are several solver options: IPOPT, SQP, CUDA_SQP, MIXED, corresponding to different solving methods
    solve_method: CUDA_SQP
    #  If using CUDA_SQP, the following settings are required (only needed for CUDA_SQP)
    SQP_step: 0.1
    ADMM_step: 10
```

## Main Contributions

- Integration of the new CUDA version of OSQP with CasADi
- Implementation of SQP system matrix updates
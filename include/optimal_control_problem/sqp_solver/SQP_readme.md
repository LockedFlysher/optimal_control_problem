# CuCaQP 类接口说明文档

## 概述

`CuCaQP` 是一个用于求解二次规划 (Quadratic Programming, QP) 问题的类，它封装了 OSQP (Operator Splitting Quadratic Program) 求解器，并提供了与 CasADi 库的集成接口。该类主要用于优化控制问题中的二次规划求解。

## 标准二次规划问题形式

```
最小化    (1/2) x^T P x + q^T x
约束条件  l <= A x <= u
```

其中：
- `x` 是优化变量
- `P` 是正定或半正定的 Hessian 矩阵
- `q` 是线性目标向量
- `A` 是线性约束矩阵
- `l` 和 `u` 分别是约束的下界和上界

## 构造函数和析构函数

```cpp
CuCaQP();
~CuCaQP();
```

- **构造函数**：初始化一个空的 QP 求解器实例。
- **析构函数**：清理求解器资源，释放分配的内存。

## 问题设置方法

### 设置问题维度

```cpp
bool setDimension(int numOfVariables, int numOfConstraints);
```

- **功能**：设置优化问题的变量数量和约束数量。
- **参数**：
    - `numOfVariables`：优化变量的数量
    - `numOfConstraints`：约束的数量
- **返回值**：设置是否成功
- **注意**：必须在设置其他问题数据之前调用此方法。如果求解器已初始化，会先清理资源。

### 设置 Hessian 矩阵

```cpp
bool setHessianMatrix(const casadi::DM &hessian);
bool setHessianMatrix(const Eigen::SparseMatrix<OSQPFloat> &P);
```

- **功能**：设置问题的 Hessian 矩阵 P。
- **参数**：
    - `hessian`：CasADi DM 格式的 Hessian 矩阵
    - `P`：Eigen 稀疏矩阵格式的 Hessian 矩阵
- **返回值**：设置是否成功
- **注意**：矩阵维度必须与 `numOfVariables` 一致。

### 设置梯度向量

```cpp
bool setGradient(const casadi::DM &q);
bool setGradient(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &q);
```

- **功能**：设置问题的线性目标向量 q。
- **参数**：
    - `q`：CasADi DM 或 Eigen 向量格式的梯度向量
- **返回值**：设置是否成功
- **注意**：向量维度必须与 `numOfVariables` 一致。

### 设置线性约束矩阵

```cpp
bool setLinearConstraintsMatrix(const casadi::DM &A);
bool setLinearConstraintsMatrix(const Eigen::SparseMatrix<OSQPFloat> &A);
```

- **功能**：设置问题的线性约束矩阵 A。
- **参数**：
    - `A`：CasADi DM 或 Eigen 稀疏矩阵格式的约束矩阵
- **返回值**：设置是否成功
- **注意**：矩阵维度必须为 `numOfConstraints` × `numOfVariables`。

### 设置约束下界

```cpp
bool setLowerBound(const casadi::DM &l);
bool setLowerBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &l);
```

- **功能**：设置约束的下界 l。
- **参数**：
    - `l`：CasADi DM 或 Eigen 向量格式的下界向量
- **返回值**：设置是否成功
- **注意**：向量维度必须与 `numOfConstraints` 一致。

### 设置约束上界

```cpp
bool setUpperBound(const casadi::DM &u);
bool setUpperBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &u);
```

- **功能**：设置约束的上界 u。
- **参数**：
    - `u`：CasADi DM 或 Eigen 向量格式的上界向量
- **返回值**：设置是否成功
- **注意**：向量维度必须与 `numOfConstraints` 一致。

### 一次性设置整个系统

```cpp
void setSystem(DMVector vector1);
```

- **功能**：一次性设置整个 QP 问题的所有参数。
- **参数**：
    - `vector1`：包含 [P, q, A, l, u] 的 CasADi DMVector
- **注意**：在设置新系统之前会清理所有之前分配的资源。

## 更新方法

以下方法用于在不重新分配内存的情况下更新问题数据：

```cpp
bool updateHessianMatrix(const casadi::DM &hessian);
bool updateGradient(const casadi::DM &q);
bool updateLinearConstraintsMatrix(const casadi::DM &A);
bool updateLowerBound(const casadi::DM &l);
bool updateUpperBound(const casadi::DM &u);
```

- **功能**：更新相应的问题参数，而不重新分配内存。
- **参数**：与对应的 set 方法相同。
- **返回值**：更新是否成功。
- **注意**：这些方法比重新设置更高效，但要求求解器已经初始化，且问题结构（非零元素模式）保持不变。

## 求解器参数设置

```cpp
void setVerbosity(bool verbosity);
void setWarmStart(bool warmStart);
void setAbsoluteTolerance(OSQPFloat tolerance);
void setRelativeTolerance(OSQPFloat tolerance);
void setMaxIteration(int maxIteration);
```

- **功能**：设置 OSQP 求解器的各种参数。
- **参数**：
    - `verbosity`：是否输出详细求解信息
    - `warmStart`：是否启用热启动
    - `tolerance`：绝对或相对收敛容差
    - `maxIteration`：最大迭代次数

## 求解方法

```cpp
bool initSolver();
bool solve();
```

- **initSolver**：
    - **功能**：初始化求解器，准备求解。
    - **返回值**：初始化是否成功。
    - **注意**：在设置完所有问题数据后调用，在求解前必须调用。

- **solve**：
    - **功能**：求解当前设置的 QP 问题。
    - **返回值**：求解是否成功。
    - **注意**：必须在 `initSolver()` 之后调用。

## 结果获取

```cpp
Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> getSolution();
casadi::DM getSolutionAsDM();
```

- **getSolution**：
    - **功能**：获取 Eigen 向量格式的解。
    - **返回值**：优化变量的解向量。

- **getSolutionAsDM**：
    - **功能**：获取 CasADi DM 格式的解。
    - **返回值**：优化变量的解向量。

## 调试方法

```cpp
void printSolverData();
```

- **功能**：打印求解器内部数据，用于调试。
- **输出**：包括梯度向量、约束上下界、Hessian 矩阵和约束矩阵的非零元素。

## 使用示例

```cpp
// 创建求解器实例
CuCaQP qpSolver;

// 设置问题维度（变量数量和约束数量）
qpSolver.setDimension(3, 2);

// 设置 Hessian 矩阵 (使用 CasADi DM 格式)
casadi::DM P = casadi::DM::zeros(3, 3);
P(0, 0) = 2.0; P(1, 1) = 1.0; P(2, 2) = 1.0;
qpSolver.setHessianMatrix(P);

// 设置梯度向量
casadi::DM q = casadi::DM::zeros(3, 1);
q(0) = -1.0; q(1) = -2.0; q(2) = -3.0;
qpSolver.setGradient(q);

// 设置约束矩阵
casadi::DM A = casadi::DM::zeros(2, 3);
A(0, 0) = 1.0; A(0, 1) = 1.0; A(0, 2) = 1.0;
A(1, 0) = -1.0; A(1, 1) = -1.0; A(1, 2) = -1.0;
qpSolver.setLinearConstraintsMatrix(A);

// 设置约束上下界
casadi::DM lowerBound = casadi::DM::zeros(2, 1);
lowerBound(0) = 1.0; lowerBound(1) = -INFINITY;
qpSolver.setLowerBound(lowerBound);

casadi::DM upperBound = casadi::DM::zeros(2, 1);
upperBound(0) = INFINITY; upperBound(1) = -1.0;
qpSolver.setUpperBound(upperBound);

// 设置求解器参数
qpSolver.setVerbosity(false);
qpSolver.setAbsoluteTolerance(1e-6);
qpSolver.setRelativeTolerance(1e-6);
qpSolver.setMaxIteration(1000);

// 初始化并求解
if (qpSolver.initSolver() && qpSolver.solve()) {
    // 获取结果
    casadi::DM solution = qpSolver.getSolutionAsDM();
    std::cout << "Solution: " << solution << std::endl;
} else {
    std::cerr << "Failed to solve QP problem." << std::endl;
}
```

## 注意事项

1. 在设置问题数据之前必须先调用 `setDimension` 方法。
2. 在求解之前必须先调用 `initSolver` 方法。
3. 如果需要多次求解具有相同结构但不同数据的问题，可以使用 `update*` 方法更新数据，而不是重新设置。
4. 所有输入矩阵和向量的维度必须与问题维度一致，否则会返回错误。
5. 该类会自动管理内存，在重新设置数据之前会清理之前分配的资源。

# SQPOptimizationSolver 类接口说明文档

## 概述

`SQPOptimizationSolver` 是一个基于序列二次规划 (Sequential Quadratic Programming, SQP) 算法的非线性优化问题求解器。它通过自动微分技术将非线性优化问题在当前迭代点附近线性化，转化为二次规划子问题，并使用 `CuCaQP` 求解器迭代求解。

## 问题形式

该求解器可以处理如下形式的非线性优化问题：

```
最小化    f(x, p)
约束条件  lbg ≤ g(x, p) ≤ ubg
         lbx ≤ x ≤ ubx
```

其中：
- `x` 是优化变量
- `p` 是参考/参数变量（可选）
- `f(x, p)` 是非线性目标函数
- `g(x, p)` 是非线性约束函数
- `lbx`, `ubx` 是变量的上下界
- `lbg`, `ubg` 是约束的上下界

## 构造函数

```cpp
explicit SQPOptimizationSolver(::casadi::SXDict &nlp, ::casadi::Dict &options);
```

### 参数

- **nlp**: 非线性规划问题定义，包含以下键值对：
    - `"f"`: (必需) 目标函数表达式 (SX)
    - `"x"`: (必需) 优化变量 (SX)
    - `"g"`: (可选) 约束函数表达式 (SX)
    - `"p"`: (可选) 参考/参数变量 (SX)

- **options**: 求解器配置选项，包含：
    - `"max_iter"`: 最大迭代次数
    - `"alpha"`: 步长因子 (0 < α ≤ 1)

### 示例
```cpp
// 定义优化变量
casadi::SX x = casadi::SX::sym("x", 2);

// 定义目标函数：f(x) = x1^2 + x2^2
casadi::SX f = x(0)*x(0) + x(1)*x(1);

// 定义约束：g(x) = x1 + x2 - 1
casadi::SX g = x(0) + x(1) - 1;

// 构造问题字典
casadi::SXDict nlp = {
    {"x", x},
    {"f", f},
    {"g", g}
};

// 设置求解器选项
casadi::Dict options = {
    {"max_iter", 100},
    {"alpha", 0.5}
};

// 创建求解器实例
SQPOptimizationSolver solver(nlp, options);
```

## 主要方法

### 获取最优解

```cpp
casadi::DMDict getOptimalSolution(const DMDict &arg);
```

- **功能**：执行 SQP 迭代求解优化问题
- **参数**：
    - `arg`: 包含问题边界条件的字典：
        - `"lbx"`: 变量下界
        - `"ubx"`: 变量上界
        - `"lbg"`: 约束下界
        - `"ubg"`: 约束上界
        - `"p"`: (可选) 参考/参数值
- **返回值**：
    - 包含求解结果的字典：
        - `"x"`: 最优解
        - `"f"`: 目标函数值
- **注意**：
    - 如果是 MPC 问题，初始点通过限制第一项的 `lbx` 和 `ubx` 来设定
    - 求解器使用迭代方法，收敛性取决于初始点和问题性质

### 获取局部系统函数

```cpp
casadi::Function getSXLocalSystemFunction() const;
```

- **功能**：返回用于生成局部二次规划问题的 CasADi Function
- **返回值**：包含以下输入输出的 Function：
    - 输入：[reference, variables, lower_bounds, upper_bounds]
    - 输出：[Hessian, gradient, A, l, u]

## 使用示例

```cpp
// 创建求解器（见构造函数示例）
SQPOptimizationSolver solver(nlp, options);

// 设置求解参数
casadi::DMDict arg = {
    {"lbx", casadi::DM::zeros(2)},      // x ≥ 0
    {"ubx", casadi::DM::inf(2)},        // x < ∞
    {"lbg", casadi::DM::zeros(1)},      // g(x) ≥ 0
    {"ubg", casadi::DM::inf(1)}         // g(x) < ∞
};

// 求解优化问题
casadi::DMDict result = solver.getOptimalSolution(arg);

// 获取结果
casadi::DM x_opt = result.at("x");
casadi::DM f_opt = result.at("f");
```

## 内部工作原理

1. **初始化**：
    - 构造函数创建自动微分器，用于计算目标函数和约束的梯度和 Hessian
    - 设置 QP 求解器参数（收敛容差、最大迭代次数等）

2. **SQP 迭代**：
    - 在每次迭代中：
        1. 计算当前点的局部二次规划近似
        2. 使用 `CuCaQP` 求解局部 QP 问题
        3. 沿搜索方向以步长 α 更新解
        4. 重复直到达到最大迭代次数或收敛

3. **局部系统生成**：
    - `getLocalSystem` 方法生成每次迭代的 QP 问题数据：
        - Hessian 矩阵（目标函数的二阶导数）
        - 梯度向量（目标函数的一阶导数）
        - 线性化约束矩阵和边界

## 注意事项

1. 问题定义：
    - 必须提供目标函数 `f` 和优化变量 `x`
    - 约束函数 `g` 和参数 `p` 是可选的

2. 收敛性：
    - SQP 是局部优化方法，结果依赖于初始点
    - 步长因子 α 影响收敛速度和稳定性
    - 可能需要调整最大迭代次数和收敛容差

3. 内存管理：
    - 求解器自动管理内部资源
    - 使用智能指针管理自动微分器

4. 错误处理：
    - 构造函数检查必需参数
    - 在参数缺失时抛出异常

## 依赖项

- CasADi：用于符号计算和自动微分
- CuCaQP：二次规划求解器
- yaml-cpp：配置文件解析
- ament_index_cpp：ROS 2 包路径查找
# OptimalControlProblem 类说明文档

## 简介

OptimalControlProblem 是一个用于构建和求解最优控制问题的基类，它提供了：
- 状态变量和输入变量的管理
- 约束条件的添加和管理
- 代价函数的构建
- 变量边界条件的设置

## 核心功能

### 1. 初始化与配置

```cpp
OptimalControlProblem(std::string configFilePath)
```

通过 YAML 配置文件初始化最优控制问题：
- 设置预测时域长度 (horizon)
- 初始化状态变量框架 (status frame)
- 初始化控制输入框架 (input frame)

### 2. 变量管理

#### 状态和输入变量访问
```cpp
casadi::SX getStatusVariable(int frameID, const std::string &fieldName)
casadi::SX getInputVariable(int frameID, const std::string &fieldName)
```
- frameID: 时间步索引 (0 到 horizon-1)
- fieldName: 变量字段名称

#### 变量边界获取
```cpp
casadi::DM getVariableLowerBounds()
casadi::DM getVariableUpperBounds()
```

### 3. 约束条件管理

#### 添加不等式约束
```cpp
void setInequalityConstraint(
    const std::string &constraintName,
    const casadi::DM &lowerBound,
    const casadi::SX &expression,
    const casadi::DM &upperBound
)
```

#### 添加等式约束
```cpp
void addEquationConstraint(
    const std::string &constraintName, 
    const casadi::SX &leftSX,
    const casadi::SX &rightSX
)
```

```cpp
void addEquationConstraint(
    const std::string &constraintName, 
    const casadi::SX &expression
)
```

### 4. 代价函数管理
```cpp
void addCost(const casadi::SX &cost)
```
添加代价项到总代价函数中

### 5. 约束导出
```cpp
void saveConstraintsToCSV(const std::string &filename)
```
将所有约束条件导出到 CSV 文件，包含：
- 约束名称
- 下界
- 约束表达式
- 上界

## 数据结构

### Frame 结构
用于管理变量框架：
- fields: 字段列表，包含名称和大小
- fieldOffsets: 字段偏移量映射
- totalSize: 总大小

## 配置文件格式

YAML 配置文件示例：
```yaml
mpc_options:
  horizon: 10  # 预测时域长度

OCP_variables:
  status_frame:
    - name: "position"
      size: 3
    - name: "velocity"
      size: 3
  
  input_frame:
    - name: "force"
      size: 3
    - name: "torque"
      size: 3

verbose:
  variables: true  # 是否输出变量信息
```

## 使用示例

1. 创建最优控制问题实例：
```cpp
OptimalControlProblem ocp("config.yaml");
```

2. 添加约束：
```cpp
// 添加不等式约束
casadi::DM lb = -1;
casadi::DM ub = 1;
casadi::SX expr = getStatusVariable(0, "position");
ocp.setInequalityConstraint("position_bounds", lb, expr, ub);

// 添加等式约束
casadi::SX left = getStatusVariable(1, "position");
casadi::SX right = getStatusVariable(0, "position");
ocp.addEquationConstraint("position_continuity", left, right);
```

3. 添加代价函数：
```cpp
casadi::SX cost = pow(getStatusVariable(0, "position"), 2);
ocp.addCost(cost);
```

## 注意事项

1. 确保配置文件中的变量框架定义完整且正确
2. 约束添加时注意维度匹配
3. 变量访问时确保 frameID 在有效范围内
4. 所有数值约束都使用 casadi::DM 类型
5. 所有符号表达式都使用 casadi::SX 类型

## 依赖库

- CasADi: 用于符号计算和优化问题构建
- YAML-CPP: 配置文件解析


# 计算框架更新

# QP问题求解方法分析与比较

## 1. 核心方法
这里描述的是一个基于惩罚(penalty-based)的方法来求解QP问题，主要特点是：
- 将不等式约束转换为惩罚项加入目标函数中
- 保留等式约束
- 通过迭代增加惩罚参数μ来逼近原问题的解

## 2. 问题形式
原问题被转化为：
\[
\min_{\delta z} \frac{1}{2}\delta z^T P_k\delta z + c_k^T \delta z + \mu_k \cdot p(A_k^{ineq}\delta z - b_k^{ineq})
\]
\[
s.t. \quad A_k^{eq}\delta z = b_k^{eq}
\]

## 3. 与传统QP的区别
传统QP问题通常形式为：
\[
\min_x \frac{1}{2}x^TQx + c^Tx
\]
\[
s.t. \quad Ax \leq b
\]
\[
\quad \quad \quad Ex = d
\]

主要区别：
1. **约束处理方式**：
   - 传统QP：直接处理等式和不等式约束
   - 这种方法：将不等式约束转换为惩罚项，只保留等式约束

2. **求解复杂度**：
   - 传统QP：需要同时处理等式和不等式约束，可能需要使用内点法等复杂算法
   - 这种方法：简化为只有等式约束的问题，可以使用LDLT分解等更简单的方法

## 4. 与SQP的区别
Sequential Quadratic Programming (SQP)：
1. **问题性质**：
   - SQP：用于求解一般非线性优化问题，每次迭代求解一个QP子问题
   - 这种方法：专注于求解QP问题本身

2. **迭代特点**：
   - SQP：在每次迭代中更新二次近似
   - 这种方法：在迭代中只更新惩罚参数μ

## 5. 这种方法的优势
1. **计算效率**：
   - 只需要解决等式约束问题
   - 可以使用高效的LDLT分解方法

2. **实现简单**：
   - 算法结构清晰
   - 不需要复杂的不等式约束处理机制

3. **收敛性保证**：
   - 当惩罚参数μ足够大时，可以保证得到原问题的解

## 6. 潜在的局限性
1. 惩罚参数的选择可能影响收敛速度
2. 如果惩罚参数过大，可能导致数值稳定性问题
3. 需要迭代增加惩罚参数，可能需要多次求解

这种方法本质上是一种在保持问题求解简单性的同时，通过惩罚机制来处理不等式约束的折中方案。相比传统QP和SQP，它提供了一个可能更容易实现和计算的替代方案。

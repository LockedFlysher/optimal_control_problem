#pragma once

#include "optimal_control_problem/sqp_solver/AutoDifferentiator.h"
#include "optimal_control_problem/sqp_solver/CuCaQP.h"
#include <yaml-cpp/yaml.h>
#include <torch/script.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono> // 添加计时功能

class SQPOptimizationSolver {
public:
    /**
     * @brief 构造函数，初始化SQP求解器
     * @param nlp 非线性规划问题的符号表达式字典
     * @param options 求解器选项
     */
    explicit SQPOptimizationSolver(::casadi::SXDict &nlp, ::casadi::Dict &options);

    /**
     * @brief 析构函数，释放资源
     */
    ~SQPOptimizationSolver() = default;

    /**
     * @brief SQP求解方法
     * @note 如果是MPC问题初始点是通过限制第一项的lbx和ubx来完成的，和OptimalControlProblem一致
     * @param arg <string,DM>字典，用到了lbx,ubx,lbg,ubg,p(可选)
     * @retval DMDict result 包含最优解x和目标函数值f
     */
    casadi::DMDict getOptimalSolution(const casadi::DMDict &arg);

    /**
     * @brief 获取局部系统函数
     * @return 返回局部系统函数
     */
    casadi::Function getSXLocalSystemFunction() const;

    /**
     * @brief 设置是否输出详细信息
     * @param verbose 是否输出详细信息
     */
    void setVerbose(bool verbose);

private:
    // 自动微分器
    std::shared_ptr<AutoDifferentiator> objectiveFunctionAutoDifferentiatorPtr_;
    std::shared_ptr<AutoDifferentiator> constraintsAutoDifferentiator_;

    // QP求解器
    CuCaQP qpSolver_;

    // SQP参数
    int stepNum_;         // 最大迭代次数
    double alpha_;        // 步长因子
    bool verbose_;        // 是否输出详细信息

    // 优化结果
    casadi::DMDict result_;

    // 约束边界
    casadi::DM lowerBounds_;
    casadi::DM upperBounds_;

    // 目标函数
    casadi::Function objectiveFunction_;

    /**
     * @brief 局部系统函数
     * @note 依次序产生: 1. Hessian，2. gradient，3. A, 4. l，5. u
     * @param [0]reference [1]point [2]lowerBound [3]upperBound
     * @retval [0]Hessian [1]Gradient [2]A [3]l [4]u
     */
    casadi::Function localSystemFunction_;

private:
    /**
     * @brief 获取局部系统
     * @param arg 输入参数字典
     * @return 局部系统的DMVector
     */
    casadi::DMVector getLocalSystem(const casadi::DMDict &arg);
};

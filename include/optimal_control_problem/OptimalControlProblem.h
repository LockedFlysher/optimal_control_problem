#pragma once

#include <casadi/casadi.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <iostream>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "sqp_solver/SQPOptimizationSolver.h"
#include "optimal_control_problem/OCP_config/OCPConfig.h"
/*
 * 负责的内容是构建求解器，求解器的调用应该由子类完成
 * 使用符合规范的YAML文件来初始化
 * 离散的MPC构建器
 * */
class OptimalControlProblem {
private:
    enum class SolverType {
        IPOPT,
        SQP,
        CUDA_SQP,
        MIXED
    };

    YAML::Node configNode_;

    std::vector<casadi::SX> constraints_;
    std::vector<std::string> constraintNames_;
    std::vector<casadi::DM> constraintLowerBounds_;
    std::vector<casadi::DM> constraintUpperBounds_;

    casadi::SXVector costs_;
    bool setInitialGuess_{false};
    bool firstTime_{true};
    casadi::DM optimalTrajectory_;
    // OCP问题构建和求解的接口
    bool genCode_{false};
    bool loadLib_{false};
    bool verbose_{true};

    std::string packagePath_;

    SolverType selectedSolver_ = SolverType::IPOPT;  // IPOPT是默认的
    casadi::Function IPOPTSolver_;
    casadi::Function SQPSolver_;
    casadi::Function libIPOPTSolver_;
    casadi::Function libSQPSolver_;
    std::shared_ptr<SQPOptimizationSolver> OSQPSolverPtr_;

private:
    /*
     * 直接给定所有数据帧的状态变量的上界和下界
     * */

public:
    std::unique_ptr<OCPConfig> OCPConfigPtr_;
    // 添加设置求解器类型的方法
    void setSolverType(SolverType type);
    SolverType getSolverType() const;
public:
    casadi::SX getReference() const;
    /*
     * 返回最优解变量，不做计算
     * */
    casadi::DM getOptimalTrajectory();
    //    求解功能使用到的变量们
    casadi::SX reference_;
    casadi::SX totalCost_;

    /*
     * 根据配置文件决定
     * 1.是否生成c代码和动态链接库
     * 2.是否使用SQP类型的求解器
     * 3.是否从.so加载求解器
     * */
    void genSolver();
    /*
     * 把OCP的当前的状态输入到这里，reference的具体的数值发到这里，就行了
     * */
    void computeOptimalTrajectory(const casadi::DM &frame, const casadi::DM &reference);
    /*
     * 构造函数
     * */
    explicit OptimalControlProblem(const std::string &configFilePath);
    /*
     * costFunction是从这里进行
     * */
    void addCost(const casadi::SX &cost);
    /*
     * 添加不等式约束其实是一个通用的函数，是添加等式约束的基础函数
     * */
    void addInequalityConstraint(const std::string &constraintName,
                                 const casadi::DM &lowerBound,
                                 const casadi::SX &expression,
                                 const casadi::DM &upperBound);
    void addEquationConstraint(const std::string &constraintName, const casadi::SX &leftSX, const casadi::SX &rightSX);
    void addEquationConstraint(const std::string &constraintName, const casadi::SX &expression);
    /*
     * OCP创建的时候取得损失函数
     * */
    casadi::SX getCostFunction();
    casadi::DMVector getConstraintLowerBounds() const;
    casadi::DMVector getConstraintUpperBounds() const;
    /*
     * OCP创建的时候用得到这个限制
     * */
    std::vector<casadi::SX> getConstraints() const;
    /*
     * 子类必须实现这个函数添加约束
     * */
    virtual void deployConstraintsAndAddCost() = 0;
    /*
     * 检查变量的维度
     * */
    bool solverInputCheck(std::map<std::string, casadi::DM> arg) const;
};

std::ostream &operator<<(std::ostream &os, const OptimalControlProblem &ocp);

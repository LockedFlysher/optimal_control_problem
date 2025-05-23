//
// Created by lock on 25-3-11.
//
#include "optimal_control_problem/sqp_solver/SQPOptimizationSolver.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace casadi;
using namespace std::chrono;

SQPOptimizationSolver::SQPOptimizationSolver(::casadi::SXDict &nlp, ::casadi::Dict &options)
        : verbose_(false) {  // 初始化verbose_为false
    // 加载SQP参数
    stepNum_ = options.at("max_iter").as_int();
    alpha_ = options.at("alpha").as_double();
    setVerbose(options.at("verbose"));

    // 必需参数检查
    if (nlp.find("f") == nlp.end()) {
        throw std::invalid_argument("目标函数'f'未定义");
    }
    auto objectExpr = nlp["f"];

    if (nlp.find("x") == nlp.end()) {
        throw std::invalid_argument("优化变量'x'未定义");
    }
    auto variables = nlp["x"];

    // 可选参数处理
    SX constraints;
    if (nlp.find("g") != nlp.end()) {
        constraints = nlp["g"];
    } else {
        // 设置为空约束
        constraints = ::casadi::SX();
    }

    SX reference;
    if (nlp.find("p") != nlp.end()) {
        reference = nlp["p"];
    } else {
        reference = ::casadi::SX();
    }

    // 创建目标函数
    objectiveFunction_ = Function("objective", {reference, variables}, {objectExpr});

    // 创建增广变量
    auto augmentedVariables = SX::vertcat({reference, variables});

    // 初始化自动微分器
    objectiveFunctionAutoDifferentiatorPtr_ = std::make_shared<AutoDifferentiator>(augmentedVariables, objectExpr);
    SX augmentedConstraints = SX::vertcat({reference, variables, constraints});
    constraintsAutoDifferentiator_ = std::make_shared<AutoDifferentiator>(augmentedVariables, augmentedConstraints);

    // 计算Hessian和梯度
    auto hessian = objectiveFunctionAutoDifferentiatorPtr_->getHessian(augmentedVariables);
    auto gradient = objectiveFunctionAutoDifferentiatorPtr_->getGradient(augmentedVariables);

    // 线性化约束
    auto linearizedIneqConstraints = constraintsAutoDifferentiator_->getLinearization(augmentedVariables);
    auto numOfConstraints = linearizedIneqConstraints[1].size1();

    // 创建约束边界符号
    auto l = SX::sym("l", numOfConstraints);
    auto u = SX::sym("u", numOfConstraints);

    // 计算线性化后的约束边界
    auto l_linearized = l + linearizedIneqConstraints[1];
    auto u_linearized = u + linearizedIneqConstraints[1];

    // 创建局部系统函数
    localSystemFunction_ = Function("localSystemFunction",
                                    {reference, variables, l, u},
                                    {hessian, gradient, linearizedIneqConstraints[0],
                                     l_linearized, u_linearized});

    // 配置QP求解器
    qpSolver_.setDimension(augmentedVariables.size1(), augmentedConstraints.size1());
    qpSolver_.setVerbosity(false);
    qpSolver_.setWarmStart(true);
    qpSolver_.setAbsoluteTolerance(1e-3);
    qpSolver_.setRelativeTolerance(1e-3);
    qpSolver_.setMaxIteration(10000);

    // 初始化结果字典
    result_ = {
            {"x", DM::zeros(variables.size1())},
            {"f", DM::zeros(1)}
    };
}

/**
* @brief 获取局部系统
* @note 输入的arg保持和原来一致
* @param arg 输入参数字典
* @retval 局部系统的DMVector
*/
DMVector SQPOptimizationSolver::getLocalSystem(const DMDict &arg) {
    // 获取边界条件
    DM lbx = arg.at("lbx");
    DM ubx = arg.at("ubx");
    DM lbg = arg.at("lbg");
    DM ubg = arg.at("ubg");

    // 处理参考变量p
    DM p;
    if (arg.find("p") != arg.end()) {
        p = arg.at("p");
    } else {
        p = DM::zeros(0, 1);  // 创建空矩阵作为默认值
    }

    // 调用局部系统函数
    DMVector localSystem = localSystemFunction_(
            DMVector{p, result_.at("x"), DM::vertcat({p, lbx, lbg}), DM::vertcat({p, ubx, ubg})});

    return localSystem;
}

/**
* @brief SQP求解方法
* @param arg 输入参数字典
* @retval 包含最优解和目标函数值的字典
*/
DMDict SQPOptimizationSolver::getOptimalSolution(const DMDict &arg) {
    if (verbose_) {
        std::cout << "=== SQP优化开始 ===" << std::endl;
        std::cout << "最大迭代次数: " << stepNum_ << ", 步长因子: " << alpha_ << std::endl;
    }

    auto totalStartTime = high_resolution_clock::now();
    double totalQpSolveTime = 0.0;
    double totalLocalSystemTime = 0.0;

    for (int i = 0; i < stepNum_; ++i) {
        if (verbose_) {
            std::cout << "\n迭代 " << i+1 << "/" << stepNum_ << std::endl;
        }

        // 计算局部系统时间
        auto localSystemStartTime = high_resolution_clock::now();
        DMVector localSystem = getLocalSystem(arg);
        auto localSystemEndTime = high_resolution_clock::now();
        double localSystemTime = duration_cast<microseconds>(localSystemEndTime - localSystemStartTime).count() / 1000.0;
        totalLocalSystemTime += localSystemTime;

        if (verbose_) {
            std::cout << "  局部系统计算时间: " << localSystemTime << " ms" << std::endl;
        }

        // 设置并求解QP问题
        auto qpStartTime = high_resolution_clock::now();
        qpSolver_.setSystem(localSystem);
        qpSolver_.initSolver();
        qpSolver_.solve();
        auto qpEndTime = high_resolution_clock::now();
        double qpSolveTime = duration_cast<microseconds>(qpEndTime - qpStartTime).count() / 1000.0;
        totalQpSolveTime += qpSolveTime;

        if (verbose_) {
            std::cout << "  QP求解时间: " << qpSolveTime << " ms" << std::endl;
        }

        // 获取解并更新
        DM solution = qpSolver_.getSolutionAsDM();
        DM oldRes = result_.at("x");

        // 根据是否存在参考变量p来更新结果
        if (arg.find("p") == arg.end() || arg.at("p").size1() == 0) {
            result_.at("x") += alpha_ * solution;
        } else {
            // 从完整解中提取变量部分（排除参考变量p）
            int pSize = arg.at("p").size1();
            result_.at("x") += alpha_ * solution(Slice(pSize, solution.size1()));
        }

        // 计算并更新目标函数值
        DM p = (arg.find("p") != arg.end()) ? arg.at("p") : DM::zeros(0, 1);
        result_.at("f") = objectiveFunction_(DMVector{p, result_.at("x")});

        if (verbose_) {
            std::cout << "  当前解: " << result_.at("x") << std::endl;
            std::cout << "  当前目标函数值: " << result_.at("f") << std::endl;

            // 计算并显示解的变化量
            DM delta = result_.at("x") - oldRes;
            double normDelta = norm_2(delta).scalar();
            std::cout << "  解的变化量: " << normDelta << std::endl;

            // 如果变化很小，可以提前终止
            if (normDelta < 1e-6) {
                std::cout << "  解收敛，提前终止迭代" << std::endl;
                break;
            }
        }
    }

    auto totalEndTime = high_resolution_clock::now();
    double totalTime = duration_cast<microseconds>(totalEndTime - totalStartTime).count() / 1000.0;

    if (verbose_) {
        std::cout << "\n=== SQP优化完成 ===" << std::endl;
        std::cout << "总耗时: " << totalTime << " ms" << std::endl;
        std::cout << "局部系统计算总时间: " << totalLocalSystemTime << " ms ("
                  << std::fixed << std::setprecision(1) << (totalLocalSystemTime / totalTime * 100) << "%)" << std::endl;
        std::cout << "QP求解总时间: " << totalQpSolveTime << " ms ("
                  << std::fixed << std::setprecision(1) << (totalQpSolveTime / totalTime * 100) << "%)" << std::endl;
        std::cout << "最终结果:" << std::endl;
        std::cout << "  x = " << result_.at("x") << std::endl;
        std::cout << "  f = " << result_.at("f") << std::endl;
    }

    return result_;
}

/**
* @brief 获取局部系统函数
* @return 局部系统函数
*/
casadi::Function SQPOptimizationSolver::getSXLocalSystemFunction() const {
    return localSystemFunction_;
}

/**
* @brief 设置是否输出详细信息
* @param verbose 是否输出详细信息
*/
void SQPOptimizationSolver::setVerbose(bool verbose) {
    verbose_ = verbose;
    // 同时设置QP求解器的详细程度
    qpSolver_.setVerbosity(verbose);
}

//
// Created by lock on 25-3-11.
//
#include "sqp_solver/SQPOptimizationSolver.h"
#include <iostream>
/*
 * 最小单步求解示例
 * */
using namespace casadi;

SQPOptimizationSolver::SQPOptimizationSolver(::casadi::SXDict nlp, YAML::Node configNode) {
    //加载SQP和ADMM的参数

    stepNum_ = configNode["solver_settings"]["ADMM_step"].as<int>();
    alpha_ = configNode["solver_settings"]["SQP_step"].as<double>();

    // 必需参数
    if (nlp.find("f") == nlp.end()) {
        throw std::invalid_argument("目标函数'f'未定义");
    }
    auto objectExpr = nlp["f"];
    if (nlp.find("x") == nlp.end()) {
        throw std::invalid_argument("优化变量'x'未定义");
    }
    auto variables = nlp["x"];
    // 可选参数
    SX constraints;
    if (nlp.find("g") != nlp.end()) {
        constraints = nlp["g"];
    } else {
        // 设置默认值或标记为未定义
        constraints = ::casadi::SX();  // 假设空SX表示未定义
    }

    SX reference;
    if (nlp.find("p") != nlp.end()) {
        reference = nlp["p"];
    } else {
        reference = ::casadi::SX();  // 假设空SX表示未定义
    }
    static int referenceDim = reference.size1();
    auto augmentedVariables = SX::vertcat({reference, variables});
    objectiveFunctionAutoDifferentiatorPtr_ = std::make_shared<AutoDifferentiator>(augmentedVariables, objectExpr);
    SX augmentedConstraints = SX::vertcat({reference, variables, constraints});
    constraintsAutoDifferentiator_ = std::make_shared<AutoDifferentiator>(augmentedVariables, augmentedConstraints);
    auto hessian = objectiveFunctionAutoDifferentiatorPtr_->getHessian(augmentedVariables);
    auto gradient = objectiveFunctionAutoDifferentiatorPtr_->getGradient(augmentedVariables);
    auto linearizedIneqConstraints = constraintsAutoDifferentiator_->getLinearization(augmentedVariables);
    auto numOfConstraints = linearizedIneqConstraints[1].size1();
    auto l = SX::sym("l", numOfConstraints);
    auto u = SX::sym("u", numOfConstraints);
    auto l_linearized = l + linearizedIneqConstraints[1];
    auto u_linearized = u + linearizedIneqConstraints[1];
    localSystemFunction_ = Function("localSystemFunction", {reference, variables, l, u},
                                    {hessian, gradient, linearizedIneqConstraints[0], l_linearized, u_linearized});
    qpSolver_.setDimension(augmentedVariables.size1(), augmentedConstraints.size1());
    qpSolver_.setVerbosity(true);
    qpSolver_.setWarmStart(true);

    qpSolver_.setAbsoluteTolerance(1e-3);
    qpSolver_.setRelativeTolerance(1e-3);
    qpSolver_.setMaxIteration(50);
    result_ = {
            {"x", DM::zeros(variables.size1())},
            {"f", DM::zeros(1)}
    };
}

/**
* @brief 输入的arg保持和原来一致
* @note None
* @param None
* @retval None
*/

DMVector SQPOptimizationSolver::getLocalSystem(const DMDict &arg) {
    DM lbx = arg.at("lbx");
    DM ubx = arg.at("ubx");
    DM lbg = arg.at("lbg");
    DM ubg = arg.at("ubg");
    DM p = DM::zeros();
    if (arg.find("p") == arg.end()) {
        DM p = DM::zeros();
    } else {
        p = arg.at("p");
    }

    DMVector localSystem = localSystemFunction_(
            DMVector{p, result_.at("x"), DM::vertcat({p, lbx, lbg}), DM::vertcat({p, ubx, ubg})});
    return localSystem;
}

DMDict SQPOptimizationSolver::getOptimalSolution(const DMDict &arg) {
//    auto argCopy = arg;

    for (int i = 0; i < stepNum_; ++i) {
        DMVector localSystem = getLocalSystem(arg);
        qpSolver_.setSystem(localSystem);
        qpSolver_.initSolver();
        qpSolver_.solve();
        DM solution = qpSolver_.getSolutionAsDM();
        DM oldRes = result_.at("x");
        if (arg.find("p") == arg.end()) {
            result_.at("x") += alpha_ * solution;
        } else {
            if (arg.at("p").size1() == 0) {
                result_.at("x") += alpha_ * solution;
            } else {
                result_.at("x") += alpha_ * solution(Slice(arg.at("p").size1(), solution.size1()));
            }
        }
//        std::cout << "解更新: " << oldRes << " -> " << result_.at("x") << std::endl;
    }
//    std::cout << "最终结果: " << std::endl;
//    std::cout << "  x = " << result_.at("x") << std::endl;
//    std::cout << "  f = " << result_.at("f") << std::endl;
    return result_;
}

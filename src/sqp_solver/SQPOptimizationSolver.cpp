//
// Created by lock on 25-3-11.
//
#include "sqp_solver/SQPOptimizationSolver.h"
#include <iostream>

/*
 * 最小单步求解示例
 * */
using namespace casadi;

DMVector SQPOptimizationSolver::getLocalSystem(const DMDict &arg) {

    DM lbx = arg.at("lbx");
    DM ubx = arg.at("ubx");
    DM lbg = arg.at("lbg");
    DM ubg = arg.at("ubg");
    DM p = arg.at("p");

    DMVector localSystem = localSystemFunction_(DMVector{result_.at("res"),
                                                         DM::vertcat({lbx, lbg}),
                                                         DM::vertcat({ubx, ubg}),
                                                         p});
    return localSystem;
}

SQPOptimizationSolver::SQPOptimizationSolver(::casadi::SXDict nlp) {
    std::cout << "\n==== SQP优化求解器初始化 ====" << std::endl;

    // 必需参数
    if (nlp.find("f") == nlp.end()) {
        throw std::invalid_argument("目标函数'f'未定义");
    }
    objectExpr_ = nlp["f"];
    std::cout << "目标函数已设置: " << objectExpr_ << std::endl;

    if (nlp.find("x") == nlp.end()) {
        throw std::invalid_argument("优化变量'x'未定义");
    }
    variables_ = nlp["x"];
    std::cout << "优化变量已设置，维度: " << variables_.size1() << "x" << variables_.size2() << std::endl;

    // 可选参数
    if (nlp.find("g") != nlp.end()) {
        constraints_ = nlp["g"];
        std::cout << "约束条件已设置，维度: " << constraints_.size1() << "x" << constraints_.size2() << std::endl;
    } else {
        // 设置默认值或标记为未定义
        constraints_ = ::casadi::SX();  // 假设空SX表示未定义
        std::cout << "未设置约束条件，使用默认空约束" << std::endl;
    }

    if (nlp.find("p") != nlp.end()) {
        reference_ = nlp["p"];
        std::cout << "参考值已设置，维度: " << reference_.size1() << "x" << reference_.size2() << std::endl;
    } else {
        // 设置默认值或标记为未定义
        reference_ = ::casadi::SX();  // 假设空SX表示未定义
        std::cout << "未设置参考值，使用默认空参考" << std::endl;
    }

    std::cout << "创建目标函数自动微分器..." << std::endl;
    objectiveFunctionAutoDifferentiatorPtr_ = std::make_shared<AutoDifferentiator>(variables_, objectExpr_);

    //    就是线性化的不等式约束要额外处理一下，我们在运行的时候一定写的是:
    std::cout << "创建增广约束..." << std::endl;
    SX augmentedConstraints_ = SX::zeros(variables_.size1()+constraints_.size1());
    for (int i = 0; i < augmentedConstraints_.size1(); ++i) {
        if (i<variables_.size1()){
            augmentedConstraints_(i) = variables_(i);
        } else{
            augmentedConstraints_(i) = constraints_(i-variables_.size1());
        }
    }
    std::cout << "增广约束维度: " << augmentedConstraints_.size1() << "x" << augmentedConstraints_.size2() << std::endl;

    std::cout << "创建约束条件自动微分器..." << std::endl;
    constraintsAutoDifferentiator_ = std::make_shared<AutoDifferentiator>(variables_, augmentedConstraints_);

    //    Trajectory的跟踪是需要外部生成一系列的参数来完成
    std::cout << "计算Hessian矩阵..." << std::endl;
    auto hessian = objectiveFunctionAutoDifferentiatorPtr_->getHessian(variables_);
    std::cout << "计算梯度向量..." << std::endl;
    auto gradient = objectiveFunctionAutoDifferentiatorPtr_->getGradient(variables_);

    //    在不等式内，AX产生不了常数项，l，b是常数项，需要减去AX在X_k点上的值才对，总之先产生P、Q、A
    std::cout << "线性化不等式约束..." << std::endl;
    auto linearizedIneqConstraints = constraintsAutoDifferentiator_->getLinearization(variables_);
    auto numOfConstraints = linearizedIneqConstraints[1].size1();
    std::cout << "约束数量: " << numOfConstraints << std::endl;

    //    使用轨迹跟踪作为reference给定的依据，更新系统矩阵的时候在Hessian和Gradient这两块地方会出现ref
    std::cout << "创建局部系统函数..." << std::endl;
    auto reference = SX::sym("ref", variables_.size1());
    auto l = SX::sym("l", numOfConstraints);
    auto u = SX::sym("u", numOfConstraints);
    auto l_linearized = l - linearizedIneqConstraints[1];
    auto u_linearized = u - linearizedIneqConstraints[1];
    localSystemFunction_ = Function("localSystemFunction", {variables_, l, u, reference},
                                    {hessian, gradient, linearizedIneqConstraints[0], l_linearized, u_linearized});
    std::cout << "局部系统函数: " << std::endl << localSystemFunction_ << std::endl;
    qpSolver_.setDimension(variables_.size1(), augmentedConstraints_.size1());
    std::cout << "QP求解器维度设置为: " << variables_.size1() << "x" << augmentedConstraints_.size1() << std::endl;

    qpSolver_.setVerbosity(true);
    qpSolver_.setWarmStart(true);
    qpSolver_.setAbsoluteTolerance(1e-4);
    qpSolver_.setRelativeTolerance(1e-4);
    qpSolver_.setMaxIteration(5000);

    //    初始化求解结果
    std::cout << "初始化求解结果..." << std::endl;
    result_ = {
            {"x", DM::zeros(variables_.size1())},
            {"f",   DM::zeros(1)}
    };
    std::cout << "初始结果: res=" << result_.at("res") << ", f=" << result_.at("f") << std::endl;
    std::cout << "SQP优化求解器初始化完成!" << std::endl;
}

DMDict SQPOptimizationSolver::getOptimalSolution(const DMDict &arg) {
    std::cout << "\n==== 开始求解最优解 ====" << std::endl;
    std::cout << "计划迭代步数: " << stepNum_ << std::endl;

    for (int i = 0; i < stepNum_; ++i) {
        std::cout << "\n--- 迭代步骤 " << i+1 << "/" << stepNum_ << " ---" << std::endl;

        std::cout << "获取局部系统..." << std::endl;
        DMVector localSystem = getLocalSystem(arg);
        std::cout << "设置QP求解器参数..." << std::endl;
        qpSolver_.setHessianMatrix(localSystem[0]);
        qpSolver_.setGradient(localSystem[1]);
        qpSolver_.setLinearConstraintsMatrix(localSystem[2]);
        qpSolver_.setLowerBound(localSystem[3]);
        qpSolver_.setUpperBound(localSystem[4]);

        std::cout << "初始化并求解QP问题..." << std::endl;
        qpSolver_.initSolver();
        DM solution = qpSolver_.getSolutionAsDM();
        std::cout << "QP求解结果: " << solution << std::endl;

        std::cout << "更新当前解..." << std::endl;
        DM oldRes = result_.at("x");
        result_.at("x") += alpha_ * solution;
        std::cout << "解更新: " << oldRes << " -> " << result_.at("x") << std::endl;

    }

    std::cout << "\n==== 优化求解完成 ====" << std::endl;
    std::cout << "最终结果: " << std::endl;
    std::cout << "  res = " << result_.at("res") << std::endl;
    std::cout << "  f = " << result_.at("f") << std::endl;

    return result_;
}

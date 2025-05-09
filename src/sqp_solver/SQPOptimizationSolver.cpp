//
// Created by lock on 25-3-11.
//
#include "optimal_control_problem/sqp_solver/SQPOptimizationSolver.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

using namespace casadi;
using namespace std::chrono;

SQPOptimizationSolver::SQPOptimizationSolver(::casadi::SXDict &nlp, ::casadi::Dict &options, int numSolvers)
        : numSolvers_(numSolvers > 0 ? numSolvers : 1), verbose_(false) {
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

    // 初始化多个求解器实例
    qpSolvers_.resize(numSolvers_);
    solvers_.resize(numSolvers_);
    results_.resize(numSolvers_);
    resultsTensor_.resize(numSolvers_);

    for (int i = 0; i < numSolvers_; ++i) {
        // 配置QP求解器
        qpSolvers_[i].setDimension(augmentedVariables.size1(), augmentedConstraints.size1());
        qpSolvers_[i].setVerbosity(false);
        qpSolvers_[i].setWarmStart(true);
        qpSolvers_[i].setAbsoluteTolerance(1e-3);
        qpSolvers_[i].setRelativeTolerance(1e-3);
        qpSolvers_[i].setMaxIteration(10000);

        // 初始化结果字典
        results_[i] = {
                {"x", DM::zeros(variables.size1())},
                {"f", DM::zeros(1)}
        };

        // 初始化Tensor结果字典
        resultsTensor_[i] = {
                {"x", torch::zeros({static_cast<int64_t>(variables.size1()), 1})},
                {"f", torch::zeros({1, 1})}
        };
    }

    // 默认启用CUDA
    setBackend(true);
}

void SQPOptimizationSolver::loadFromFile() {
    // cusadi相关参数
    // 加载localsystemfunction的路径
    functionFilePath_ = ament_index_cpp::get_package_share_directory("optimal_control_problem") +
                        "/code_gen/localSystemFunction.casadi";
    N_ENVS = 1;
    fn = casadi::Function::load(functionFilePath_);
    std::cout << "Loaded CasADi function: " << fn.name() << "\n";

    // 为每个求解器实例加载函数
    for (int i = 0; i < numSolvers_; ++i) {
        solvers_[i] = std::make_unique<CasadiGpuEvaluator>(fn);
    }
}

casadi::Function SQPOptimizationSolver::getSXLocalSystemFunction() const {
    return localSystemFunction_;
}



std::vector<torch::Tensor> SQPOptimizationSolver::getLocalSystem(const DMDict &arg, int solverIndex) {
    // 获取边界条件
    DM lbx = arg.at("lbx");
    DM ubx = arg.at("ubx");
    DM lbg = arg.at("lbg");
    DM ubg = arg.at("ubg");
    DM p_dm = (arg.find("p") != arg.end()) ? arg.at("p") : DM::zeros(0, 1);
    DM x_dm = results_[solverIndex].at("x");
    DM l_dm = DM::vertcat({p_dm, lbx, lbg});
    DM u_dm = DM::vertcat({p_dm, ubx, ubg});

    // 2. 将DM转换为Tensor，并确保数据类型为kFloat64
    auto p_tensor = dmToTensor(p_dm).to(torch::kFloat64).to(torch::kCUDA);
    auto x_tensor = dmToTensor(x_dm).to(torch::kFloat64).to(torch::kCUDA);

    auto l_tensor = dmToTensor(l_dm).to(torch::kFloat64).to(torch::kCUDA);
    auto u_tensor = dmToTensor(u_dm).to(torch::kFloat64).to(torch::kCUDA);
    // 确保输入Tensor形状正确（二维列向量）
    p_tensor = p_tensor.view({p_dm.size1(), 1});
    x_tensor = x_tensor.view({x_dm.size1(), 1});
    l_tensor = l_tensor.view({l_dm.size1(), 1});
    u_tensor = u_tensor.view({u_dm.size1(), 1});

    std::vector<torch::Tensor> inputs = {p_tensor, x_tensor, l_tensor, u_tensor};
    solvers_[solverIndex]->compute(inputs);
    std::vector<torch::Tensor> outTensorVector;
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(0).cpu());
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(1).cpu());
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(2).cpu());
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(3).cpu());
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(4).cpu());
    return outTensorVector;
}

DMDict SQPOptimizationSolver::getOptimalSolution(const DMDict &arg) {
    // 使用第一个求解器处理单个输入
    if (verbose_) {
        std::cout << "=== SQP优化开始 (单个实例) ===" << std::endl;
        std::cout << "最大迭代次数: " << stepNum_ << ", 步长因子: " << alpha_ << std::endl;
    }
    auto totalStartTime = high_resolution_clock::now();
    double totalQpSolveTime = 0.0;
    double totalLocalSystemTime = 0.0;
    double totalConvertingTime = 0.0;

    for (int i = 0; i < stepNum_; ++i) {
        if (verbose_) {
            std::cout << "\n迭代 " << i + 1 << "/" << stepNum_ << std::endl;
        }
        // 计算局部系统时间
        auto localSystemStartTime = high_resolution_clock::now();
        std::vector<torch::Tensor> localSystem = getLocalSystem(arg, 0);
        auto localSystemEndTime = high_resolution_clock::now();
        double localSystemTime =
                duration_cast<microseconds>(localSystemEndTime - localSystemStartTime).count() / 1000.0;
        totalLocalSystemTime += localSystemTime;
        if (verbose_) {
            std::cout << "  局部系统计算时间: " << localSystemTime << " ms" << std::endl;
        }

        // 计算数据类型转换时间
        auto convertingStartTime = high_resolution_clock::now();
        qpSolvers_[0].setSystem(localSystem);
        auto convertingEndTime = high_resolution_clock::now();
        double convertingTime = duration_cast<microseconds>(convertingEndTime - convertingStartTime).count() / 1000.0;
        totalConvertingTime += convertingTime;
        if (verbose_) {
            std::cout << "  系统数据类型转换计算时间: " << convertingTime << " ms" << std::endl;
        }

        // 设置并求解QP问题
        auto qpStartTime = high_resolution_clock::now();
        qpSolvers_[0].initSolver();
        qpSolvers_[0].solve();
        auto qpEndTime = high_resolution_clock::now();
        double qpSolveTime = duration_cast<microseconds>(qpEndTime - qpStartTime).count() / 1000.0;
        totalQpSolveTime += qpSolveTime;
        if (verbose_) {
            std::cout << "  QP求解时间: " << qpSolveTime << " ms" << std::endl;
        }
        // 获取解并更新
        DM solution = qpSolvers_[0].getSolutionAsDM();
        DM oldRes = results_[0].at("x");
        // 根据是否存在参考变量p来更新结果
        if (arg.find("p") == arg.end() || arg.at("p").size1() == 0) {
            results_[0].at("x") += alpha_ * solution;
        } else {
            // 从完整解中提取变量部分（排除参考变量p）
            int pSize = arg.at("p").size1();
            results_[0].at("x") += alpha_ * solution(Slice(pSize, solution.size1()));
        }

        // 计算并更新目标函数值
        DM p = (arg.find("p") != arg.end()) ? arg.at("p") : DM::zeros(0, 1);
        results_[0].at("f") = objectiveFunction_(DMVector{p, results_[0].at("x")});

        if (verbose_) {
            std::cout << "  当前解: " << results_[0].at("x") << std::endl;
            std::cout << "  当前目标函数值: " << results_[0].at("f") << std::endl;

            // 计算并显示解的变化量
            DM delta = results_[0].at("x") - oldRes;
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
                  << std::fixed << std::setprecision(1) << (totalLocalSystemTime / totalTime * 100) << "%)"
                  << std::endl;
        std::cout << "数据类型转换时间: " << totalConvertingTime << " ms ("
                  << std::fixed << std::setprecision(1) << (totalConvertingTime / totalTime * 100) << "%)"
                  << std::endl;
        std::cout << "QP求解总时间: " << totalQpSolveTime << " ms ("
                  << std::fixed << std::setprecision(1) << (totalQpSolveTime / totalTime * 100) << "%)" << std::endl;
        std::cout << "最终结果:" << std::endl;
        std::cout << "  x = " << results_[0].at("x") << std::endl;
        std::cout << "  f = " << results_[0].at("f") << std::endl;
    }

    return results_[0];
}

std::vector<DMDict> SQPOptimizationSolver::getOptimalSolution(const std::vector<DMDict> &args) {
    std::vector<DMDict> outputResults;
    int numInputs = static_cast<int>(args.size());
    if (numInputs > numSolvers_) {
        std::cerr << "警告: 输入的数量 (" << numInputs << ") 超过求解器数量 (" << numSolvers_
                  << ")，将仅处理前 " << numSolvers_ << " 个输入。" << std::endl;
        numInputs = numSolvers_;
    }

    if (verbose_) {
        std::cout << "=== SQP优化开始 (多个实例，处理 " << numInputs << " 个输入) ===" << std::endl;
        std::cout << "最大迭代次数: " << stepNum_ << ", 步长因子: " << alpha_ << std::endl;
    }

    outputResults.resize(numInputs);

    for (int solverIndex = 0; solverIndex < numInputs; ++solverIndex) {
        const auto& arg = args[solverIndex];
        if (verbose_) {
            std::cout << "\n处理实例 " << solverIndex + 1 << "/" << numInputs << std::endl;
        }

        auto totalStartTime = high_resolution_clock::now();
        double totalQpSolveTime = 0.0;
        double totalLocalSystemTime = 0.0;
        double totalConvertingTime = 0.0;

        for (int i = 0; i < stepNum_; ++i) {
            if (verbose_) {
                std::cout << "  迭代 " << i + 1 << "/" << stepNum_ << std::endl;
            }
            // 计算局部系统时间
            auto localSystemStartTime = high_resolution_clock::now();
            std::vector<torch::Tensor> localSystem = getLocalSystem(arg, solverIndex);
            auto localSystemEndTime = high_resolution_clock::now();
            double localSystemTime =
                    duration_cast<microseconds>(localSystemEndTime - localSystemStartTime).count() / 1000.0;
            totalLocalSystemTime += localSystemTime;
            if (verbose_) {
                std::cout << "    局部系统计算时间: " << localSystemTime << " ms" << std::endl;
            }

            // 计算数据类型转换时间
            auto convertingStartTime = high_resolution_clock::now();
            qpSolvers_[solverIndex].setSystem(localSystem);
            auto convertingEndTime = high_resolution_clock::now();
            double convertingTime = duration_cast<microseconds>(convertingEndTime - convertingStartTime).count() / 1000.0;
            totalConvertingTime += convertingTime;
            if (verbose_) {
                std::cout << "    系统数据类型转换计算时间: " << convertingTime << " ms" << std::endl;
            }

            // 设置并求解QP问题
            auto qpStartTime = high_resolution_clock::now();
            qpSolvers_[solverIndex].initSolver();
            qpSolvers_[solverIndex].solve();
            auto qpEndTime = high_resolution_clock::now();
            double qpSolveTime = duration_cast<microseconds>(qpEndTime - qpStartTime).count() / 1000.0;
            totalQpSolveTime += qpSolveTime;
            if (verbose_) {
                std::cout << "    QP求解时间: " << qpSolveTime << " ms" << std::endl;
            }
            // 获取解并更新
            DM solution = qpSolvers_[solverIndex].getSolutionAsDM();
            DM oldRes = results_[solverIndex].at("x");
            // 根据是否存在参考变量p来更新结果
            if (arg.find("p") == arg.end() || arg.at("p").size1() == 0) {
                results_[solverIndex].at("x") += alpha_ * solution;
            } else {
                // 从完整解中提取变量部分（排除参考变量p）
                int pSize = arg.at("p").size1();
                results_[solverIndex].at("x") += alpha_ * solution(Slice(pSize, solution.size1()));
            }

            // 计算并更新目标函数值
            DM p = (arg.find("p") != arg.end()) ? arg.at("p") : DM::zeros(0, 1);
            results_[solverIndex].at("f") = objectiveFunction_(DMVector{p, results_[solverIndex].at("x")});

            if (verbose_) {
                std::cout << "    当前解: " << results_[solverIndex].at("x") << std::endl;
                std::cout << "    当前目标函数值: " << results_[solverIndex].at("f") << std::endl;

                // 计算并显示解的变化量
                DM delta = results_[solverIndex].at("x") - oldRes;
                double normDelta = norm_2(delta).scalar();
                std::cout << "    解的变化量: " << normDelta << std::endl;

                // 如果变化很小，可以提前终止
                if (normDelta < 1e-6) {
                    std::cout << "    解收敛，提前终止迭代" << std::endl;
                    break;
                }
            }
        }

        auto totalEndTime = high_resolution_clock::now();
        double totalTime = duration_cast<microseconds>(totalEndTime - totalStartTime).count() / 1000.0;

        if (verbose_) {
            std::cout << "\n  === 实例 " << solverIndex + 1 << " 优化完成 ===" << std::endl;
            std::cout << "  总耗时: " << totalTime << " ms" << std::endl;
            std::cout << "  局部系统计算总时间: " << totalLocalSystemTime << " ms ("
                      << std::fixed << std::setprecision(1) << (totalLocalSystemTime / totalTime * 100) << "%)"
                      << std::endl;
            std::cout << "  数据类型转换时间: " << totalConvertingTime << " ms ("
                      << std::fixed << std::setprecision(1) << (totalConvertingTime / totalTime * 100) << "%)"
                      << std::endl;
            std::cout << "  QP求解总时间: " << totalQpSolveTime << " ms ("
                      << std::fixed << std::setprecision(1) << (totalQpSolveTime / totalTime * 100) << "%)" << std::endl;
            std::cout << "  最终结果:" << std::endl;
            std::cout << "    x = " << results_[solverIndex].at("x") << std::endl;
            std::cout << "    f = " << results_[solverIndex].at("f") << std::endl;
        }

        // 将结果存入输出向量
        outputResults[solverIndex] = results_[solverIndex];
    }

    if (verbose_) {
        std::cout << "\n=== 所有实例优化完成 ===" << std::endl;
    }

    return outputResults;
}

void SQPOptimizationSolver::setVerbose(bool verbose) {
    verbose_ = verbose;
}




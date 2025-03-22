//
// Created by lock on 2024/10/18.
//

#include "optimal_control_problem/OptimalControlProblem.h"
#include <iostream>

//通过解析yaml文件，初始化dt、horizon,创建系统状态和输入变量
OptimalControlProblem::OptimalControlProblem(const std::string &configFilePath) {
    // 获取包路径
    try {
        packagePath_ = ament_index_cpp::get_package_share_directory("optimal_control_problem");
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to get package path: " + std::string(e.what()));
    }
    OCPConfigPtr_ = std::make_unique<OCPConfig>();

    configNode_ = YAML::LoadFile(
            ament_index_cpp::get_package_share_directory("optimal_control_problem") + "/config/OCP_config.yaml");
    //    初始化reference
    reference_ = ::casadi::SX::sym("ref", OCPConfigPtr_->getFrameSize());
    genCode_ = configNode_["solver_settings"]["gen_code"].as<bool>();
    loadLib_ = configNode_["solver_settings"]["load_lib"].as<bool>();

    if (configNode_["solver_settings"]["solve_method"].as<std::string>() == "IPOPT") {
        setSolverType(SolverType::IPOPT);
    }
    if (configNode_["solver_settings"]["solve_method"].as<std::string>() == "MIXED") {
        setSolverType(SolverType::MIXED);
    }
    if (configNode_["solver_settings"]["solve_method"].as<std::string>() == "SQP") {
        setSolverType(SolverType::SQP);
    }
    if (configNode_["solver_settings"]["solve_method"].as<std::string>() == "CUDA_SQP") {
        setSolverType(SolverType::CUDA_SQP);
    }
    if (configNode_["solver_settings"]["verbose"].as<bool>()) {
        std::cout << "输出c代码且编译动态链接库：" << genCode_ << std::endl;
        std::cout << "使用动态链接库对求解器进行加载：" << genCode_ << std::endl;
    }
}

void OptimalControlProblem::addCost(const casadi::SX &cost) {
    costs_.push_back(cost);
}

void OptimalControlProblem::addInequalityConstraint(const std::string &constraintName,
                                                    const casadi::DM &lowerBound,
                                                    const casadi::SX &expression,
                                                    const casadi::DM &upperBound) {
    if (lowerBound.size1() != expression.size1() || expression.size1() != upperBound.size1()) {
        throw std::invalid_argument("SX used for inequality constraints has different dimensions!");
    }
    if (lowerBound.size2() != 1 || expression.size2() != 1 || upperBound.size2() != 1) {
        throw std::invalid_argument("SX used for inequality constraints has invalid column number!");
    }

    constraints_.push_back(expression);
    for (int i = 0; i < expression.size1(); ++i) {
        constraintNames_.push_back(constraintName);
    }
    constraintLowerBounds_.push_back(lowerBound);
    constraintUpperBounds_.push_back(upperBound);
}

void OptimalControlProblem::addEquationConstraint(const std::string &constraintName,
                                                  const casadi::SX &leftSX,
                                                  const casadi::SX &rightSX) {
    if (leftSX.size1() != rightSX.size1()) {
        throw std::invalid_argument("SX used for constraints has different dimension!");
    }
    if (leftSX.size2() != 1 || rightSX.size2() != 1) {
        throw std::invalid_argument("SX used for constraints has invalid column number!");
    }
    constraints_.push_back(leftSX - rightSX);
    for (int i = 0; i < leftSX.size1(); ++i) {
        constraintNames_.push_back(constraintName);
    }
    constraintLowerBounds_.push_back(casadi::DM::zeros(leftSX.size1()));
    constraintUpperBounds_.push_back(casadi::DM::zeros(leftSX.size1()));
}

void OptimalControlProblem::addEquationConstraint(const std::string &constraintName, const casadi::SX &expression) {
    if (expression.size2() != 1) {
        throw std::invalid_argument("SX used for constraints has invalid column number!");
    }
    addEquationConstraint(constraintName, expression, casadi::SX::zeros(expression.size1()));
}

casadi::SX OptimalControlProblem::getCostFunction() {
    totalCost_ = casadi::SX::zeros(1);
    for (auto const &cost: costs_) {
        totalCost_ += cost;
    }
    return totalCost_;
}

void OptimalControlProblem::setSolverType(SolverType type) {
    selectedSolver_ = type;
}

OptimalControlProblem::SolverType OptimalControlProblem::getSolverType() const {
    return selectedSolver_;
}

/*
 * note : 必需在应用约束和添加损失以后使用！！！！！！！
 * */
// todo : 需要在这里把问题SQP化，约束线性化，A\B矩阵都要作为参数放到ADMM求解器作为参数，问题就来了，A\B肯定是稀疏的才比较好求啊，
void OptimalControlProblem::genSolver() {
    try {
        // 构建NLP问题
        ::casadi::SX vars = OCPConfigPtr_->getVariables();
        if (vars.is_empty()) {
            throw std::runtime_error("Status or input variables are empty");
        }
        ::casadi::SX constraints = ::casadi::SX::vertcat(this->getConstraints());
        if (constraints.is_empty()) {
            throw std::runtime_error("Constraints are empty");
        }
        ::casadi::SXDict nlp = {
                {"x", vars},
                {"f", getCostFunction()},
                {"g", constraints},
                {"p", reference_}
        };
        // 创建求解器
        try {
            // 设置求解器选项
            ::casadi::Dict solver_opts;
            solver_opts["print_in"] = 0;
            solver_opts["print_out"] = 0;
            solver_opts["print_time"] = 0;
            switch (selectedSolver_) {
                case SolverType::IPOPT: {
                    IPOPTSolver_ = ::casadi::nlpsol("solver", "ipopt", nlp, solver_opts);
                }
                case SolverType::SQP: {
                    SQPSolver_ = ::casadi::nlpsol("solver", "sqpmethod", nlp, solver_opts);
                }
                case SolverType::MIXED: {
                    IPOPTSolver_ = ::casadi::nlpsol("solver", "ipopt", nlp, solver_opts);
                    SQPSolver_ = ::casadi::nlpsol("solver", "sqpmethod", nlp, solver_opts);
                }
                case SolverType::CUDA_SQP: {
//                    OSQPSolver_ =  SQPOptimizationSolver("solver",nlp,solver_opts);
                }
            }

        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to create solvers: " + std::string(e.what()));
        }
//        如果不生成c代码，那么直接返回
        if (!genCode_) {
            return;
        } else {
            // 文件路径设置
            const std::string code_dir = packagePath_ + "/code_gen/";
            // 确保目标目录存在
            if (std::system(("mkdir -p " + code_dir).c_str()) != 0) {
                throw std::runtime_error("Failed to create code generation directory");
            }
            // 复制文件
            auto copyFile = [](const std::string &src, const std::string &dst) {
                std::ifstream source(src, std::ios::binary);
                if (!source) {
                    throw std::runtime_error("Cannot open source file: " + src);
                }
                std::ofstream dest(dst, std::ios::binary);
                if (!dest) {
                    throw std::runtime_error("Cannot open destination file: " + dst);
                }

                dest << source.rdbuf();
                if (!dest) {
                    throw std::runtime_error("Failed to copy file content from " + src + " to " + dst);
                }
            };
            // 修改后的编译函数，使用 this 指针访问 verbose_
            auto compileLibrary = [this](const std::string &source_file,
                                         const std::string &output_file,
                                         const std::string &compile_flags) {
                std::string compile_cmd = "gcc " + compile_flags + " " + source_file + " -o " + output_file;

                if (this->verbose_) {
                    std::cout << "Compiling with command: " << compile_cmd << std::endl;
                }

                if (std::system(compile_cmd.c_str()) != 0) {
                    throw std::runtime_error("Compilation failed for " + source_file);
                }

                if (this->verbose_) {
                    std::cout << "Successfully compiled " << output_file << std::endl;
                }
            };
            const std::string compile_flags = "-fPIC -shared -O3";
            // 生成代码文件
            try {
                const std::string IPOPT_solver_file_name = "IPOPT_nlp_code";
                const std::string IPOPT_solver_source_file = IPOPT_solver_file_name + ".c";
                const std::string IPOPT_target_file = code_dir + IPOPT_solver_source_file;
                const std::string IPOPT_shared_lib = code_dir + IPOPT_solver_file_name + ".so";
                IPOPTSolver_.generate_dependencies(IPOPT_solver_source_file);
                compileLibrary(IPOPT_target_file, IPOPT_shared_lib, compile_flags);
                copyFile(IPOPT_solver_source_file, IPOPT_target_file);
            } catch (const std::exception &e) {
                throw std::runtime_error("Failed to generate solver dependencies: " + std::string(e.what()));
            }
            // 生成代码文件
            try {
                const std::string SQP_solver_file_name = "SQP_nlp_code";
                const std::string SQP_solver_source_file = SQP_solver_file_name + ".c";
                const std::string SQP_target_file = code_dir + SQP_solver_source_file;
                const std::string SQP_shared_lib = code_dir + SQP_solver_file_name + ".so";
                SQPSolver_.generate_dependencies(SQP_solver_source_file);
                compileLibrary(SQP_target_file, SQP_shared_lib, compile_flags);
                copyFile(SQP_solver_source_file, SQP_target_file);
            } catch (const std::exception &e) {
                throw std::runtime_error("Failed to generate solver dependencies: " + std::string(e.what()));
            }
            // 输出问题规模信息
            if (verbose_) {
                const auto num_vars = vars.size1();
                const auto num_constraints = constraints.size1();
                const auto num_params = reference_.size1();
                std::cout << "Problem dimensions:\n"
                          << "Variables: " << num_vars << "\n"
                          << "Constraints: " << num_constraints << "\n"
                          << "Parameters: " << num_params << std::endl;
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error in genSolver: " << e.what() << std::endl;
        throw; // 重新抛出异常以允许上层处理
    }
}

::casadi::SX OptimalControlProblem::getReference() const {
    return reference_;
}

/*
 * 输入当前OCP问题系统的状态，以及参考轨迹，计算出最优轨迹
 * bug : 在调用这个函数的时候，发现求解器接收的constraint的大小是不对的
 * */
void OptimalControlProblem::computeOptimalTrajectory(const ::casadi::DM &frame, const ::casadi::DM &reference) {
    if (frame.size1() != OCPConfigPtr_->getFrameSize()) {
        std::cerr << "优化问题的状态维度不对，收到状态是" << frame.size1() << "维，期望是"
                  << OCPConfigPtr_->getFrameSize()
                  << "维\n";
    }
    if (reference.size1() != reference_.size1()) {
        std::cerr << "参考轨迹的维度不对，收到参考轨迹是" << reference.size1() << "维，期望是" << reference_.size1()
                  << "维\n";
    }

    // 1. 准备优化器输入参数
    std::map<std::string, ::casadi::DM> arg, res;
    // 设置变量和约束的上下界
    ::casadi::DM lbx = ::casadi::DM::vertcat(OCPConfigPtr_->getLowerBounds());
    ::casadi::DM ubx = ::casadi::DM::vertcat(OCPConfigPtr_->getUpperBounds());
    std::cout << "OCPConfigPtr_->getFrameSize()" << OCPConfigPtr_->getFrameSize();
    lbx(::casadi::Slice(0, OCPConfigPtr_->getFrameSize())) = frame;
    ubx(::casadi::Slice(0, OCPConfigPtr_->getFrameSize())) = frame;
    if (verbose_) {
        std::cout << "变量下界:\n" << lbx << std::endl;
        std::cout << "变量上界:\n" << ubx << std::endl;
    }
    ::casadi::DM lbg = ::casadi::DM::vertcat({getConstraintLowerBounds()});
    ::casadi::DM ubg = ::casadi::DM::vertcat({getConstraintUpperBounds()});

    // 设置初始猜测全都是0
    const int variableSize = OCPConfigPtr_->getFrameSize();

    ::casadi::DM initialGuess;
    if (setInitialGuess_) {
        initialGuess = OCPConfigPtr_->getInitialGuess();
    } else {
        initialGuess = ::casadi::DM::repmat(::casadi::DM::zeros(variableSize, 1), OCPConfigPtr_->getHorizon());
    }
    // 组装求解器输入
    arg["lbx"] = lbx;
    arg["ubx"] = ubx;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    if (firstTime_) {
        arg["x0"] = initialGuess;
    } else {
        arg["x0"] = optimalTrajectory_;
    }
    arg["p"] = reference;

    // 2. 求解优化问题
    if (solverInputCheck(arg)) {
        try {
            if (genCode_) {
                if (firstTime_) {
                    // 使用生成的代码求解
                    libIPOPTSolver_ = ::casadi::nlpsol("ipopt_solver", "ipopt",
                                                       packagePath_ + "/code_gen/IPOPT_nlp_code.so");
                    libSQPSolver_ = ::casadi::nlpsol("sqpmethod_solver", "sqpmethod",
                                                     packagePath_ + "/code_gen/SQP_nlp_code.so");

                    res = libIPOPTSolver_(arg);
                    firstTime_ = false;
                    std::cout << "暖机完成，已取得当前的全局最优解\n";
                } else {
                    // 根据求解器类型选择不同的求解器
                    switch (selectedSolver_) {
                        case SolverType::IPOPT:
                            res = libIPOPTSolver_(arg);
                            break;
                        case SolverType::SQP:
                            res = libSQPSolver_(arg);
                            break;
                        default:
                            res = libIPOPTSolver_(arg);  // 默认使用IPOPT
                            break;
                    }
                }
            } else {
                if (firstTime_) {
                    // 根据求解器类型选择不同的求解器
                    switch (selectedSolver_) {
                        case SolverType::IPOPT:
                            res = IPOPTSolver_(arg);
                            break;
                        case SolverType::SQP:
                            res = SQPSolver_(arg);
                            break;
                        case SolverType::MIXED:
                            res = IPOPTSolver_(arg);
                            break;
                        default:
                            res = IPOPTSolver_(arg);  // 默认使用IPOPT
                            break;
                    }
                    res = IPOPTSolver_(arg);
                    firstTime_ = false;
                } else {
                    // 根据求解器类型选择不同的求解器
                    switch (selectedSolver_) {
                        case SolverType::IPOPT:
                            res = IPOPTSolver_(arg);
                            break;
                        case SolverType::SQP:
                            res = SQPSolver_(arg);
                            break;
                        case SolverType::MIXED:
                            res = SQPSolver_(arg);
                            break;
                        case SolverType::CUDA_SQP:
//                            todo :使用SQPSolverUtils内的算法求解
                            break;
                        default:
                            res = IPOPTSolver_(arg);  // 默认使用IPOPT
                            break;
                    }
                }
            }
            // 3. 输出结果
            std::cout << "\n=================== 优化结果 ===================" << std::endl;
            std::cout << "目标函数值: " << res.at("f") << std::endl;
            optimalTrajectory_ = res.at("x");
            if (verbose_) {
                std::cout << "最优解: " << res.at("x") << std::endl;
            }
            // saveResultsToCSV(res.at("x"));
        } catch (const std::exception &e) {
            std::cerr << "优化求解失败: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "求解器输入检查失败" << std::endl;
    }
}


bool OptimalControlProblem::solverInputCheck(std::map<std::string, ::casadi::DM> arg) const {
    auto printDimensionMismatch = [](const std::string &name, int expected, int actual) {
        std::cerr << name << " 的维度不对: 期望 " << expected << ", 实际 " << actual << std::endl;
    };

    int expected_lbg_ubg_size = ::casadi::DM::vertcat(getConstraintLowerBounds()).size1();
    if (arg["lbg"].size1() != expected_lbg_ubg_size) {
        printDimensionMismatch("lbg", expected_lbg_ubg_size, arg["lbg"].size1());
        return false;
    }
    if (arg["ubg"].size1() != expected_lbg_ubg_size) {
        printDimensionMismatch("ubg", expected_lbg_ubg_size, arg["ubg"].size1());
        return false;
    }

    int expected_lbx_ubx_x0_size = OCPConfigPtr_->getVariables().size1();
    if (arg["lbx"].size1() != expected_lbx_ubx_x0_size) {
        printDimensionMismatch("lbx", expected_lbx_ubx_x0_size, arg["lbx"].size1());
        return false;
    }
    if (arg["ubx"].size1() != expected_lbx_ubx_x0_size) {
        printDimensionMismatch("ubx", expected_lbx_ubx_x0_size, arg["ubx"].size1());
        return false;
    }
    if (arg["x0"].size1() != expected_lbx_ubx_x0_size) {
        printDimensionMismatch("x0", expected_lbx_ubx_x0_size, arg["x0"].size1());
        return false;
    }
//    reference的维度就是状态status的维度
    int expected_p_size = OCPConfigPtr_->getFrameSize();
    if (arg["p"].size1() != expected_p_size) {
        printDimensionMismatch("p", expected_p_size, arg["p"].size1());
        return false;
    }
    if (verbose_) {
        std::cout << "所有维度检查通过。" << std::endl;
        std::cout << "lbg/ubg 维度: " << expected_lbg_ubg_size << std::endl;
        std::cout << "lbx/ubx/x0 维度: " << expected_lbx_ubx_x0_size << std::endl;
        std::cout << "p 维度: " << expected_p_size << std::endl;
    }
    return true;
}

casadi::DM OptimalControlProblem::getOptimalTrajectory() {
    return optimalTrajectory_;
}

std::vector<casadi::SX> OptimalControlProblem::getConstraints() const {
    return constraints_;
}

casadi::DMVector OptimalControlProblem::getConstraintLowerBounds() const {
    return constraintLowerBounds_;
}

casadi::DMVector OptimalControlProblem::getConstraintUpperBounds() const {
    return constraintUpperBounds_;
}

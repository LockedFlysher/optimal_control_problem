//
// Created by lock on 2024/10/18.
//

#include "optimal_control_problem/OptimalControlProblem.h"
#include <iostream>
#include <rclcpp/logging.hpp>

//通过解析yaml文件，初始化dt、horizon,创建系统状态和输入变量
OptimalControlProblem::OptimalControlProblem(YAML::Node configNode) {
    // 获取包路径
    packagePath_ = ament_index_cpp::get_package_share_directory("optimal_control_problem");
    OCPConfigPtr_ = std::make_unique<OCPConfig>(configNode);
    configNode_ = configNode;
    solverSettings.maxIter = configNode["solver_settings"]["max_iter"].as<int>();
    solverSettings.warmStart = configNode["solver_settings"]["warm_start"].as<bool>();
    solverSettings.SQP_settings.alpha = configNode["solver_settings"]["SQP_settings"]["alpha"].as<double>();
    solverSettings.SQP_settings.stepNum = configNode["solver_settings"]["SQP_settings"]["step_num"].as<int>();
    solverSettings.verbose = configNode["solver_settings"]["verbose"].as<bool>();

    solverSettings.genCode = configNode["solver_settings"]["gen_code"].as<bool>();
    //    todo load lib并没有起作用
    solverSettings.loadLib = configNode["solver_settings"]["load_lib"].as<bool>();

    if (configNode["solver_settings"]["solve_method"].as<std::string>() == "IPOPT") {
        setSolverType(SolverSettings::SolverType::IPOPT);
    }
    if (configNode["solver_settings"]["solve_method"].as<std::string>() == "MIXED") {
        setSolverType(SolverSettings::SolverType::MIXED);
    }
    if (configNode["solver_settings"]["solve_method"].as<std::string>() == "SQP") {
        setSolverType(SolverSettings::SolverType::SQP);
    }
    if (configNode["solver_settings"]["solve_method"].as<std::string>() == "CUDA_SQP") {
        setSolverType(SolverSettings::SolverType::CUDA_SQP);
    }
    if (solverSettings.verbose) {
        std::cout << "输出c代码且编译动态链接库：" << solverSettings.genCode << std::endl;
        std::cout << "使用动态链接库对求解器进行加载：" << solverSettings.genCode << std::endl;
    }

}

void OptimalControlProblem::addScalarCost(const casadi::SX &cost) {
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

void OptimalControlProblem::setSolverType(SolverSettings::SolverType type) {
    solverSettings.solverType = type;
}

OptimalControlProblem::SolverSettings::SolverType OptimalControlProblem::getSolverType() const {
    return solverSettings.solverType;
}

/*
 * note : 必需在应用约束和添加损失以后使用！！！！！！！
 * 原则 ： solver一定要先生成
 * 再进行gencode
 * */
// todo : 需要在这里把问题SQP化，约束线性化，A\B矩阵都要作为参数放到ADMM求解器作为参数，问题就来了，A\B肯定是稀疏的才比较好求啊，
void OptimalControlProblem::genSolver() {
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
    // 设置求解器选项
    ::casadi::Dict basicOptions;
    basicOptions["verbose"] = solverSettings.verbose ? 1 : 0;
//    可以提高求解的效率，但是同时开启变慢
//    basicOptions["jit"] = true;
    switch (solverSettings.solverType) {
        case SolverSettings::SolverType::IPOPT: {
            ::casadi::Dict ipopt_options;
            // ipopt_options["print_level"] = 0;      // 静默模式
            // ipopt_options["max_iter"] = solverSettings.maxIter;      // 最大迭代次数
            // ipopt_options["tol"] = 1e-6;           // 收敛容差
            // ipopt_options["acceptable_tol"] = 1e-4; // 可接受的容差
            // ipopt_options["linear_solver"] = "mumps"; // 线性求解器选择
            // 将 basicOptions 中的键值对添加到 ipopt_options 中
            for (const auto& option : basicOptions) {
                ipopt_options[option.first] = option.second;
            }
            IPOPTSolver_ = ::casadi::nlpsol("solver", "ipopt", nlp, ipopt_options);
//            IPOPTSolver_.print_options();
            break;

        }
        case SolverSettings::SolverType::SQP: {
            casadi::Dict sqp_options;
            Dict solver_options;
            sqp_options["error_on_fail"] = false;
//            qpsol: 指定用于求解二次规划子问题的 QP 求解器，例如 'qpoases'、'osqp' 或 'nlpsol'。
            solver_options["qpsol"] = "nlpsol";
            solver_options["qpsol_options"] = sqp_options;
            sqp_options["error_on_fail"] = false;
//            hessian_approximation: 设置 Hessian 矩阵的近似方法，可能包括 'exact'（精确 Hessian）或 'limited-memory'（如 BFGS 近似）。
            sqp_options["hessian_approximation"] = "limited-memory";
            sqp_options["max_iter"] = solverSettings.SQP_settings.stepNum;
            for (const auto& option : basicOptions) {
                sqp_options[option.first] = option.second;
            }
            SQPSolver_ = ::casadi::nlpsol("solver", "sqpmethod", nlp, solver_options);
            break;
        }
        case SolverSettings::SolverType::MIXED: {

            ::casadi::Dict ipopt_options;
            ipopt_options["warm_start_init_point"] = solverSettings.warmStart ? "yes" : "no";
            ipopt_options["warm_start_bound_push"] = 0.001;
            // 将 basicOptions 中的键值对添加到 ipopt_options 中
            for (const auto& option : basicOptions) {
                ipopt_options[option.first] = option.second;
            }
            casadi::Dict sqp_options;

            sqp_options["qpsol"] = "qpoases";
//            hessian_approximation: 设置 Hessian 矩阵的近似方法，可能包括 'exact'（精确 Hessian）或 'limited-memory'（如 BFGS 近似）。
            sqp_options["hessian_approximation"] = "exact";
            sqp_options["max_iter"] = solverSettings.SQP_settings.stepNum;
            sqp_options["error_on_fail"] = false;

            for (const auto& option : basicOptions) {
                sqp_options[option.first] = option.second;
            }
            IPOPTSolver_ = ::casadi::nlpsol("solver", "ipopt", nlp, ipopt_options);
            SQPSolver_ = ::casadi::nlpsol("solver", "sqpmethod", nlp, sqp_options);
            break;
        }
        case SolverSettings::SolverType::CUDA_SQP: {
            casadi::Dict sqp_options;
//            qpsol: 指定用于求解二次规划子问题的 QP 求解器，例如 'qpoases'、'osqp' 或 'nlpsol'。
            sqp_options["qpsol"] = "cuda_sqp";
//            hessian_approximation: 设置 Hessian 矩阵的近似方法，可能包括 'exact'（精确 Hessian）或 'limited-memory'（如 BFGS 近似）。
            sqp_options["hessian_approximation"] = "exact";
            sqp_options["max_iter"] = solverSettings.SQP_settings.stepNum;
            sqp_options["alpha"] = solverSettings.SQP_settings.alpha;
            for (const auto& option : basicOptions) {
                sqp_options[option.first] = option.second;
            }
            OSQPSolverPtr_ = std::make_shared<SQPOptimizationSolver>(nlp, sqp_options);
            break;
        }
    }
    //        如果不生成c代码，那么直接返回
    if (!solverSettings.genCode) {
        return;
    } else {
        //保存localSystemFunction_.save("localSystemFunction.casadi")
        casadi::Function localSystemFunction = OSQPSolverPtr_->getSXLocalSystemFunction();
        // std::string CUSADIpath_ = ament_index_cpp::get_package_share_directory("cusadi");
        localSystemFunction.save("/home/andew/project/NEBULA_ws/src/Cusadi-SQP/function/localSystemFunction.casadi");
        std::cout << OSQPSolverPtr_->getSXLocalSystemFunction() << std::endl;
        std::cout << "LocalSystemFunction is saved" << std::endl;
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
        // 修改后的编译函数，使用 this 指针访问 solverSettings.verbose
        auto compileLibrary = [this](const std::string &source_file,
                                     const std::string &output_file,
                                     const std::string &compile_flags) {
            std::string compile_cmd = "gcc " + compile_flags + " " + source_file + " -o " + output_file;

            if (solverSettings.verbose) {
                std::cout << "Compiling with command: " << compile_cmd << std::endl;
            }

            if (std::system(compile_cmd.c_str()) != 0) {
                throw std::runtime_error("Compilation failed for " + source_file);
            }

            if (solverSettings.verbose) {
                std::cout << "Successfully compiled " << output_file << std::endl;
            }
        };
        const std::string compile_flags = "-fPIC -shared -O3";
        switch (solverSettings.solverType) {
            case SolverSettings::SolverType::IPOPT: {
                // 生成代码文件
                const std::string IPOPT_solver_file_name = "IPOPT_nlp_code";
                const std::string IPOPT_solver_source_file = IPOPT_solver_file_name + ".c";
                const std::string IPOPT_target_file = code_dir + IPOPT_solver_source_file;
                const std::string IPOPT_shared_lib = code_dir + IPOPT_solver_file_name + ".so";
                IPOPTSolver_.generate_dependencies(IPOPT_solver_source_file);
                copyFile(IPOPT_solver_source_file, IPOPT_target_file);
                compileLibrary(IPOPT_target_file, IPOPT_shared_lib, compile_flags);
                // copyFile(IPOPT_solver_source_file, IPOPT_target_file);
                break;
            }
            case SolverSettings::SolverType::SQP: {
                const std::string SQP_solver_file_name = "SQP_nlp_code";
                const std::string SQP_solver_source_file = SQP_solver_file_name + ".c";
                const std::string SQP_target_file = code_dir + SQP_solver_source_file;
                const std::string SQP_shared_lib = code_dir + SQP_solver_file_name + ".so";
                SQPSolver_.generate_dependencies(SQP_solver_source_file);
                compileLibrary(SQP_target_file, SQP_shared_lib, compile_flags);
                copyFile(SQP_solver_source_file, SQP_target_file);
                break;
            }
            case SolverSettings::SolverType::MIXED: {
//                两个都进行编译
                const std::string IPOPT_solver_file_name = "IPOPT_nlp_code";
                const std::string IPOPT_solver_source_file = IPOPT_solver_file_name + ".c";
                const std::string IPOPT_target_file = code_dir + IPOPT_solver_source_file;
                const std::string IPOPT_shared_lib = code_dir + IPOPT_solver_file_name + ".so";
                IPOPTSolver_.generate_dependencies(IPOPT_solver_source_file);
                compileLibrary(IPOPT_target_file, IPOPT_shared_lib, compile_flags);
                copyFile(IPOPT_solver_source_file, IPOPT_target_file);
                IPOPTSolver_ = ::casadi::nlpsol("solver", "ipopt", nlp, basicOptions);
                const std::string SQP_solver_file_name = "SQP_nlp_code";
                const std::string SQP_solver_source_file = SQP_solver_file_name + ".c";
                const std::string SQP_target_file = code_dir + SQP_solver_source_file;
                const std::string SQP_shared_lib = code_dir + SQP_solver_file_name + ".so";
                SQPSolver_.generate_dependencies(SQP_solver_source_file);
                compileLibrary(SQP_target_file, SQP_shared_lib, compile_flags);
                copyFile(SQP_solver_source_file, SQP_target_file);
                break;
            }
            case SolverSettings::SolverType::CUDA_SQP: {
                break;
            }
        }
        // 输出问题规模信息
        if (solverSettings.verbose) {
            const auto num_vars = vars.size1();
            const auto num_constraints = constraints.size1();
            const auto num_params = reference_.size1();
            std::cout << "Problem dimensions:\n"
                      << "Variables: " << num_vars << "\n"
                      << "Constraints: " << num_constraints << "\n"
                      << "Parameters: " << num_params << std::endl;
        }
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
    lbx(::casadi::Slice(0, OCPConfigPtr_->getFrameSize())) = frame;
    ubx(::casadi::Slice(0, OCPConfigPtr_->getFrameSize())) = frame;
    if (solverSettings.verbose) {
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

    // 2. 求解优化问题，如果使用代码生成则使用编译好的库
    if (solverInputCheck(arg)) {
        if (solverSettings.genCode||solverSettings.loadLib) {
            if (firstTime_) {
                // 使用生成的代码求解
                switch (solverSettings.solverType) {
                    case SolverSettings::SolverType::IPOPT:
                        libIPOPTSolver_ = ::casadi::nlpsol("ipopt_solver", "ipopt",
                                                           packagePath_ + "/code_gen/IPOPT_nlp_code.so");
                        res = libIPOPTSolver_(arg);
                        break;
                    case SolverSettings::SolverType::SQP:
                        libSQPSolver_ = ::casadi::nlpsol("sqpmethod_solver", "sqpmethod",
                                                         packagePath_ + "/code_gen/SQP_nlp_code.so");
                        res = libSQPSolver_(arg);
                        break;
                    case SolverSettings::SolverType::MIXED:
                        libIPOPTSolver_ = ::casadi::nlpsol("ipopt_solver", "ipopt",
                                                           packagePath_ + "/code_gen/IPOPT_nlp_code.so");
                        libSQPSolver_ = ::casadi::nlpsol("sqpmethod_solver", "sqpmethod",
                                                         packagePath_ + "/code_gen/SQP_nlp_code.so");
                        res = libIPOPTSolver_(arg);
                        break;
                    default:
                        libIPOPTSolver_ = ::casadi::nlpsol("ipopt_solver", "ipopt",
                                                           packagePath_ + "/code_gen/IPOPT_nlp_code.so");
                        res = libIPOPTSolver_(arg);
                        break;
                }
                firstTime_ = false;
                std::cout << "暖机完成，已取得当前的全局最优解\n";
            } else {
                // 根据求解器类型选择不同的求解器
                switch (solverSettings.solverType) {
                    case SolverSettings::SolverType::IPOPT:
                        res = libIPOPTSolver_(arg);
                        break;
                    case SolverSettings::SolverType::SQP:
                        res = libSQPSolver_(arg);
                        break;
                    case SolverSettings::SolverType::CUDA_SQP:
                        res = OSQPSolverPtr_->getOptimalSolution(arg);
                        break;
                    default:
                        res = libIPOPTSolver_(arg);  // 默认使用IPOPT
                        break;
                }
            }
        } else {
            if (firstTime_) {
                // 根据求解器类型选择不同的求解器
                switch (solverSettings.solverType) {
                    case SolverSettings::SolverType::IPOPT:
                        res = IPOPTSolver_(arg);
                        break;
                    case SolverSettings::SolverType::SQP:
                        res = SQPSolver_(arg);
                        break;
                    case SolverSettings::SolverType::MIXED:
                        res = IPOPTSolver_(arg);
                        break;
                    case SolverSettings::SolverType::CUDA_SQP:
                        res = OSQPSolverPtr_->getOptimalSolution(arg);
                        break;
                }
                firstTime_ = false;
            } else {
                // 根据求解器类型选择不同的求解器
                switch (solverSettings.solverType) {
                    case SolverSettings::SolverType::IPOPT:
                        res = IPOPTSolver_(arg);
                        break;
                    case SolverSettings::SolverType::SQP:
                        res = SQPSolver_(arg);
                        break;
                    case SolverSettings::SolverType::MIXED:
                        res = SQPSolver_(arg);
                        break;
                    case SolverSettings::SolverType::CUDA_SQP:
                        res = OSQPSolverPtr_->getOptimalSolution(arg);
                        break;
                }
            }
        }
        // 3. 输出结果
        std::cout << "\n=================== 优化结果 ===================" << std::endl;
        std::cout << "目标函数值: " << res.at("f") << std::endl;
        optimalTrajectory_ = res.at("x");
        if (solverSettings.verbose) {
            std::cout << "最优解: " << res.at("x") << std::endl;
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

    int expected_p_size = reference_.size1();
    if (arg["p"].size1() != expected_p_size) {
        printDimensionMismatch("p", expected_p_size, arg["p"].size1());
        return false;
    }
    if (solverSettings.verbose) {
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

void OptimalControlProblem::setReference(const casadi::SX &reference) {
    reference_ = reference;
}

::casadi::DM OptimalControlProblem::getOptimalInputFirstFrame() {
    //根据你的命名来提取输入的变量
    int inputFrameSize_ = OCPConfigPtr_->getVariable(0, "F").size1();
    const int variableSize = OCPConfigPtr_->getFrameSize();
    // 收集所有输入元素的索引
    std::vector<casadi_int> inputIndices;
    for (int i = 0; i < OCPConfigPtr_->getHorizon(); ++i) {
        int cycleStart = i * (variableSize);               // 当前周期的起始索引
        int inputStart = cycleStart + variableSize - inputFrameSize_; // 当前周期内输入的起始索引
        for (int j = 0; j < inputFrameSize_; ++j) {
            inputIndices.push_back(inputStart + j);   // 将输入索引逐个加入列表
        }
    }
    // 提取所有输入元素
    ::casadi::DM inputs = optimalTrajectory_(inputIndices);
    std::cout << "输入的所有帧是" << std::endl;
    std::cout << inputs;
    ::casadi::DM inputFirstFrame = inputs(::casadi::Slice(inputFrameSize_, 2 * inputFrameSize_, 1));
    std::vector<double> vec;
    vec = inputFirstFrame.get_elements(); // 将所有元素复制到
    std::cout << "\n输出的第二帧是" << vec << std::endl;
    return ::casadi::DM(vec);
}

void OptimalControlProblem::addVectorCost(const casadi::DM &param, const SX &cost) {
    if (param.size1() != cost.size1()) {
        std::cout << "损失的符号向量和参数向量维度不一致\n";
        return;
    }
    // 计算二次形式的标量损失: cost^T * diag(param) * cost
    SX weightedCost = SX::zeros(1, 1);
    for (int i = 0; i < cost.size1(); ++i) {
        weightedCost += param(i).scalar() * cost(i) * cost(i);
    }
    // 将标量损失添加到总损失中
    addScalarCost(weightedCost);
}

void OptimalControlProblem::addVectorCost(const std::vector<double> &param, const SX &cost) {
    if (param.size() != cost.size1()) {
        std::cout << "损失的符号向量和参数向量维度不一致\n";
        exit(-5);
    }
    // 计算二次形式的标量损失: cost^T * diag(param) * cost
    SX weightedCost = SX::zeros(1, 1);
    for (int i = 0; i < cost.size1(); ++i) {
        weightedCost += param[i] * cost(i) * cost(i);
    }
    // 将标量损失添加到总损失中
    addScalarCost(weightedCost);
}


//
// Created by lock on 2024/10/18.
//

#include "optimal_control_problem/OptimalControlProblem.h"

#include "optimal_control_problem/OptimalControlProblem.h"
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <cstdlib>  // system调用python脚本


OptimalControlProblem::OptimalControlProblem(YAML::Node configNode) {
    try {
        if (!validateConfig(configNode)) {
            throw std::runtime_error("Invalid configuration file");
        }

        packagePath_ = ament_index_cpp::get_package_share_directory("optimal_control_problem");
        OCPConfigPtr_ = std::make_unique<OCPConfig>(configNode);
        configNode_ = configNode;

        // 读取solver设置
        const auto& solverSettings_node = configNode["solver_settings"];
        solverSettings.maxIter = solverSettings_node["max_iter"].as<int>();
        solverSettings.warmStart = solverSettings_node["warm_start"].as<bool>();
        solverSettings.SQP_settings.alpha = solverSettings_node["SQP_settings"]["alpha"].as<double>();
        solverSettings.SQP_settings.stepNum = solverSettings_node["SQP_settings"]["step_num"].as<int>();
        solverSettings.verbose = solverSettings_node["verbose"].as<bool>();
        solverSettings.genCode = solverSettings_node["gen_code"].as<bool>();
        solverSettings.loadLib = solverSettings_node["load_lib"].as<bool>();

        const std::string solveMethod = solverSettings_node["solve_method"].as<std::string>();
        if (solveMethod == "IPOPT") {
            setSolverType(SolverSettings::SolverType::IPOPT);
        } else if (solveMethod == "MIXED") {
            setSolverType(SolverSettings::SolverType::MIXED);
        } else if (solveMethod == "SQP") {
            setSolverType(SolverSettings::SolverType::SQP);
        } else if (solveMethod == "CUDA_SQP") {
            setSolverType(SolverSettings::SolverType::CUDA_SQP);
        } else {
            throw std::invalid_argument("Unknown solver type: " + solveMethod);
        }

        if (solverSettings.verbose) {
            std::cout << "输出c代码且编译动态链接库：" << solverSettings.genCode << std::endl;
            std::cout << "使用动态链接库对求解器进行加载：" << solverSettings.loadLib << std::endl;
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing YAML configuration: " + std::string(e.what()));
    }
}

bool OptimalControlProblem::validateConfig(const YAML::Node& config) {
    if (!config["solver_settings"]) return false;
    const auto& solver = config["solver_settings"];

    return solver["max_iter"] && solver["warm_start"] &&
           solver["SQP_settings"] && solver["SQP_settings"]["alpha"] &&
           solver["SQP_settings"]["step_num"] && solver["verbose"] &&
           solver["gen_code"] && solver["load_lib"] && solver["solve_method"];
}

bool OptimalControlProblem::checkDirectoryPermissions(const std::string& path) {
    try {
        std::filesystem::path dir_path(path);
        if (!std::filesystem::exists(dir_path)) {
            return std::filesystem::create_directories(dir_path);
        }
        // 检查写入权限
        return access(path.c_str(), W_OK) == 0;
    } catch (const std::filesystem::filesystem_error& e) {
        return false;
    }
}


void OptimalControlProblem::computeOptimalTrajectory(const ::casadi::DM &frame, const ::casadi::DM &reference) {
    if (frame.size1() != OCPConfigPtr_->getFrameSize()) {
        throw std::invalid_argument("State dimension mismatch: received " +
                                    std::to_string(frame.size1()) +
                                    ", expected " +
                                    std::to_string(OCPConfigPtr_->getFrameSize()));
    }
    if (reference.size1() != reference_.size1()) {
        throw std::invalid_argument("Reference dimension mismatch: received " +
                                    std::to_string(reference.size1()) +
                                    ", expected " +
                                    std::to_string(reference_.size1()));
    }

    std::map<std::string, ::casadi::DM> arg, res;
    ::casadi::DM lbx = ::casadi::DM::vertcat(OCPConfigPtr_->getLowerBounds());
    ::casadi::DM ubx = ::casadi::DM::vertcat(OCPConfigPtr_->getUpperBounds());
    lbx(::casadi::Slice(0, OCPConfigPtr_->getFrameSize())) = frame;
    ubx(::casadi::Slice(0, OCPConfigPtr_->getFrameSize())) = frame;

    ::casadi::DM lbg = ::casadi::DM::vertcat({getConstraintLowerBounds()});
    ::casadi::DM ubg = ::casadi::DM::vertcat({getConstraintUpperBounds()});

    const int variableSize = OCPConfigPtr_->getFrameSize();
    ::casadi::DM initialGuess;
    if (setInitialGuess_) {
        initialGuess = OCPConfigPtr_->getInitialGuess();
    } else {
        initialGuess = ::casadi::DM::repmat(::casadi::DM::zeros(variableSize, 1), OCPConfigPtr_->getHorizon());
    }

    arg["lbx"] = lbx;
    arg["ubx"] = ubx;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    arg["x0"] = firstTime_ ? initialGuess : optimalTrajectory_;
    arg["p"] = reference;

    if (!solverInputCheck(arg)) {
        throw std::runtime_error("Solver input validation failed");
    }
    try {
        if (solverSettings.genCode || solverSettings.loadLib) {
            if (firstTime_) {
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
                    case SolverSettings::SolverType::CUDA_SQP:
                        res = OSQPSolverPtr_->getOptimalSolution(arg);
                        break;
                }
                firstTime_ = false;
            } else {
                switch (solverSettings.solverType) {
                    case SolverSettings::SolverType::IPOPT:
                        res = libIPOPTSolver_(arg);
                        break;
                    case SolverSettings::SolverType::SQP:
                        res = libSQPSolver_(arg);
                        break;
                    case SolverSettings::SolverType::MIXED:
                        // 根据收敛性能决定使用哪个求解器
                        if (optimalTrajectory_.is_empty() ||
                            (res.count("f") > 0 && res.at("f").scalar() > 1e-6)) {
                            res = libIPOPTSolver_(arg);
                        } else {
                            res = libSQPSolver_(arg);
                        }
                        break;
                    case SolverSettings::SolverType::CUDA_SQP:
                        res = OSQPSolverPtr_->getOptimalSolution(arg);
                        break;
                }
            }
        } else {
            if (firstTime_) {
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
                switch (solverSettings.solverType) {
                    case SolverSettings::SolverType::IPOPT:
                        res = IPOPTSolver_(arg);
                        break;
                    case SolverSettings::SolverType::SQP:
                        res = SQPSolver_(arg);
                        break;
                    case SolverSettings::SolverType::MIXED:
                        // 根据收敛性能决定使用哪个求解器
                        if (optimalTrajectory_.is_empty() ||
                            (res.count("f") > 0 && res.at("f").scalar() > 1e-6)) {
                            res = IPOPTSolver_(arg);
                        } else {
                            res = SQPSolver_(arg);
                        }
                        break;
                    case SolverSettings::SolverType::CUDA_SQP:
                        res = OSQPSolverPtr_->getOptimalSolution(arg);
                        break;
                }
            }
        }

        if (res.empty()) {
            throw std::runtime_error("Solver returned empty result");
        }

        optimalTrajectory_ = res.at("x");

        if (solverSettings.verbose) {
            std::cout << "\n=================== 优化结果 ===================" << std::endl;
            std::cout << "目标函数值: " << res.at("f") << std::endl;
            std::cout << "最优解: " << res.at("x") << std::endl;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Optimization failed: " + std::string(e.what()));
    }
}

void OptimalControlProblem::genSolver() {
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

    ::casadi::Dict basicOptions;
    basicOptions["verbose"] = solverSettings.verbose ? 1 : 0;
    basicOptions["jit"] = false;  // 临时禁用JIT以避免代码生成时的冲突

    // 获取当前工作目录的绝对路径
    std::filesystem::path current_path = std::filesystem::current_path();
    const std::string code_dir = packagePath_ + "/code_gen/";

    // 确保code_dir是绝对路径
    std::filesystem::path code_dir_abs = std::filesystem::absolute(code_dir);

    if (!checkDirectoryPermissions(code_dir_abs.string())) {
        throw std::runtime_error("Cannot create or write to code generation directory: " + code_dir_abs.string());
    }

    try {
        switch (solverSettings.solverType) {
            case SolverSettings::SolverType::IPOPT: {
                ::casadi::Dict ipopt_options = basicOptions;
                IPOPTSolver_ = ::casadi::nlpsol("ipopt_solver", "ipopt", nlp, ipopt_options);

                if (solverSettings.genCode) {
                    // 使用当前目录的绝对路径
                    std::filesystem::path temp_source = current_path / "ipopt_temp.c";
                    std::filesystem::path target_source = code_dir_abs / "IPOPT_nlp_code.c";
                    std::filesystem::path target_lib = code_dir_abs / "IPOPT_nlp_code.so";

                    if (solverSettings.verbose) {
                        std::cout << "Generating IPOPT solver code at: " << temp_source << std::endl;
                    }

                    // 在当前目录生成临时文件
                    IPOPTSolver_.generate_dependencies("ipopt_temp.c");

                    // 验证临时文件生成
                    if (!std::filesystem::exists(temp_source)) {
                        throw std::runtime_error("Failed to generate IPOPT source file at: " + temp_source.string());
                    }

                    // 移动到目标目录
                    std::filesystem::copy_file(temp_source, target_source,
                                               std::filesystem::copy_options::overwrite_existing);
                    std::filesystem::remove(temp_source);

                    compileLibrary(target_source.string(), target_lib.string(), "-fPIC -shared -O1");
                }
                break;
            }
            case SolverSettings::SolverType::SQP: {
                casadi::Dict sqp_options;
                Dict solver_options;
                solver_options["qpsol"] = "qpoases";
                solver_options["qpsol_options"] = sqp_options;
                sqp_options["error_on_fail"] = false;
                sqp_options["hessian_approximation"] = "exact";
                sqp_options["max_iter"] = solverSettings.SQP_settings.stepNum;

                for (const auto& option : basicOptions) {
                    sqp_options[option.first] = option.second;
                }

                SQPSolver_ = ::casadi::nlpsol("sqp_solver", "sqpmethod", nlp, solver_options);

                if (solverSettings.genCode) {
                    std::filesystem::path temp_source = current_path / "sqp_temp.c";
                    std::filesystem::path target_source = code_dir_abs / "SQP_nlp_code.c";
                    std::filesystem::path target_lib = code_dir_abs / "SQP_nlp_code.so";

                    if (solverSettings.verbose) {
                        std::cout << "Generating SQP solver code at: " << temp_source << std::endl;
                    }

                    SQPSolver_.generate_dependencies("sqp_temp.c");

                    if (!std::filesystem::exists(temp_source)) {
                        throw std::runtime_error("Failed to generate SQP source file at: " + temp_source.string());
                    }

                    std::filesystem::copy_file(temp_source, target_source,
                                               std::filesystem::copy_options::overwrite_existing);
                    std::filesystem::remove(temp_source);

                    compileLibrary(target_source.string(), target_lib.string(), "-fPIC -shared -O1");
                }
                break;
            }
            case SolverSettings::SolverType::MIXED: {
                // IPOPT配置
                ::casadi::Dict ipopt_options = basicOptions;
                IPOPTSolver_ = ::casadi::nlpsol("mixed_ipopt_solver", "ipopt", nlp, ipopt_options);

                // SQP配置
                casadi::Dict sqp_options;
                sqp_options["qpsol"] = "qpoases";
                sqp_options["error_on_fail"] = false;
                sqp_options["hessian_approximation"] = "exact";
                sqp_options["max_iter"] = solverSettings.SQP_settings.stepNum;

                for (const auto& option : basicOptions) {
                    sqp_options[option.first] = option.second;
                }

                SQPSolver_ = ::casadi::nlpsol("mixed_sqp_solver", "sqpmethod", nlp, sqp_options);

                if (solverSettings.genCode) {
                    // 生成IPOPT代码
                    std::filesystem::path ipopt_temp = current_path / "mixed_ipopt_temp.c";
                    std::filesystem::path ipopt_target = code_dir_abs / "IPOPT_nlp_code.c";
                    std::filesystem::path ipopt_lib = code_dir_abs / "IPOPT_nlp_code.so";

                    if (solverSettings.verbose) {
                        std::cout << "Generating mixed IPOPT solver code at: " << ipopt_temp << std::endl;
                    }

                    IPOPTSolver_.generate_dependencies("mixed_ipopt_temp.c");

                    if (!std::filesystem::exists(ipopt_temp)) {
                        throw std::runtime_error("Failed to generate IPOPT source file at: " + ipopt_temp.string());
                    }

                    std::filesystem::copy_file(ipopt_temp, ipopt_target,
                                               std::filesystem::copy_options::overwrite_existing);
                    std::filesystem::remove(ipopt_temp);

                    compileLibrary(ipopt_target.string(), ipopt_lib.string(), "-fPIC -shared -O1");

                    // 生成SQP代码
                    std::filesystem::path sqp_temp = current_path / "mixed_sqp_temp.c";
                    std::filesystem::path sqp_target = code_dir_abs / "SQP_nlp_code.c";
                    std::filesystem::path sqp_lib = code_dir_abs / "SQP_nlp_code.so";

                    if (solverSettings.verbose) {
                        std::cout << "Generating mixed SQP solver code at: " << sqp_temp << std::endl;
                    }

                    SQPSolver_.generate_dependencies("mixed_sqp_temp.c");

                    if (!std::filesystem::exists(sqp_temp)) {
                        throw std::runtime_error("Failed to generate SQP source file at: " + sqp_temp.string());
                    }

                    std::filesystem::copy_file(sqp_temp, sqp_target,
                                               std::filesystem::copy_options::overwrite_existing);
                    std::filesystem::remove(sqp_temp);

                    compileLibrary(sqp_target.string(), sqp_lib.string(), "-fPIC -shared -O1");
                }
                break;
            }
            case SolverSettings::SolverType::CUDA_SQP: {
                casadi::Dict sqp_options;
                sqp_options["qpsol"] = "cuda_sqp";
                sqp_options["max_iter"] = solverSettings.SQP_settings.stepNum;
                sqp_options["alpha"] = solverSettings.SQP_settings.alpha;

                for (const auto& option : basicOptions) {
                    sqp_options[option.first] = option.second;
                }

                OSQPSolverPtr_ = std::make_shared<SQPOptimizationSolver>(nlp, sqp_options);

                if (solverSettings.genCode) {
                    casadi::Function localSystemFunction = OSQPSolverPtr_->getSXLocalSystemFunction();
                    std::filesystem::path temp_file = current_path / "localSystemFunction_temp.casadi";
                    std::filesystem::path target_file = code_dir_abs / "localSystemFunction.casadi";

                    if (solverSettings.verbose) {
                        std::cout << "Saving LocalSystemFunction at: " << temp_file << std::endl;
                    }

                    localSystemFunction.save(temp_file.string());

                    if (!std::filesystem::exists(temp_file)) {
                        throw std::runtime_error("Failed to save LocalSystemFunction at: " + temp_file.string());
                    }

                    std::filesystem::copy_file(temp_file, target_file,
                                               std::filesystem::copy_options::overwrite_existing);
                    std::filesystem::remove(temp_file);

                    if (solverSettings.verbose) {
                        std::cout << "LocalSystemFunction saved successfully to: " << target_file << std::endl;
                    }
                    const std::string cusadi_function_path = packagePath_ + "/script/cusadi/src/casadi_functions";
                    std::filesystem::copy_file(target_file, cusadi_function_path,
                           std::filesystem::copy_options::overwrite_existing);
                    const std::string run_codegen_path = packagePath_ + "/script/cusadi/run_codegen.py";
                    const std::string command = "python3 " + run_codegen_path + " --fn=localSystemFunction";
                    int result = std::system(command.c_str());
                    // 检查命令执行结果
                    if (result != 0) {
                        std::cerr << "错误：run_codegen脚本执行失败 (退出码 " << result << ")" << std::endl;
                }
                break;
            }
        }

        if (solverSettings.verbose) {
            const auto num_vars = vars.size1();
            const auto num_constraints = constraints.size1();
            const auto num_params = reference_.size1();
            std::cout << "Problem dimensions:\n"
                      << "Variables: " << num_vars << "\n"
                      << "Constraints: " << num_constraints << "\n"
                      << "Parameters: " << num_params << std::endl;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to generate solver: " + std::string(e.what()));
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

::casadi::SX OptimalControlProblem::getReference() const {
    return reference_;
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

void OptimalControlProblem::compileLibrary(const std::string &source_file,
                                           const std::string &output_file,
                                           const std::string &compile_flags) {
    std::string compile_cmd = "gcc " + compile_flags + " " + source_file + " -o " + output_file;

    if (solverSettings.verbose) {
        std::cout << "Compiling with command: " << compile_cmd << std::endl;
    }

    // 检查源文件是否存在
    if (!std::filesystem::exists(source_file)) {
        throw std::runtime_error("Source file does not exist: " + source_file);
    }

    // 检查输出目录是否存在，如果不存在则创建
    std::filesystem::path output_path(output_file);
    auto output_dir = output_path.parent_path();
    if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
        if (!std::filesystem::create_directories(output_dir)) {
            throw std::runtime_error("Failed to create output directory: " + output_dir.string());
        }
    }

    // 执行编译命令
    int compile_result = std::system(compile_cmd.c_str());
    if (compile_result != 0) {
        throw std::runtime_error("Compilation failed for " + source_file +
                                 " with exit code: " + std::to_string(compile_result));
    }

    // 验证输出文件是否生成
    if (!std::filesystem::exists(output_file)) {
        throw std::runtime_error("Compilation completed but output file not found: " + output_file);
    }

    if (solverSettings.verbose) {
        std::cout << "Successfully compiled " << output_file << std::endl;
    }
}

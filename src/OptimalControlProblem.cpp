//
// Created by lock on 2024/10/18.
//

#include "optimal_control_problem/OptimalControlProblem.h"
#include <iostream>

//通过解析yaml文件，初始化dt、horizon,创建系统状态和输入变量
OptimalControlProblem::OptimalControlProblem(const std::string& configFilePath) {
    configNode_ = YAML::LoadFile(configFilePath);
    if(!configNode_["mpc_options"]["horizon"]){
        throw std::invalid_argument("node [mpc_options.horizon] not found in YAML file");
    }else{
        horizon_ = configNode_["mpc_options"]["horizon"].as<int>();
    }
    if(!configNode_["mpc_options"]["dt"]){
        throw std::invalid_argument("node [mpc_options.dt] not found in YAML file");
    }else{
        dt_ = configNode_["mpc_options"]["dt"].as<float>();
    }
    parseOCPBounds();
    statusVariables = casadi::SX::sym("X", horizon_ * statusFrame_.totalSize, 1);
    inputVariables = casadi::SX::sym("U", horizon_ * inputFrame_.totalSize, 1);
    //    这里的reference是一个符号变量，是用来计算cost的，求解器的创建也需要它
    reference_ = ::casadi::SX::sym("ref", statusFrame_.totalSize, 1);
    if (configNode_["verbose"]["variables"].as<bool>()) {
        std::cout << "变量输出，可以通过[verbose.variables]进行关闭\n";
        std::cout <<"statusVariables:\n"<< statusVariables<<"\n inputVariables:\n" << inputVariables<<"\n";
        std::cout<< "reference预占位符号变量\n："<<reference_<<"\n";
        std::cout<<"状态变量下界"<<statusLowerBounds_<<std::endl;
        std::cout<<"状态变量上界"<<statusUpperBounds_<<"\n";
        std::cout<<"formatted status variables:\n";
    }
    genCode_ = configNode_["gen_code"].as<bool>();
    loadLib_ = configNode_["load_lib"].as<bool>();
    std::cout<<"输出c代码且编译动态链接库："<<genCode_<<std::endl;
    std::cout<<"使用动态链接库对求解器进行加载："<<genCode_<<std::endl;
}

void OptimalControlProblem::parseOCPBounds() {
    casadi::SXVector lowerStatusBound, upperStatusBound;
    if (!configNode_["OCP_variables"]) {
        throw std::invalid_argument("node [OCP_variables] not found in YAML file");
    }
    const YAML::Node& status_frame = configNode_["OCP_variables"]["status_frame"];
    initializeFrameWithYAML(statusFrame_, status_frame);
    if (!status_frame.IsSequence()) {
        throw std::invalid_argument("status_frame should be a sequence");
    }
    // 遍历所有状态变量
    for (const auto& var : status_frame) {
        // 获取变量大小
        int size = var["size"].as<int>();

        // 处理下界
        const YAML::Node& lower_bound = var["lower_bound"];
        casadi::SX lower_sx = casadi::SX::zeros(size);
        if (lower_bound.IsSequence()) {
            for (size_t i = 0; i < lower_bound.size(); ++i) {
                if (lower_bound[i].IsScalar()) {
                    std::string value = lower_bound[i].as<std::string>();
                    if (value == ".inf" || value == ".Inf" || value == ".INF") {
                        lower_sx(i) = casadi::inf;
                    } else if (value == "-.inf" || value == "-.Inf" || value == "-.INF") {
                        lower_sx(i) = -casadi::inf;
                    } else {
                        // 尝试转换为数值
                        lower_sx(i) = std::stod(value);
                    }
                }
            }
        }
        lowerStatusBound.push_back(lower_sx);

        // 处理上界
        const YAML::Node& upper_bound = var["upper_bound"];
        casadi::SX upper_sx = casadi::SX::zeros(size);
        if (upper_bound.IsSequence()) {
            for (size_t i = 0; i < upper_bound.size(); ++i) {
                if (upper_bound[i].IsScalar()) {
                    auto value = upper_bound[i].as<std::string>();
                    if (value == ".inf" || value == ".Inf" || value == ".INF") {
                        upper_sx(i) = casadi::inf;
                    } else if (value == "-.inf" || value == "-.Inf" || value == "-.INF") {
                        upper_sx(i) = -casadi::inf;
                    } else {
                        // 尝试转换为数值
                        upper_sx(i) = std::stod(value);
                    }
                }
            }
        }
        upperStatusBound.push_back(upper_sx);
    }

    casadi::SXVector lowerInputBound, upperInputBound;
    if (!configNode_["OCP_variables"]) {
        throw std::invalid_argument("node [OCP_variables] not found in YAML file");
    }
    const YAML::Node& input_frame = configNode_["OCP_variables"]["input_frame"];
    initializeFrameWithYAML(inputFrame_, input_frame);
    if (!status_frame.IsSequence()) {
        throw std::invalid_argument("input_frame should be a sequence");
    }
    // 遍历所有状态变量
    for (const auto& var : input_frame) {
        // 获取变量大小
        int size = var["size"].as<int>();

        // 处理下界
        const YAML::Node& lower_bound = var["lower_bound"];
        casadi::SX lower_sx = casadi::SX::zeros(size);
        if (lower_bound.IsSequence()) {
            for (size_t i = 0; i < lower_bound.size(); ++i) {
                if (lower_bound[i].IsScalar()) {
                    auto value = lower_bound[i].as<std::string>();
                    if (value == ".inf" || value == ".Inf" || value == ".INF") {
                        lower_sx(i) = casadi::inf;
                    } else if (value == "-.inf" || value == "-.Inf" || value == "-.INF") {
                        lower_sx(i) = -casadi::inf;
                    } else {
                        // 尝试转换为数值
                        lower_sx(i) = std::stod(value);
                    }
                }
            }
        }
        lowerInputBound.push_back(lower_sx);

        // 处理上界
        const YAML::Node& upper_bound = var["upper_bound"];
        casadi::SX upper_sx = casadi::SX::zeros(size);
        if (upper_bound.IsSequence()) {
            for (size_t i = 0; i < upper_bound.size(); ++i) {
                if (upper_bound[i].IsScalar()) {
                    std::string value = upper_bound[i].as<std::string>();
                    if (value == ".inf" || value == ".Inf" || value == ".INF") {
                        upper_sx(i) = casadi::inf;
                    } else if (value == "-.inf" || value == "-.Inf" || value == "-.INF") {
                        upper_sx(i) = -casadi::inf;
                    } else {
                        // 尝试转换为数值
                        upper_sx(i) = std::stod(value);
                    }
                }
            }
        }
        upperInputBound.push_back(upper_sx);
    }
    coverLowerInputBounds(::casadi::SX::vertcat(lowerInputBound));
    coverUpperInputBounds(::casadi::SX::vertcat(upperInputBound));
    coverLowerStatusBounds(::casadi::SX::vertcat(lowerStatusBound));
    coverUpperStatusBounds(::casadi::SX::vertcat(upperStatusBound));
}

/*
输入的是YAML节点，节点示例如下，必须包含不同的name词条和size词条：
  status_frame:
    - name: "q"
      size: 10
    - name: "dq"
      size: 10
 * */
void OptimalControlProblem::initializeFrameWithYAML(Frame &frame, const YAML::Node &config) {
    frame.totalSize = 0;
    for (const auto &fieldConfig: config) {
        std::string fieldName;
        int fieldSize;
        if(!fieldConfig["name"]){
            throw std::invalid_argument("Field name not found in frame");
        }
        else{
            fieldName = fieldConfig["name"].as<std::string>();
        }
        if(!fieldConfig["size"])
        {
            throw std::invalid_argument("Field size not found in frame");
        }
        else if(fieldConfig["size"].as<int>() <= 0){
            throw std::invalid_argument("Field size must be positive: " + fieldName);
        }else{
            fieldSize = fieldConfig["size"].as<int>();
        }

        if (fieldSize <= 0) {
            throw std::invalid_argument("Field size must be positive: " + fieldName);
        }

        frame.fields.emplace_back(fieldName, fieldSize);
        frame.fieldOffsets[fieldName] = frame.totalSize;
        frame.totalSize += fieldSize;
    }
}

casadi::SX OptimalControlProblem::getStatusVariable(int frameID, const std::string &fieldName) const {
    if (frameID < 0 || frameID >= horizon_) {
        throw std::out_of_range("Frame ID out of range");
    }

    auto it = statusFrame_.fieldOffsets.find(fieldName);
//    end就是没找到，find的话，
    if (it == statusFrame_.fieldOffsets.end()) {
        throw std::invalid_argument("Field name not found in status frame");
    }

    int startIndex = frameID * statusFrame_.totalSize + it->second;
    int fieldSize = 0;
    for (const auto &field: statusFrame_.fields) {
        if (field.first == fieldName) {
            fieldSize = field.second;
            break;
        }
    }

    return statusVariables(casadi::Slice(startIndex, startIndex + fieldSize));
}

casadi::SX OptimalControlProblem::getInputVariable(int frameID, const std::string &fieldName) const {
    if (frameID < 0 || frameID >= horizon_) {
        throw std::out_of_range("Frame ID out of range");
    }

    auto it = inputFrame_.fieldOffsets.find(fieldName);
    if (it == inputFrame_.fieldOffsets.end()) {
        throw std::invalid_argument("Field name not found in input frame");
    }

    int startIndex = frameID * inputFrame_.totalSize + it->second;
    int fieldSize = 0;
    for (const auto &field: inputFrame_.fields) {
        if (field.first == fieldName) {
            fieldSize = field.second;
            break;
        }
    }

    return inputVariables(casadi::Slice(startIndex, startIndex + fieldSize));
}

void OptimalControlProblem::addCost(const casadi::SX &cost) {
    cost_ += cost;
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

void OptimalControlProblem::addEquationConstraint(const std::string &constraintName, const casadi::SX &leftSX,
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

void OptimalControlProblem::saveConstraintsToCSV(const std::string &filename) {
    if (constraintNames_.size() != ::casadi::DM::vertcat(constraintLowerBounds_).size1() ||
        constraintNames_.size() != ::casadi::SX::vertcat(constraints_).size1() ||
        constraintNames_.size() != ::casadi::DM::vertcat(constraintUpperBounds_).size1()) {
        std::cerr << "Error: Inconsistent vector sizes" << std::endl;
        return;
    }
//        准备写入文件
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header
    file << "Constraint Name,Lower Bound,Constraint Expression,Upper Bound\n";

    // Helper function to format DM values
    auto formatDM = [](const ::casadi::DM &dm) -> std::string {
        if (dm.is_scalar()) {
            double value = dm.scalar();
            if (std::isinf(value)) {
                return (value > 0) ? "posInf" : "negInf";
            } else {
                return std::to_string(value);
            }
        } else {
            std::stringstream ss;
            ss << dm;
            std::string str = ss.str();
            // Replace inf and -inf in the string
            size_t pos;
            while ((pos = str.find("inf")) != std::string::npos) {
                if (pos > 0 && str[pos - 1] == '-') {
                    str.replace(pos - 1, 4, "-inf");
                } else {
                    str.replace(pos, 3, "inf");
                }
            }
            return "\"" + str + "\"";
        }
    };

    // Write data
    for (size_t i = 0; i < constraintNames_.size(); ++i) {
        file << constraintNames_[i] << ",";
        // Lower bound
        file << formatDM(::casadi::DM::vertcat(constraintLowerBounds_)(i)) << ",";
        // Constraint expression
        file << "\"" << ::casadi::SX::vertcat(constraints_)(i) << "\",";
        // Upper bound
        file << formatDM(::casadi::DM::vertcat(constraintUpperBounds_)(i));
        file << "\n";
    }
    file.close();
    std::cout << "Constraints saved to " << filename << std::endl;

};

casadi::SX OptimalControlProblem::getStatusVariables() const { return statusVariables; }

casadi::SX OptimalControlProblem::getInputVariables() const { return inputVariables; }

casadi::SX OptimalControlProblem::getCostFunction() const { return cost_; }

std::vector<casadi::SX> OptimalControlProblem::getConstraints() const { return constraints_; }

std::vector<casadi::DM> OptimalControlProblem::getConstraintUpperBounds() const { return constraintUpperBounds_; }

std::vector<casadi::DM> OptimalControlProblem::getConstraintLowerBounds() const { return constraintLowerBounds_; }

int OptimalControlProblem::getHorizon() const { return horizon_; }

int OptimalControlProblem::getStatusFrameSize() const { return statusFrame_.totalSize; }

int OptimalControlProblem::getInputFrameSize() const { return inputFrame_.totalSize; }

std::ostream &operator<<(std::ostream &os, const OptimalControlProblem &ocp) {
    os << "Status Frame:\n";
    for (const auto &field: ocp.statusFrame_.fields) {
        os << field.first << ": " << field.second << "\n";
    }
    os << "Total status frame size: " << ocp.statusFrame_.totalSize << "\n";

    os << "\nInput Frame:\n";
    for (const auto &field: ocp.inputFrame_.fields) {
        os << field.first << ": " << field.second << "\n";
    }
    os << "Total input frame size: " << ocp.inputFrame_.totalSize << "\n";
    return os;
}

std::vector<casadi::DM> OptimalControlProblem::getStatusLowerBounds() const {
    return statusLowerBounds_;
}

std::vector<casadi::DM> OptimalControlProblem::getInputLowerBounds() const {
    return inputLowerBounds_;
}

std::vector<casadi::DM> OptimalControlProblem::getStatusUpperBounds() const {
    return statusUpperBounds_;
}

std::vector<casadi::DM> OptimalControlProblem::getInputUpperBounds() const {
    return inputUpperBounds_;
}

casadi::DM OptimalControlProblem::getVariableLowerBounds() const {
    return ::casadi::DM::vertcat({::casadi::DM::vertcat(statusLowerBounds_), ::casadi::DM::vertcat(inputLowerBounds_)});
}

casadi::DM OptimalControlProblem::getVariableUpperBounds() const {
    return ::casadi::DM::vertcat({::casadi::DM::vertcat(statusUpperBounds_), ::casadi::DM::vertcat(inputUpperBounds_)});
}

float OptimalControlProblem::getDt() const {
    return dt_;
}
/*
 * note : 必需在应用约束和添加损失以后使用！！！！！！！
 * */
void OptimalControlProblem::genSolver() {
    try {
        // 设置求解器选项
        ::casadi::Dict solver_opts;
        solver_opts["print_in"] = 0;
        solver_opts["print_out"] = 0;
        solver_opts["print_time"] = 0;

        // 获取包路径
        try {
            packagePath_ = ament_index_cpp::get_package_share_directory("optimal_control_problem");
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to get package path: " + std::string(e.what()));
        }

        // 构建NLP问题
        ::casadi::SX status_vars = getStatusVariables();
        ::casadi::SX input_vars = getInputVariables();
        if (status_vars.is_empty() || input_vars.is_empty()) {
            throw std::runtime_error("Status or input variables are empty");
        }

        ::casadi::SX constraints = ::casadi::SX::vertcat(getConstraints());
        if (constraints.is_empty()) {
            throw std::runtime_error("Constraints are empty");
        }

        ::casadi::SXDict nlp = {
                {"x", ::casadi::SX::vertcat({status_vars, input_vars})},
                {"f", getCostFunction()},
                {"g", constraints},
                {"p", reference_}
        };

        // 创建求解器
        try {
            IPOPTSolver_ = ::casadi::nlpsol("solver", "ipopt", nlp, solver_opts);
            BlockSQPSolver_ = ::casadi::nlpsol("solver", "blocksqp", nlp, solver_opts);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create solvers: " + std::string(e.what()));
        }

        if (!genCode_) {
            return;
        }

        // 文件路径设置
        const std::string code_dir = packagePath_ + "/code_gen/";
        const std::string IPOPT_solver_file_name = "IPOPT_nlp_code";
        const std::string BlockSQP_solver_file_name = "BlockSQP_nlp_code";
        const std::string IPOPT_solver_source_file = IPOPT_solver_file_name + ".c";
        const std::string BlockSQP_solver_source_file = BlockSQP_solver_file_name + ".c";
        const std::string IPOPT_target_file = code_dir + IPOPT_solver_source_file;
        const std::string BlockSQP_target_file = code_dir + BlockSQP_solver_source_file;
        const std::string IPOPT_shared_lib = code_dir + IPOPT_solver_file_name + ".so";
        const std::string BlockSQP_shared_lib = code_dir + BlockSQP_solver_file_name + ".so";

        // 确保目标目录存在
        if (std::system(("mkdir -p " + code_dir).c_str()) != 0) {
            throw std::runtime_error("Failed to create code generation directory");
        }

        // 生成代码文件
        try {
            IPOPTSolver_.generate_dependencies(IPOPT_solver_source_file);
            BlockSQPSolver_.generate_dependencies(BlockSQP_solver_source_file);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to generate solver dependencies: " + std::string(e.what()));
        }

        // 复制文件
        auto copyFile = [](const std::string& src, const std::string& dst) {
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

        try {
            copyFile(IPOPT_solver_source_file, IPOPT_target_file);
            copyFile(BlockSQP_solver_source_file, BlockSQP_target_file);
        } catch (const std::exception& e) {
            throw std::runtime_error("File copy failed: " + std::string(e.what()));
        }

        // 输出问题规模信息
        if (verbose_) {
            const auto num_vars = ::casadi::SX::vertcat({status_vars, input_vars}).size1();
            const auto num_constraints = constraints.size1();
            const auto num_params = reference_.size1();

            std::cout << "Problem dimensions:\n"
                      << "Variables: " << num_vars << "\n"
                      << "Constraints: " << num_constraints << "\n"
                      << "Parameters: " << num_params << std::endl;
        }

        // 编译共享库
        const std::string compile_flags = "-fPIC -shared -O3";

        // 修改后的编译函数，使用 this 指针访问 verbose_
        auto compileLibrary = [this](const std::string& source_file,
                                     const std::string& output_file,
                                     const std::string& compile_flags) {
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

        try {
            compileLibrary(IPOPT_target_file, IPOPT_shared_lib, compile_flags);
            compileLibrary(BlockSQP_target_file, BlockSQP_shared_lib, compile_flags);
        } catch (const std::exception& e) {
            throw std::runtime_error("Compilation error: " + std::string(e.what()));
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in genSolver: " << e.what() << std::endl;
        throw; // 重新抛出异常以允许上层处理
    }
}


::casadi::SX OptimalControlProblem::getReference() {
    return reference_;
};

/*
 * 输入当前OCP问题系统的状态，以及参考轨迹，计算出最优轨迹
 * bug : 在调用这个函数的时候，发现求解器接收的constraint的大小是不对的
 * */
void OptimalControlProblem::computeOptimalTrajectory(const ::casadi::DM &statusFrame, const ::casadi::DM& reference) {
    if (statusFrame.size1() != statusFrame_.totalSize) {
        std::cerr << "优化问题的状态维度不对，收到状态是" << statusFrame.size1() << "维，期望是" << statusFrame_.totalSize
                  << "维\n";
    }
    if(reference.size1()!= statusFrame_.totalSize){
        std::cerr << "参考轨迹的维度不对，收到参考轨迹是" << reference.size1() << "维，期望是" << reference_.size1()
                  << "维\n";
    }

    // 1. 准备优化器输入参数
    std::map<std::string, ::casadi::DM> arg, res;
    // 设置变量和约束的上下界
    ::casadi::DM lbx = getVariableLowerBounds();
    ::casadi::DM ubx = getVariableUpperBounds();
    lbx(::casadi::Slice(0, statusFrame_.totalSize)) = statusFrame;
    ubx(::casadi::Slice(0, statusFrame_.totalSize)) = statusFrame;

    if (verbose_) {
        std::cout << "变量下界:\n" << lbx << std::endl;
        std::cout << "变量上界:\n" << ubx << std::endl;
    }

    ::casadi::DM lbg = ::casadi::DM::vertcat({getConstraintLowerBounds()});
    ::casadi::DM ubg = ::casadi::DM::vertcat({getConstraintUpperBounds()});

    // 设置初始猜测全都是0
    const int stateSize = getStatusFrameSize();
    const int inputSize = getInputFrameSize();

    ::casadi::DM x0;
    if(setInitialGuess_){
        x0 = initialGuess_;
    } else{
        x0 = ::casadi::DM::repmat(::casadi::DM::zeros(stateSize + inputSize, 1), getHorizon());
    }

    // 组装求解器输入
    arg["lbx"] = lbx;
    arg["ubx"] = ubx;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    if (firstTime_) {
        arg["x0"] = x0;
    } else {
        arg["x0"] = optimalTrajectory_;
    }
    arg["p"] = reference;

    // 2. 求解优化问题
    if (solverInputCheck(arg)) {
        try {
            if (genCode_) {
                if (firstTime_) {
                    saveConstraintsToCSV(packagePath_ + "/log/constraints.csv");
                    // 使用生成的代码求解
                    libIPOPTSolver_ = ::casadi::nlpsol("ipopt_solver", "ipopt", packagePath_ + "/code_gen/IPOPT_nlp_code.so");
                    libBlockSQPSolver_ = ::casadi::nlpsol("snopt_solver", "blocksqp", packagePath_ + "/code_gen/BlockSQP_nlp_code.so");
                    res = libIPOPTSolver_(arg);
                    firstTime_ = false;
                    std::cout<<"暖机完成，已取得当前的全局最优解\n";
                } else {
                    res = libBlockSQPSolver_(arg);
                }
            } else {
                if(firstTime_){
                    // 使用默认求解器
                    res = IPOPTSolver_(arg);
                } else{
                    res = BlockSQPSolver_(arg);
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


bool OptimalControlProblem::solverInputCheck(std::map<std::string, ::casadi::DM> arg) {
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

    int expected_lbx_ubx_x0_size =
            (getStatusFrameSize() + getInputFrameSize()) *
            getHorizon();
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
    int expected_p_size = getStatusFrameSize();
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

void OptimalControlProblem::setStatusBounds(const ::casadi::DM& lowerBound, const ::casadi::DM& upperBound) {
    if(lowerBound.size1()!=statusFrame_.totalSize*horizon_){
        std::cerr << "状态变量的下界维度不对，收到" << lowerBound.size1() << "维，期望" << statusFrame_.totalSize*horizon_ << "维\n";
    }
    if(upperBound.size1()!=statusFrame_.totalSize*horizon_){
        std::cerr << "状态变量的上界维度不对，收到" << upperBound.size1() << "维，期望" << statusFrame_.totalSize*horizon_ << "维\n";
    }
    if(lowerBound.size2()!=1 || upperBound.size2()!=1){
        std::cerr << "状态变量的下界和上界的维度不对，收到" << lowerBound.size2() << "维和" << upperBound.size2() << "维，期望1维\n";
    }
    statusUpperBounds_.clear();
    statusLowerBounds_.clear();
    statusUpperBounds_.emplace_back(upperBound);
    statusLowerBounds_.emplace_back(lowerBound);
}

::casadi::DM OptimalControlProblem::getOptimalInputFirstFrame() {
    // 拿到input的所有帧，再取出第一帧，得到的就是需要发送出去的数据
    ::casadi::DM inputs = optimalTrajectory_(::casadi::Slice(getHorizon() * statusFrame_.totalSize, -1, 1));
//            ::casadi::DM inputFirstFrame = inputs(::casadi::Slice(0, inputFrame_.totalSize, 1));
//            float input = inputFirstFrame(0).scalar();
    std::cout << "输入的所有帧是" << std::endl;
    std::cout<<inputs;
    ::casadi::DM inputFirstFrame = inputs(::casadi::Slice(0, inputFrame_.totalSize, 1));
    std::vector<double> vec;
    vec = inputFirstFrame.get_elements(); // 将所有元素复制到
    std::cout << "输出的第一帧是" << vec << std::endl;
    return ::casadi::DM(vec);
}

/**
 * @brief 设置优化问题的初始猜测解
 * @param initialGuess 包含整个预测时域的状态和输入初始值
 */
void OptimalControlProblem::setInitialGuess(const ::casadi::DM& initialGuess) {
    const int expectedDim = horizon_ * (statusFrame_.totalSize + inputFrame_.totalSize);

    if (initialGuess.size1() != expectedDim) {
        throw std::invalid_argument(
                "初始猜测维度错误: 输入" + std::to_string(initialGuess.size1()) +
                "维, 期望" + std::to_string(expectedDim) + "维"
        );
    }
    initialGuess_ = initialGuess;
}

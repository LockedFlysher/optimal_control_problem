//
// Created by lock on 25-3-7.
//

#include "optimal_control_problem/OCP_config/OCPConfig.h"

casadi::SX OCPConfig::getVariable(int stepID, const std::string &variableName) const {
    if (stepID < 0 || stepID >= horizon_) {
        throw std::out_of_range("Frame ID out of range");
    }
    auto it = variableFrame_.fieldOffsets.find(variableName);
    if (it == variableFrame_.fieldOffsets.end()) {
        throw std::invalid_argument("Field name not found in frame");
    }
    int startIndex = stepID * variableFrame_.totalSize + it->second;
    int fieldSize = 0;
    for (const auto &field: variableFrame_.fields) {
        if (field.first == variableName) {
            fieldSize = field.second;
            break;
        }
    }
    return variables_(casadi::Slice(startIndex, startIndex + fieldSize));
}

/*
输入的是YAML节点，节点示例如下，必须包含不同的name词条和size词条：
  status_frame:
    - name: "q"
      size: 10
    - name: "dq"
      size: 10
 * */
void OCPConfig::initializeFrame(Frame &frame, const YAML::Node &config) {
    frame.totalSize = 0;
    for (const auto &fieldConfig: config) {
        std::string fieldName;
        int fieldSize;
        if (!fieldConfig["name"]) {
            throw std::invalid_argument("Field name not found in frame");
        } else {
            fieldName = fieldConfig["name"].as<std::string>();
        }
        if (!fieldConfig["size"]) {
            throw std::invalid_argument("Field size not found in frame");
        } else if (fieldConfig["size"].as<int>() <= 0) {
            throw std::invalid_argument("Field size must be positive: " + fieldName);
        } else {
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

OCPConfig::OCPConfig(YAML::Node configNode) {
    // 初始化默认值
    horizon_ = 10; // 默认值
    dt_ = 0.1;     // 默认值
    verbose_ = false;
    // 解析问题配置
    OCP_INFO("优化问题配置解析");
    this->dt_ = configNode["discretization_settings"]["dt"].as<double>();
    OCP_INFO("dt: " + std::to_string(this->dt_));
    this->horizon_ = configNode["discretization_settings"]["horizon"].as<int>();
    OCP_INFO("horizon: " + std::to_string(this->horizon_));
    this->verbose_ = configNode["solver_settings"]["verbose"].as<bool>();
    // 解析边界条件
    OCP_INFO("开始解析边界条件...");
    parseOCPBounds(configNode);
    OCP_INFO("边界条件解析完成");
    // 创建变量
    OCP_INFO("创建状态和输入变量");
    OCP_INFO("帧大小: " + std::to_string(variableFrame_.totalSize));
    variables_ = casadi::SX::sym("X", horizon_ * variableFrame_.totalSize, 1);
    OCP_INFO("OCPConfig初始化完成");
    printSummary();
}


void OCPConfig::parseOCPBounds(YAML::Node configNode) {
    try {
        OCP_INFO("开始解析OCP边界条件...");

        casadi::SXVector lowerBound, upperBound;
        // 检查OCP_variables节点是否存在
        if (!configNode["OCP_variables"]) {
            OCP_ERROR("找不到OCP_variables节点");
            throw std::invalid_argument("node [OCP_variables] not found in YAML file");
        }
        OCP_INFO("找到OCP_variables节点");

        const YAML::Node &frame = configNode["OCP_variables"];
        if (!frame.IsSequence()) {
            OCP_ERROR("OCP_variables不是序列类型");
            throw std::invalid_argument("status_frame should be a sequence");
        }
        OCP_INFO("初始化帧...");
        initializeFrame(variableFrame_, frame);
        OCP_INFO("状态帧初始化完成，总大小: " + std::to_string(variableFrame_.totalSize));
        // 遍历所有状态变量
        OCP_INFO("开始解析状态变量边界...");
        for (size_t varIdx = 0; varIdx < frame.size(); ++varIdx) {
            const auto &var = frame[varIdx];
            std::string varName = var["name"].as<std::string>();
            int size = var["size"].as<int>();

            OCP_INFO("处理状态变量: " + varName + ", 大小: " + std::to_string(size));

            // 处理下界
            if (!var["lower_bound"]) {
                OCP_ERROR("变量 " + varName + " 缺少lower_bound节点");
                throw std::invalid_argument("Missing lower_bound for variable: " + varName);
            }

            const YAML::Node &lower_bound = var["lower_bound"];
            casadi::SX lower_sx = casadi::SX::zeros(size);

            if (lower_bound.IsSequence()) {
                if (lower_bound.size() != size) {
                    OCP_WARN("变量 " + varName + " 的lower_bound大小 (" +
                             std::to_string(lower_bound.size()) + ") 与变量大小 (" +
                             std::to_string(size) + ") 不匹配");
                }

                OCP_DEBUG("解析lower_bound值...");
                for (size_t i = 0; i < lower_bound.size() && i < size; ++i) {
                    if (lower_bound[i].IsScalar()) {
                        auto value = lower_bound[i].as<std::string>();
                        if (value == ".inf" || value == ".Inf" || value == ".INF") {
                            lower_sx(i) = casadi::inf;
                            OCP_DEBUG("  索引 " + std::to_string(i) + ": 正无穷");
                        } else if (value == "-.inf" || value == "-.Inf" || value == "-.INF") {
                            lower_sx(i) = -casadi::inf;
                            OCP_DEBUG("  索引 " + std::to_string(i) + ": 负无穷");
                        } else {
                            try {
                                lower_sx(i) = std::stod(value);
                                OCP_DEBUG("  索引 " + std::to_string(i) + ": " + value);
                            } catch (const std::exception &e) {
                                OCP_ERROR("  无法将值 '" + value + "' 转换为数值: " + e.what());
                                throw;
                            }
                        }
                    } else {
                        OCP_ERROR("  索引 " + std::to_string(i) + " 不是标量值");
                    }
                }
            } else {
                OCP_WARN("变量 " + varName + " 的lower_bound不是序列类型");
            }
            lowerBound.emplace_back(lower_sx);
            // 处理上界
            if (!var["upper_bound"]) {
                OCP_ERROR("变量 " + varName + " 缺少upper_bound节点");
                throw std::invalid_argument("Missing upper_bound for variable: " + varName);
            }
            const YAML::Node &upper_bound = var["upper_bound"];
            casadi::SX upper_sx = casadi::SX::zeros(size);
            if (upper_bound.IsSequence()) {
                if (upper_bound.size() != size) {
                    OCP_WARN("变量 " + varName + " 的upper_bound大小 (" +
                             std::to_string(upper_bound.size()) + ") 与变量大小 (" +
                             std::to_string(size) + ") 不匹配");
                }
                OCP_DEBUG("解析upper_bound值...");
                for (size_t i = 0; i < upper_bound.size() && i < size; ++i) {
                    if (upper_bound[i].IsScalar()) {
                        auto value = upper_bound[i].as<std::string>();
                        if (value == ".inf" || value == ".Inf" || value == ".INF") {
                            upper_sx(i) = casadi::inf;
                            OCP_DEBUG("  索引 " + std::to_string(i) + ": 正无穷");
                        } else if (value == "-.inf" || value == "-.Inf" || value == "-.INF") {
                            upper_sx(i) = -casadi::inf;
                            OCP_DEBUG("  索引 " + std::to_string(i) + ": 负无穷");
                        } else {
                            try {
                                upper_sx(i) = std::stod(value);
                                OCP_DEBUG("  索引 " + std::to_string(i) + ": " + value);
                            } catch (const std::exception &e) {
                                OCP_ERROR("  无法将值 '" + value + "' 转换为数值: " + e.what());
                                throw;
                            }
                        }
                    } else {
                        OCP_ERROR("  索引 " + std::to_string(i) + " 不是标量值");
                    }
                }
            } else {
                OCP_WARN("变量 " + varName + " 的upper_bound不是序列类型");
            }
            upperBound.emplace_back(upper_sx);
        }

        // 检查边界向量是否为空
        if (lowerBound.empty()) {
            OCP_ERROR("变量下界向量为空");
        }
        if (upperBound.empty()) {
            OCP_ERROR("变量上界向量为空");
        }

        // 设置边界
        OCP_INFO("合并下界和上界...");
        casadi::SX lowerBoundCombined = ::casadi::SX::vertcat(lowerBound);
        casadi::SX upperBoundCombined = ::casadi::SX::vertcat(upperBound);

        OCP_INFO("变量下界大小: " + std::to_string(lowerBoundCombined.size1()) + "x" +
                 std::to_string(lowerBoundCombined.size2()));
        OCP_INFO("变量上界大小: " + std::to_string(upperBoundCombined.size1()) + "x" +
                 std::to_string(upperBoundCombined.size2()));
        OCP_INFO("设置边界...");
        coverLowerBounds(lowerBoundCombined);
        coverUpperBounds(upperBoundCombined);
        OCP_INFO("边界设置完成");
        // 打印边界向量大小
        OCP_INFO("变量下界向量大小: " + std::to_string(lowerBounds_.size()));
        OCP_INFO("变量上界向量大小: " + std::to_string(upperBounds_.size()));
    } catch (const std::exception &e) {
        OCP_ERROR("解析OCP边界条件失败: " + std::string(e.what()));
        throw;
    }
}

void OCPConfig::coverLowerBounds(const casadi::SX &oneFrameLowerBound) {
    OCP_INFO("设置变量上界...");
    OCP_INFO("  一帧下界大小: " + std::to_string(oneFrameLowerBound.size1()) + "x" +
             std::to_string(oneFrameLowerBound.size2()));
    OCP_INFO("  horizon: " + std::to_string(horizon_));
    lowerBounds_.clear();

    try {
        for (int i = 0; i < horizon_; ++i) {
            casadi::DM dmBound = casadi::DM(oneFrameLowerBound);
            lowerBounds_.emplace_back(dmBound);
        }
        OCP_INFO("变量下界设置完成，大小: " + std::to_string(lowerBounds_.size()));
    } catch (const std::exception &e) {
        OCP_ERROR("设置变量下界失败: " + std::string(e.what()));
        throw;
    }
}

void OCPConfig::coverUpperBounds(const casadi::SX &oneFrameUpperBound) {
    OCP_INFO("设置状态变量上界...");
    OCP_INFO("  一帧上界大小: " + std::to_string(oneFrameUpperBound.size1()) + "x" +
             std::to_string(oneFrameUpperBound.size2()));
    OCP_INFO("  horizon: " + std::to_string(horizon_));
    upperBounds_.clear();
    try {
        for (int i = 0; i < horizon_; ++i) {
            casadi::DM dmBound = casadi::DM(oneFrameUpperBound);
            upperBounds_.emplace_back(dmBound);
        }
        OCP_INFO("状态变量上界设置完成，大小: " + std::to_string(upperBounds_.size()));
    } catch (const std::exception &e) {
        OCP_ERROR("设置状态变量上界失败: " + std::string(e.what()));
        throw;
    }
}


std::vector<casadi::DM> OCPConfig::getLowerBounds() const {
    if (lowerBounds_.empty()) {
        OCP_WARN("获取状态变量下界时，边界向量为空");
    }
    return lowerBounds_;
}

//std::vector<casadi::DM> OCPConfig::getInputLowerBounds() const {
//    if (inputLowerBounds_.empty()) {
//        OCP_WARN("获取输入变量下界时，边界向量为空");
//    }
//    return inputLowerBounds_;
//}

std::vector<casadi::DM> OCPConfig::getUpperBounds() const {
    if (upperBounds_.empty()) {
        OCP_WARN("获取状态变量上界时，边界向量为空");
    }
    return upperBounds_;
}

//std::vector<casadi::DM> OCPConfig::getInputUpperBounds() const {
//    if (inputUpperBounds_.empty()) {
//        OCP_WARN("获取输入变量上界时，边界向量为空");
//    }
//    return inputUpperBounds_;
//}


casadi::SX OCPConfig::getVariables() const {
    return variables_;
}

//casadi::SX OCPConfig::getInputVariables() const {
//    return inputVariables_;
//}

int OCPConfig::getHorizon() const {
    return horizon_;
}

int OCPConfig::getFrameSize() const {
    return variableFrame_.totalSize;
}

/**
 * @brief 设置优化问题的初始猜测解
 * @param initialGuess 包含整个预测时域的状态和输入初始值
 */
void OCPConfig::setInitialGuess(const ::casadi::DM &initialGuess) {
    const int expectedDim = horizon_ * (variableFrame_.totalSize);

    if (initialGuess.size1() != expectedDim) {
        throw std::invalid_argument(
                "初始猜测维度错误: 输入" + std::to_string(initialGuess.size1()) +
                "维, 期望" + std::to_string(expectedDim) + "维"
        );
    }
    initialGuess_ = initialGuess;
}

double OCPConfig::getDt() const {
    return this->dt_;
}

::casadi::DM OCPConfig::getInitialGuess() {
    return initialGuess_;
}

/**
 * Prints a formatted summary of the OCP configuration
 * Displays horizon, time step, variable structure, and bounds in a tabular format
 */
/**
 * Prints a formatted summary of the OCP configuration
 * Displays horizon, time step, variable structure, and bounds in a tabular format
 */
void OCPConfig::printSummary() const {
    // Print header
    std::cout << "\n=============== OCP Configuration Summary ===============" << std::endl;

    // Print global settings
    std::cout << "Global Settings:" << std::endl;
    std::cout << "  Time Step (dt): " << std::fixed << std::setprecision(6) << dt_ << std::endl;
    std::cout << "  Horizon (N):    " << horizon_ << std::endl;

    // Print frame information
    std::cout << "\nFrame Definition (Variables per Time Step):" << std::endl;
    std::cout << "  Total Variables in Frame: " << variableFrame_.totalSize << std::endl;

    // Table header
    std::cout << "\n+-----------------+------+--------------------------+--------------------------+" << std::endl;
    std::cout << "| Variable Name   | Size | Lower Bounds (per step)  | Upper Bounds (per step)  |" << std::endl;
    std::cout << "+-----------------+------+--------------------------+--------------------------+" << std::endl;

    // Table rows
    for (const auto& field : variableFrame_.fields) {
        const std::string& name = field.first;
        int size = field.second;
        int offset = variableFrame_.fieldOffsets.at(name);

        // Variable name column (fixed width 15)
        std::cout << "| " << std::left << std::setw(15) << name << " | ";

        // Size column (fixed width 4)
        std::cout << std::right << std::setw(4) << size << " | ";

        // Lower bounds column (fixed width 24)
        std::cout << std::left << std::setw(24);
        std::stringstream lbStream;
        lbStream << "[";
        if (!lowerBounds_.empty()) {
            for (int i = 0; i < size; ++i) {
                if (i > 0) lbStream << ", ";
                lbStream << std::fixed << std::setprecision(1) << lowerBounds_[0](offset + i).scalar();
            }
        } else {
            lbStream << "Not set";
        }
        lbStream << "]";
        std::cout << lbStream.str() << " | ";

        // Upper bounds column (fixed width 24)
        std::cout << std::left << std::setw(24);
        std::stringstream ubStream;
        ubStream << "[";
        if (!upperBounds_.empty()) {
            for (int i = 0; i < size; ++i) {
                if (i > 0) ubStream << ", ";
                ubStream << std::fixed << std::setprecision(1) << upperBounds_[0](offset + i).scalar();
            }
        } else {
            ubStream << "Not set";
        }
        ubStream << "]";
        std::cout << ubStream.str() << " |" << std::endl;
    }

    // Table footer
    std::cout << "+-----------------+------+--------------------------+--------------------------+" << std::endl;

    // Total variables
    std::cout << "\nTotal Optimization Variables:" << std::endl;
    std::cout << "  Across Horizon (Frame Size * N): " << (variableFrame_.totalSize * horizon_)
              << " (" << variableFrame_.totalSize << " variables/step * " << horizon_ << " steps)" << std::endl;

    // Bounds application explanation
    std::cout << "\nBounds Application:" << std::endl;
    std::cout << "  The 'Lower Bounds (per step)' and 'Upper Bounds (per step)' listed in" << std::endl;
    std::cout << "  the table are applied at each of the " << horizon_ << " time steps within the horizon." << std::endl;
    std::cout << "  The OCP solver will receive " << horizon_ << " sets of these "
              << variableFrame_.totalSize << "x1 bound vectors." << std::endl;
}

//
// Created by lock on 25-3-7.
//

#include "optimal_control_problem/OCP_config/OCPConfig.h"

casadi::SX OCPConfig::getInputVariable(int frameID, const std::string &fieldName) const {
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

    return inputVariables_(casadi::Slice(startIndex, startIndex + fieldSize));
}

casadi::SX OCPConfig::getStatusVariable(int stepID, const std::string &variableName) const {
    if (stepID < 0 || stepID >= horizon_) {
        throw std::out_of_range("Frame ID out of range");
    }
    auto it = statusFrame_.fieldOffsets.find(variableName);
    if (it == statusFrame_.fieldOffsets.end()) {
        throw std::invalid_argument("Field name not found in status frame");
    }
    int startIndex = stepID * statusFrame_.totalSize + it->second;
    int fieldSize = 0;
    for (const auto &field: statusFrame_.fields) {
        if (field.first == variableName) {
            fieldSize = field.second;
            break;
        }
    }
    return statusVariables_(casadi::Slice(startIndex, startIndex + fieldSize));
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
OCPConfig::OCPConfig() {
    try {
        OCP_INFO("开始初始化OCPConfig...");
        std::string configPath = ament_index_cpp::get_package_share_directory("optimal_control_problem") + "/config/OCP_config.yaml";
        OCP_INFO("加载配置文件: " + configPath);

        auto configNode = YAML::LoadFile(configPath);
        OCP_INFO("配置文件加载成功");

        // 初始化默认值
        horizon_ = 10; // 默认值
        dt_ = 0.1;     // 默认值
        problemName_ = "default";
        verbose_ = false;

        // 解析问题配置
        if (configNode["problem"]) {
            OCP_INFO("解析problem节点");
            if (configNode["problem"]["name"]) {
                this->problemName_ = configNode["problem"]["name"].as<std::string>();
                OCP_INFO("问题名称: " + this->problemName_);
            }
            if (configNode["problem"]["dt"]) {
                this->dt_ = configNode["problem"]["dt"].as<double>();
                if (this->dt_ <= 0) {
                    OCP_ERROR("dt值无效: " + std::to_string(this->dt_));
                    throw std::invalid_argument("dt不应该小于或等于0");
                }
                OCP_INFO("dt: " + std::to_string(this->dt_));
            }
            if (configNode["problem"]["horizon"]) {
                this->horizon_ = configNode["problem"]["horizon"].as<int>();
                if (horizon_ <= 1) {
                    OCP_ERROR("horizon值无效: " + std::to_string(this->horizon_));
                    throw std::invalid_argument("horizon不应该小于或等于1");
                }
                OCP_INFO("horizon: " + std::to_string(this->horizon_));
            }
            if (configNode["problem"]["verbose"]) {
                this->verbose_ = configNode["problem"]["verbose"].as<bool>();
                OCP_INFO("verbose: " + std::string(this->verbose_ ? "true" : "false"));
            }
        } else {
            OCP_WARN("未找到problem节点，使用默认值");
        }

        // 解析边界条件
        OCP_INFO("开始解析边界条件...");
        parseOCPBounds(configNode);
        OCP_INFO("边界条件解析完成");

        // 创建变量
        OCP_INFO("创建状态和输入变量");
        OCP_INFO("状态帧大小: " + std::to_string(statusFrame_.totalSize));
        OCP_INFO("输入帧大小: " + std::to_string(inputFrame_.totalSize));

        statusVariables_ = casadi::SX::sym("X", horizon_ * statusFrame_.totalSize, 1);
        inputVariables_ = casadi::SX::sym("U", horizon_ * inputFrame_.totalSize, 1);

        OCP_INFO("OCPConfig初始化完成");
    } catch (const std::exception& e) {
        OCP_ERROR("OCPConfig初始化失败: " + std::string(e.what()));
        throw;
    }
}



void OCPConfig::parseOCPBounds(YAML::Node configNode) {
    try {
        OCP_INFO("开始解析OCP边界条件...");

        casadi::SXVector lowerStatusBound, upperStatusBound;
        casadi::SXVector lowerInputBound, upperInputBound;

        // 检查OCP_variables节点是否存在
        if (!configNode["OCP_variables"]) {
            OCP_ERROR("找不到OCP_variables节点");
            throw std::invalid_argument("node [OCP_variables] not found in YAML file");
        }
        OCP_INFO("找到OCP_variables节点");

        // 检查status_frame节点是否存在
        if (!configNode["OCP_variables"]["status_frame"]) {
            OCP_ERROR("找不到status_frame节点");
            throw std::invalid_argument("node [status_frame] not found in YAML file");
        }

        const YAML::Node &status_frame = configNode["OCP_variables"]["status_frame"];
        if (!status_frame.IsSequence()) {
            OCP_ERROR("status_frame不是序列类型");
            throw std::invalid_argument("status_frame should be a sequence");
        }

        OCP_INFO("初始化状态帧...");
        initializeFrame(statusFrame_, status_frame);
        OCP_INFO("状态帧初始化完成，总大小: " + std::to_string(statusFrame_.totalSize));

        // 遍历所有状态变量
        OCP_INFO("开始解析状态变量边界...");
        for (size_t varIdx = 0; varIdx < status_frame.size(); ++varIdx) {
            const auto &var = status_frame[varIdx];
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
                            } catch (const std::exception& e) {
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

            lowerStatusBound.emplace_back(lower_sx);

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
                            } catch (const std::exception& e) {
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

            upperStatusBound.emplace_back(upper_sx);
        }

        // 检查输入变量节点是否存在
        if (!configNode["OCP_variables"]["input_frame"]) {
            OCP_ERROR("找不到input_frame节点");
            throw std::invalid_argument("node [input_frame] not found in YAML file");
        }

        const YAML::Node &input_frame = configNode["OCP_variables"]["input_frame"];
        if (!input_frame.IsSequence()) {
            OCP_ERROR("input_frame不是序列类型");
            throw std::invalid_argument("input_frame should be a sequence");
        }

        OCP_INFO("初始化输入帧...");
        initializeFrame(inputFrame_, input_frame);
        OCP_INFO("输入帧初始化完成，总大小: " + std::to_string(inputFrame_.totalSize));

        // 遍历所有输入变量
        OCP_INFO("开始解析输入变量边界...");
        for (size_t varIdx = 0; varIdx < input_frame.size(); ++varIdx) {
            const auto &var = input_frame[varIdx];
            std::string varName = var["name"].as<std::string>();
            int size = var["size"].as<int>();

            OCP_INFO("处理输入变量: " + varName + ", 大小: " + std::to_string(size));

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
                            } catch (const std::exception& e) {
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

            lowerInputBound.push_back(lower_sx);

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
                            } catch (const std::exception& e) {
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

            upperInputBound.push_back(upper_sx);
        }

        // 检查边界向量是否为空
        if (lowerStatusBound.empty()) {
            OCP_ERROR("状态变量下界向量为空");
        }
        if (upperStatusBound.empty()) {
            OCP_ERROR("状态变量上界向量为空");
        }
        if (lowerInputBound.empty()) {
            OCP_ERROR("输入变量下界向量为空");
        }
        if (upperInputBound.empty()) {
            OCP_ERROR("输入变量上界向量为空");
        }

        // 设置边界
        OCP_INFO("合并下界和上界...");
        casadi::SX lowerStatusBoundCombined = ::casadi::SX::vertcat(lowerStatusBound);
        casadi::SX upperStatusBoundCombined = ::casadi::SX::vertcat(upperStatusBound);
        casadi::SX lowerInputBoundCombined = ::casadi::SX::vertcat(lowerInputBound);
        casadi::SX upperInputBoundCombined = ::casadi::SX::vertcat(upperInputBound);

        OCP_INFO("状态变量下界大小: " + std::to_string(lowerStatusBoundCombined.size1()) + "x" +
                 std::to_string(lowerStatusBoundCombined.size2()));
        OCP_INFO("状态变量上界大小: " + std::to_string(upperStatusBoundCombined.size1()) + "x" +
                 std::to_string(upperStatusBoundCombined.size2()));
        OCP_INFO("输入变量下界大小: " + std::to_string(lowerInputBoundCombined.size1()) + "x" +
                 std::to_string(lowerInputBoundCombined.size2()));
        OCP_INFO("输入变量上界大小: " + std::to_string(upperInputBoundCombined.size1()) + "x" +
                 std::to_string(upperInputBoundCombined.size2()));

        OCP_INFO("设置边界...");
        coverLowerInputBounds(lowerInputBoundCombined);
        coverUpperInputBounds(upperInputBoundCombined);
        coverLowerStatusBounds(lowerStatusBoundCombined);
        coverUpperStatusBounds(upperStatusBoundCombined);

        OCP_INFO("边界设置完成");

        // 打印边界向量大小
        OCP_INFO("状态变量下界向量大小: " + std::to_string(statusLowerBounds_.size()));
        OCP_INFO("状态变量上界向量大小: " + std::to_string(statusUpperBounds_.size()));
        OCP_INFO("输入变量下界向量大小: " + std::to_string(inputLowerBounds_.size()));
        OCP_INFO("输入变量上界向量大小: " + std::to_string(inputUpperBounds_.size()));

    } catch (const std::exception& e) {
        OCP_ERROR("解析OCP边界条件失败: " + std::string(e.what()));
        throw;
    }
}

//    使用一帧的下界完成所有变量的下界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的下界添加到整个状态变量的下界中
void OCPConfig::coverLowerStatusBounds(const casadi::SX &oneFrameLowerBound) {
    OCP_INFO("设置状态变量下界...");
    OCP_INFO("  一帧下界大小: " + std::to_string(oneFrameLowerBound.size1()) + "x" +
             std::to_string(oneFrameLowerBound.size2()));
    OCP_INFO("  horizon: " + std::to_string(horizon_));

    statusLowerBounds_.clear();

    try {
        for (int i = 0; i < horizon_; ++i) {
            // 将SX转换为DM
            casadi::DM dmBound = casadi::DM(oneFrameLowerBound);
            statusLowerBounds_.emplace_back(dmBound);
        }
        OCP_INFO("状态变量下界设置完成，大小: " + std::to_string(statusLowerBounds_.size()));
    } catch (const std::exception& e) {
        OCP_ERROR("设置状态变量下界失败: " + std::string(e.what()));
        throw;
    }
}

void OCPConfig::coverUpperStatusBounds(const casadi::SX &oneFrameUpperBound) {
    OCP_INFO("设置状态变量上界...");
    OCP_INFO("  一帧上界大小: " + std::to_string(oneFrameUpperBound.size1()) + "x" +
             std::to_string(oneFrameUpperBound.size2()));
    OCP_INFO("  horizon: " + std::to_string(horizon_));

    statusUpperBounds_.clear();

    try {
        for (int i = 0; i < horizon_; ++i) {
            casadi::DM dmBound = casadi::DM(oneFrameUpperBound);
            statusUpperBounds_.emplace_back(dmBound);
        }
        OCP_INFO("状态变量上界设置完成，大小: " + std::to_string(statusUpperBounds_.size()));
    } catch (const std::exception& e) {
        OCP_ERROR("设置状态变量上界失败: " + std::string(e.what()));
        throw;
    }
}

void OCPConfig::coverLowerInputBounds(const casadi::SX &oneFrameLowerBound) {
    OCP_INFO("设置输入变量下界...");
    OCP_INFO("  一帧下界大小: " + std::to_string(oneFrameLowerBound.size1()) + "x" +
             std::to_string(oneFrameLowerBound.size2()));
    OCP_INFO("  horizon: " + std::to_string(horizon_));

    inputLowerBounds_.clear();

    try {
        for (int i = 0; i < horizon_; ++i) {
            casadi::DM dmBound = casadi::DM(oneFrameLowerBound);
            inputLowerBounds_.emplace_back(dmBound);
        }
        OCP_INFO("输入变量下界设置完成，大小: " + std::to_string(inputLowerBounds_.size()));
    } catch (const std::exception& e) {
        OCP_ERROR("设置输入变量下界失败: " + std::string(e.what()));
        throw;
    }
}

void OCPConfig::coverUpperInputBounds(const casadi::SX &oneFrameUpperBound) {
    OCP_INFO("设置输入变量上界...");
    OCP_INFO("  一帧上界大小: " + std::to_string(oneFrameUpperBound.size1()) + "x" +
             std::to_string(oneFrameUpperBound.size2()));
    OCP_INFO("  horizon: " + std::to_string(horizon_));

    inputUpperBounds_.clear();

    try {
        for (int i = 0; i < horizon_; ++i) {
            casadi::DM dmBound = casadi::DM(oneFrameUpperBound);
            inputUpperBounds_.emplace_back(dmBound);
        }
        OCP_INFO("输入变量上界设置完成，大小: " + std::to_string(inputUpperBounds_.size()));
    } catch (const std::exception& e) {
        OCP_ERROR("设置输入变量上界失败: " + std::string(e.what()));
        throw;
    }
}


std::vector<casadi::DM> OCPConfig::getStatusLowerBounds() const {
    if (statusLowerBounds_.empty()) {
        OCP_WARN("获取状态变量下界时，边界向量为空");
    }
    return statusLowerBounds_;
}

std::vector<casadi::DM> OCPConfig::getInputLowerBounds() const {
    if (inputLowerBounds_.empty()) {
        OCP_WARN("获取输入变量下界时，边界向量为空");
    }
    return inputLowerBounds_;
}

std::vector<casadi::DM> OCPConfig::getStatusUpperBounds() const {
    if (statusUpperBounds_.empty()) {
        OCP_WARN("获取状态变量上界时，边界向量为空");
    }
    return statusUpperBounds_;
}

std::vector<casadi::DM> OCPConfig::getInputUpperBounds() const {
    if (inputUpperBounds_.empty()) {
        OCP_WARN("获取输入变量上界时，边界向量为空");
    }
    return inputUpperBounds_;
}


casadi::SX OCPConfig::getStatusVariables() const {
    return statusVariables_;
}

casadi::SX OCPConfig::getInputVariables() const {
    return inputVariables_;
}

int OCPConfig::getHorizon() const {
    return horizon_;
}

int OCPConfig::getStatusFrameSize() const {
    return statusFrame_.totalSize;
}

int OCPConfig::getInputFrameSize() const {
    return inputFrame_.totalSize;
}

void OCPConfig::setStatusBounds(const casadi::DM &lowerBound, const casadi::DM &upperBound) {
    if (lowerBound.size1() != statusFrame_.totalSize * horizon_) {
        std::cerr << "状态变量的下界维度不对，收到" << lowerBound.size1() << "维，期望"
                  << statusFrame_.totalSize * horizon_ << "维\n";
    }
    if (upperBound.size1() != statusFrame_.totalSize * horizon_) {
        std::cerr << "状态变量的上界维度不对，收到" << upperBound.size1() << "维，期望"
                  << statusFrame_.totalSize * horizon_ << "维\n";
    }
    if (lowerBound.size2() != 1 || upperBound.size2() != 1) {
        std::cerr << "状态变量的下界和上界的维度不对，收到" << lowerBound.size2() << "维和" << upperBound.size2()
                  << "维，期望1维\n";
    }
    statusUpperBounds_.clear();
    statusLowerBounds_.clear();
    statusUpperBounds_.emplace_back(upperBound);
    statusLowerBounds_.emplace_back(lowerBound);
}

/**
 * @brief 设置优化问题的初始猜测解
 * @param initialGuess 包含整个预测时域的状态和输入初始值
 */
void OCPConfig::setInitialGuess(const ::casadi::DM &initialGuess) {
    const int expectedDim = horizon_ * (statusFrame_.totalSize + inputFrame_.totalSize);

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

casadi::DM OCPConfig::getVariableLowerBounds() const {
    return ::casadi::DM::vertcat({::casadi::DM::vertcat(statusLowerBounds_), ::casadi::DM::vertcat(inputLowerBounds_)});
}

casadi::DM OCPConfig::getVariableUpperBounds() const {
    return ::casadi::DM::vertcat({::casadi::DM::vertcat(statusUpperBounds_), ::casadi::DM::vertcat(inputUpperBounds_)});
}
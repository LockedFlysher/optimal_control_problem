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
    auto configNode = YAML::LoadFile(
            ament_index_cpp::get_package_share_directory("optimal_control_problem") + "/config/OCP_config.yaml");
    parseOCPBounds(configNode);
    if (configNode["problem"]) {
        if (configNode["problem"]["name"]) {
            this->problemName_ = configNode["problem"]["name"].as<std::string>();
        }
        if (configNode["problem"]["dt"]) {
            this->dt_ = configNode["problem"]["dt"].as<double>();
            if (this->dt_ <= 0) {
                std::cout << "dt不应该小于0\n";
                exit(1);
            }
        }
        if (configNode["problem"]["horizon"]) {
            this->horizon_ = configNode["problem"]["horizon"].as<int>();
            if (horizon_ <= 1) {
                std::cout << "horizon不应该小于2\n";
                exit(1);
            }
        }
        if (configNode["problem"]["verbose"]) {
            this->verbose_ = configNode["problem"]["verbose"].as<bool>();
        }
    }
    statusVariables_ = casadi::SX::sym("X", horizon_ * statusFrame_.totalSize, 1);
    inputVariables_ = casadi::SX::sym("U", horizon_ * inputFrame_.totalSize, 1);
}


void OCPConfig::parseOCPBounds(YAML::Node configNode) {
    casadi::SXVector lowerStatusBound, upperStatusBound;
    if (!configNode["OCP_variables"]) {
        throw std::invalid_argument("node [OCP_variables] not found in YAML file");
    }
    const YAML::Node &status_frame = configNode["OCP_variables"]["status_frame"];
    initializeFrame(statusFrame_, status_frame);
    if (!status_frame.IsSequence()) {
        throw std::invalid_argument("status_frame should be a sequence");
    }
    // 遍历所有状态变量
    for (const auto &var: status_frame) {
        // 获取变量大小
        int size = var["size"].as<int>();

        // 处理下界
        const YAML::Node &lower_bound = var["lower_bound"];
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
        lowerStatusBound.push_back(lower_sx);

        // 处理上界
        const YAML::Node &upper_bound = var["upper_bound"];
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
    if (!configNode["OCP_variables"]) {
        throw std::invalid_argument("node [OCP_variables] not found in YAML file");
    }
    const YAML::Node &input_frame = configNode["OCP_variables"]["input_frame"];
    initializeFrame(inputFrame_, input_frame);
    if (!status_frame.IsSequence()) {
        throw std::invalid_argument("input_frame should be a sequence");
    }
    // 遍历所有状态变量
    for (const auto &var: input_frame) {
        // 获取变量大小
        int size = var["size"].as<int>();

        // 处理下界
        const YAML::Node &lower_bound = var["lower_bound"];
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
        const YAML::Node &upper_bound = var["upper_bound"];
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
        upperInputBound.push_back(upper_sx);
    }
    coverLowerInputBounds(::casadi::SX::vertcat(lowerInputBound));
    coverUpperInputBounds(::casadi::SX::vertcat(upperInputBound));
    coverLowerStatusBounds(::casadi::SX::vertcat(lowerStatusBound));
    coverUpperStatusBounds(::casadi::SX::vertcat(upperStatusBound));
}

//    使用一帧的下界完成所有变量的下界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的下界添加到整个状态变量的下界中
void OCPConfig::coverLowerStatusBounds(const casadi::SX &oneFrameLowerBound) {
    statusLowerBounds_.clear();
    for (int i = 0; i < horizon_; ++i) {
        statusLowerBounds_.emplace_back(oneFrameLowerBound);
    }
}

//    使用一帧的上界完成所有变量的上界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的上界添加到整个状态变量的上界中
void OCPConfig::coverUpperStatusBounds(const casadi::SX &oneFrameUpperBound) {
    statusUpperBounds_.clear();
    for (int i = 0; i < horizon_; ++i) {
        statusUpperBounds_.emplace_back(oneFrameUpperBound);
    }
}

//    使用一帧的下界完成所有变量的下界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的下界添加到整个状态变量的下界中
void OCPConfig::coverLowerInputBounds(const casadi::SX &oneFrameLowerBound) {
    inputLowerBounds_.clear();
    for (int i = 0; i < horizon_; ++i) {
        inputLowerBounds_.emplace_back(oneFrameLowerBound);
    }
}

//    使用一帧的上界完成所有变量的上界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的上界添加到整个状态变量的上界中
void OCPConfig::coverUpperInputBounds(const casadi::SX &oneFrameLowerBound) {
    inputUpperBounds_.clear();
    for (int i = 0; i < horizon_; ++i) {
        inputUpperBounds_.emplace_back(oneFrameLowerBound);
    }
}

// todo : 还是需要把所有的变量按照一个非常规整的形式打印出来，
std::vector<casadi::DM> OCPConfig::getStatusLowerBounds() const {
    return statusLowerBounds_;
}

std::vector<casadi::DM> OCPConfig::getInputLowerBounds() const {
    return inputLowerBounds_;
}

std::vector<casadi::DM> OCPConfig::getStatusUpperBounds() const {
    return statusUpperBounds_;
}

std::vector<casadi::DM> OCPConfig::getInputUpperBounds() const {
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
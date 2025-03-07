//
// Created by lock on 25-3-7.
//

#ifndef BUILD_OCPCONFIG_H
#define BUILD_OCPCONFIG_H

#include "yaml-cpp/yaml.h"
#include "casadi/casadi.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

// Frame是用来记录状态和输入变量的结构体，包括总大小、字段名称和偏移量等信息。
struct Frame {
    int totalSize;
    std::vector<std::pair<std::string, int>> fields;
    std::unordered_map<std::string, int> fieldOffsets;
};

class OCPConfig {
private:
    int horizon_{0};
    double dt_{0.1};
    casadi::SX statusVariables_, inputVariables_;
    Frame statusFrame_, inputFrame_;
    bool verbose_;
    std::string problemName_;
    std::vector<casadi::DM> statusUpperBounds_;
    std::vector<casadi::DM> statusLowerBounds_;
    std::vector<casadi::DM> inputUpperBounds_;
    std::vector<casadi::DM> inputLowerBounds_;
    ::casadi::DM initialGuess_;

private:
    void initializeFrame(Frame &frame, const YAML::Node &config);

    void parseOCPBounds(YAML::Node);

    void coverUpperInputBounds(const casadi::SX &oneFrameLowerBound);

    void coverUpperStatusBounds(const casadi::SX &oneFrameUpperBound);

    void coverLowerInputBounds(const casadi::SX &oneFrameUpperBound);

    void coverLowerStatusBounds(const casadi::SX &oneFrameLowerBound);


public:
    casadi::SX getStatusVariable(int stepID, const std::string &variableName) const;

    casadi::SX getInputVariable(int frameID, const std::string &fieldName) const;

    casadi::SX getStatusVariables() const;

    casadi::SX getInputVariables() const;

    std::vector<casadi::DM> getStatusLowerBounds() const;

    std::vector<casadi::DM> getInputLowerBounds() const;

    std::vector<casadi::DM> getStatusUpperBounds() const;

    std::vector<casadi::DM> getInputUpperBounds() const;

    int getHorizon() const;

    double getDt() const;

    int getStatusFrameSize() const;

    int getInputFrameSize() const;

    void setStatusBounds(const ::casadi::DM &lowerBound, const ::casadi::DM &upperBound);

    void setInitialGuess(const casadi::DM &initialGuess);

    casadi::DM getInitialGuess();

    OCPConfig();

    ~OCPConfig() = default;

    casadi::DM getVariableLowerBounds() const;

    casadi::DM getVariableUpperBounds() const;
};


#endif //BUILD_OCPCONFIG_H

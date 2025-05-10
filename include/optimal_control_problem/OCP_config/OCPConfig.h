//
// Created by lock on 25-3-7.
//

#ifndef BUILD_OCPCONFIG_H
#define BUILD_OCPCONFIG_H

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "yaml-cpp/yaml.h"
#include "casadi/casadi.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

// 简化的错误处理宏
#define OCP_ERROR_MSG(msg) \
    std::cerr << "[ERROR] " << msg << std::endl

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
    casadi::SX variables_;
    bool verbose_;
    Frame variableFrame_;
    std::vector<casadi::DM> upperBounds_;
    std::vector<casadi::DM> lowerBounds_;
    ::casadi::DM initialGuess_;

private:
    static void initializeFrame(Frame &frame, const YAML::Node &config);

    void parseOCPBounds(YAML::Node);

    void coverLowerBounds(const casadi::SX &oneFrameLowerBound);

    void coverUpperBounds(const casadi::SX &oneFrameUpperBound);


public:
    casadi::SX getVariable(int stepID, const std::string &variableName) const;

    casadi::SX getVariables() const;

    std::vector<casadi::DM> getLowerBounds() const;

    std::vector<casadi::DM> getUpperBounds() const;

    int getHorizon() const;

    double getTimeStep() const;

    int getFrameSize() const;

    void setInitialGuess(const casadi::DM &initialGuess);

    casadi::DM getInitialGuess();

    /**
     * Prints a formatted summary of the OCP configuration
     * Displays horizon, time step, variable structure, and bounds in a tabular format
     */
    void printSummary() const;

    /*
     * 传入的参数是optimal_control_problem的Node
     * */
    explicit OCPConfig(YAML::Node);

    ~OCPConfig() = default;
};


#endif //BUILD_OCPCONFIG_H

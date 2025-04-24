//
// Created by lock on 25-3-7.
//

// 在文件顶部添加
#include <iostream>
#include <iomanip>

// 调试宏，可以根据需要开关
#define DEBUG_OCP 1

#define OCP_LOG(level, msg) \
    if (DEBUG_OCP) { \
        std::cout << "[" << level << "] " << msg << std::endl; \
    }

#define OCP_ERROR(msg) OCP_LOG("ERROR", msg)
#define OCP_WARN(msg) OCP_LOG("WARN", msg)
#define OCP_INFO(msg) OCP_LOG("INFO", msg)
#define OCP_DEBUG(msg) OCP_LOG("DEBUG", msg)


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

    double getDt() const;

    int getFrameSize() const;

//    void setStatusBounds(const ::casadi::DM &lowerBound, const ::casadi::DM &upperBound);

    void setInitialGuess(const casadi::DM &initialGuess);

    casadi::DM getInitialGuess();

    /*
     * 传入的参数是optimal_control_problem的Node
     * */
    explicit OCPConfig(YAML::Node);

    ~OCPConfig() = default;
};


#endif //BUILD_OCPCONFIG_H

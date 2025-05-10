#pragma once

#include <casadi/casadi.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <iostream>
#include <filesystem>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "sqp_solver/SQPOptimizationSolver.h"
#include "optimal_control_problem/OCP_config/OCPConfig.h"

class OptimalControlProblem {
private:
    class SolverSettings{
    public:
        enum class SolverType {
            IPOPT,
            SQP,
            CUDA_SQP,
            MIXED
        };

        struct SQPSettings{
            double alpha{0.1};
            int stepNum{10};
        };

        bool verbose{true};
        bool genCode{false};
        bool recompile{false};
        bool loadLib{false};
        bool warmStart{true};
        int maxIter{1000};
        SolverType solverType;
        SQPSettings SQP_settings;
    };

    YAML::Node configNode_;
    SolverSettings solverSettings;

    std::vector<casadi::SX> constraints_;
    std::vector<std::string> constraintNames_;
    std::vector<casadi::DM> constraintLowerBounds_;
    std::vector<casadi::DM> constraintUpperBounds_;

    casadi::SXVector costs_;
    bool setInitialGuess_{false};
    bool firstTime_{true};
    casadi::DM optimalTrajectory_;

    std::string packagePath_;

    casadi::Function IPOPTSolver_;
    casadi::Function SQPSolver_;
    casadi::Function libIPOPTSolver_;
    casadi::Function libSQPSolver_;
    std::shared_ptr<SQPOptimizationSolver> OSQPSolverPtr_;

    // 新增：检查YAML配置是否有效
    bool validateConfig(const YAML::Node& config);
    // 新增：检查目录权限
    bool checkDirectoryPermissions(const std::string& path);

    void printSummary() const;

public:
    std::unique_ptr<OCPConfig> OCPConfigPtr_;

    void setSolverType(SolverSettings::SolverType type);
    SolverSettings::SolverType getSolverType() const;

    casadi::SX getReference() const;
    casadi::DM getOptimalTrajectory();

    casadi::SX reference_;
    casadi::SX totalCost_;

    void genSolver();
    void computeOptimalTrajectory(const casadi::DM &frame, const casadi::DM &reference);

    void setReference(const casadi::SX& reference);

    explicit OptimalControlProblem(YAML::Node);

    void addScalarCost(const casadi::SX &cost);
    void addVectorCost(const casadi::DM &numericParam, const casadi::SX &symbolicTerm);
    void addVectorCost(const std::vector<double> &numericParam, const SX &symbolicTerm);

    void addInequalityConstraint(const std::string &constraintName,
                                 const casadi::DM &lowerBound,
                                 const casadi::SX &expression,
                                 const casadi::DM &upperBound);
    void addEquationConstraint(const std::string &constraintName, const casadi::SX &leftSX, const casadi::SX &rightSX);
    void addEquationConstraint(const std::string &constraintName, const casadi::SX &expression);

    casadi::SX getCostFunction();
    casadi::DMVector getConstraintLowerBounds() const;
    casadi::DMVector getConstraintUpperBounds() const;

    std::vector<casadi::SX> getConstraints() const;

    virtual void deployConstraintsAndAddCost() = 0;

    bool solverInputCheck(std::map<std::string, casadi::DM> arg) const;

    void
    compileLibrary(const std::string &source_file, const std::string &output_file, const std::string &compile_flags);
};

std::ostream &operator<<(std::ostream &os, const OptimalControlProblem &ocp);

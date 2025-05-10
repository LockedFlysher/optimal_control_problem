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
#include <mutex>
#include <atomic>
#include <thread> // 我在此修改：添加线程支持
#include <future> // 我在此修改：添加future支持
#include <condition_variable> // 我在此修改：添加条件变量支持

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

    enum class SolverState {
        IDLE,
        BUSY,
        FAILED,
        COMPLETED
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

    std::vector<casadi::DM> optimalTrajectories_;

    std::string packagePath_;

    casadi::Function IPOPTSolver_;
    casadi::Function SQPSolver_;
    casadi::Function libIPOPTSolver_;
    casadi::Function libSQPSolver_;
    std::shared_ptr<SQPOptimizationSolver> OSQPSolverPtr_;

    std::vector<SolverState> solverStates_;
    std::mutex solverMutex_;
    std::atomic<int> maxSolverCount_{1}; // 默认为1个求解器

    // 我在此修改：添加线程管理相关成员
    std::vector<std::thread> solverThreads_;
    std::vector<std::future<void>> solverFutures_;
    std::vector<std::string> errorMessages_;
    std::condition_variable stateChangeCV_;

    // 新增：检查YAML配置是否有效
    bool validateConfig(const YAML::Node& config);
    // 新增：检查目录权限
    bool checkDirectoryPermissions(const std::string& path);

    // 我在此修改：添加异步求解的内部辅助函数
    void solveTrajectoryAsync(const casadi::DM &frame, const casadi::DM &reference, int solverId);
    void setSolverState(int solverId, SolverState state, const std::string& errorMsg = "");

    void printSummary() const;

public:
    std::unique_ptr<OCPConfig> OCPConfigPtr_;

    void setSolverType(SolverSettings::SolverType type);
    SolverSettings::SolverType getSolverType() const;

    casadi::SX getReference() const;

    casadi::DM getOptimalTrajectory(int solverId = 0);
    void setMaxSolverCount(int count);
    int getMaxSolverCount() const;

    casadi::SX reference_;
    casadi::SX totalCost_;

    void genSolver();

    // 我在此修改：修改计算最优轨迹的函数声明
    void computeOptimalTrajectory(const casadi::DM &frame, const casadi::DM &reference, int solverId = 0);
    // 我在此修改：添加异步计算最优轨迹的函数声明
    void computeOptimalTrajectoryAsync(const casadi::DM &frame, const casadi::DM &reference, int solverId = 0);
    // 我在此修改：添加等待求解完成的函数声明
    bool waitForSolver(int solverId, int timeoutMs = -1);
    // 我在此修改：添加获取错误信息的函数声明
    std::string getSolverErrorMessage(int solverId);

    void setReference(const casadi::SX& reference);

    explicit OptimalControlProblem(YAML::Node);
    // 我在此修改：添加析构函数声明
    ~OptimalControlProblem();

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

    SolverState getSolverState(int solverId = 0);
};

std::ostream &operator<<(std::ostream &os, const OptimalControlProblem &ocp);

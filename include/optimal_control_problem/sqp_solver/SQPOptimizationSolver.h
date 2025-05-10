#pragma once

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <chrono> // Added timing functionality
#include <sstream> // 用于字符串流处理
#include "optimal_control_problem/sqp_solver/AutoDifferentiator.h"
#include "optimal_control_problem/sqp_solver/CasadiGpuEvaluator.h"
#include "optimal_control_problem/sqp_solver/CuCaQP.h"
#include <torch/script.h>
#include <vector>

class SQPOptimizationSolver {
public:
    /**
     * @brief Constructor, initializes the SQP solver
     * @param nlp Dictionary of symbolic expressions for the nonlinear programming problem
     * @param options Solver options
     * @param numSolvers Number of solvers, default is 1
     */
    explicit SQPOptimizationSolver(::casadi::SXDict &nlp, ::casadi::Dict &options, int numSolvers = 1);

    ::casadi::Function getSXLocalSystemFunction() const;

    /**
     * @brief Destructor, releases resources
     */
    ~SQPOptimizationSolver() = default;

    /**
     * @brief SQP solving method (LibTorch interface) - single instance
     * @note For MPC problems, the initial point is set by restricting the first item's lbx and ubx, consistent with OptimalControlProblem
     * @param arg <string,DM> dictionary, using lbx, ubx, lbg, ubg, p(optional)
     * @retval DMDict result containing optimal solution x and objective function value f
     */
    casadi::DMDict getOptimalSolution(const casadi::DMDict &arg);

    /**
     * @brief SQP solving method (LibTorch interface) - multiple instances
     * @param args Vector of multiple input parameter dictionaries
     * @retval std::vector<DMDict> Vector containing multiple optimal solutions and objective function values
     */
    std::vector<casadi::DMDict> getOptimalSolution(const std::vector<casadi::DMDict> &args);

    /**
     * @brief Set whether to output detailed information
     * @param verbose Whether to output detailed information
     */
    void setVerbose(bool verbose);
    void loadFromFile();

private:
    // Auto-differentiator
    std::shared_ptr<AutoDifferentiator> objectiveFunctionAutoDifferentiatorPtr_;
    std::shared_ptr<AutoDifferentiator> constraintsAutoDifferentiator_;

    // Number of solvers
    int numSolvers_;

    // QP solver - supports multiple instances
    std::vector<CuCaQP> qpSolvers_;

    // SQP parameters
    int stepNum_;         // Maximum number of iterations
    double alpha_;        // Step size factor
    bool verbose_;        // Whether to output detailed information
    bool useCUDA_{true};  // Whether to use CUDA's CuCaQP, enabled by default

    // cusadi
    std::string functionFilePath_; // Path to store localsystemfunction
    int N_ENVS;
    casadi::Function fn;
    std::vector<std::unique_ptr<CasadiGpuEvaluator>> solvers_; // Multiple solver instances

    // Optimization results, extracted using Dict - supports multiple instances
    std::vector<casadi::DMDict> results_;
    std::vector<std::map<std::string, torch::Tensor>> resultsTensor_;

    // Constraint boundaries
    casadi::DM lowerBounds_;
    casadi::DM upperBounds_;

    // Objective function
    casadi::Function objectiveFunction_;

    /**
     * @brief Local system function
     * @note Produces in sequence: 1. Hessian, 2. gradient, 3. A, 4. l, 5. u
     * @param [0]reference [1]point [2]lowerBound [3]upperBound
     * @retval [0]Hessian [1]Gradient [2]A [3]l [4]u
     */
    casadi::Function localSystemFunction_;

private:
    /**
     * @brief Get local system (CasADi version)
     * @param arg Input parameter dictionary
     * @param solverIndex Solver index
     * @return DMVector of the local system
     */
    std::vector<torch::Tensor> getLocalSystem(const ::casadi::DMDict &arg, int solverIndex);

    /**
     * @brief Convert CasADi DM to torch::Tensor
     * @param dm CasADi DM matrix or vector
     * @return Converted torch::Tensor
     */
    torch::Tensor dmToTensor(const ::casadi::DM &dm);

    /**
     * @brief Set computation backend
     * @param useCUDA Whether to use CUDA backend
     */
    void setBackend(bool useCUDA);

    /**
     * @brief 打印向量的前几个元素
     * @param vec 要打印的向量
     * @param maxElements 最多显示的元素数量
     * @return 格式化的字符串
     */
    std::string printVectorPreview(const casadi::DM &vec, int maxElements = 5);

    /**
     * @brief 清屏函数
     */
    void clearScreen();
};

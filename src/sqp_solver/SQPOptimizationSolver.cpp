//
// Created by lock on 25-3-11.
//
#include "optimal_control_problem/sqp_solver/SQPOptimizationSolver.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

using namespace casadi;
using namespace std::chrono;

SQPOptimizationSolver::SQPOptimizationSolver(::casadi::SXDict &nlp, ::casadi::Dict &options, int numSolvers)
        : numSolvers_(numSolvers > 0 ? numSolvers : 1), verbose_(false) {
    // Load SQP parameters
    // Set default values
    int defaultMaxIter = 10;
    double defaultAlpha = 0.2;
    bool defaultVerbose = true;

    // Try to get options, use defaults if they don't exist
    try {
        stepNum_ = options.find("max_iter") != options.end() ?
                   options.at("max_iter").as_int() : defaultMaxIter;
    } catch (const std::exception& e) {
        // If type conversion fails, use default value
        stepNum_ = defaultMaxIter;
    }

    try {
        alpha_ = options.find("alpha") != options.end() ?
                 options.at("alpha").as_double() : defaultAlpha;
    } catch (const std::exception& e) {
        // If type conversion fails, use default value
        alpha_ = defaultAlpha;
    }

    try {
        bool verbose = options.find("verbose") != options.end() ?
                       options.at("verbose").as_bool() : defaultVerbose;
        setVerbose(verbose);
    } catch (const std::exception& e) {
        // If type conversion fails, use default value
        setVerbose(defaultVerbose);
    }

    // Required parameter check
    if (nlp.find("f") == nlp.end()) {
        throw std::invalid_argument("Objective function 'f' not defined");
    }
    auto objectExpr = nlp["f"];

    if (nlp.find("x") == nlp.end()) {
        throw std::invalid_argument("Optimization variable 'x' not defined");
    }
    auto variables = nlp["x"];

    // Optional parameter handling
    SX constraints;
    if (nlp.find("g") != nlp.end()) {
        constraints = nlp["g"];
    } else {
        // Set empty constraints
        constraints = ::casadi::SX();
    }

    SX reference;
    if (nlp.find("p") != nlp.end()) {
        reference = nlp["p"];
    } else {
        reference = ::casadi::SX();
    }

    // Create objective function
    objectiveFunction_ = Function("objective", {reference, variables}, {objectExpr});

    // Create augmented variables
    auto augmentedVariables = SX::vertcat({reference, variables});

    // Initialize auto-differentiator
    objectiveFunctionAutoDifferentiatorPtr_ = std::make_shared<AutoDifferentiator>(augmentedVariables, objectExpr);
    SX augmentedConstraints = SX::vertcat({reference, variables, constraints});
    constraintsAutoDifferentiator_ = std::make_shared<AutoDifferentiator>(augmentedVariables, augmentedConstraints);

    // Calculate Hessian and gradient
    auto hessian = objectiveFunctionAutoDifferentiatorPtr_->getHessian(augmentedVariables);
    auto gradient = objectiveFunctionAutoDifferentiatorPtr_->getGradient(augmentedVariables);

    // Linearize constraints
    auto linearizedIneqConstraints = constraintsAutoDifferentiator_->getLinearization(augmentedVariables);
    auto numOfConstraints = linearizedIneqConstraints[1].size1();

    // Create constraint boundary symbols
    auto l = SX::sym("l", numOfConstraints);
    auto u = SX::sym("u", numOfConstraints);

    // Calculate linearized constraint boundaries
    auto l_linearized = l + linearizedIneqConstraints[1];
    auto u_linearized = u + linearizedIneqConstraints[1];

    // Create local system function
    localSystemFunction_ = Function("localSystemFunction",
                                    {reference, variables, l, u},
                                    {hessian, gradient, linearizedIneqConstraints[0],
                                     l_linearized, u_linearized});

    // Initialize multiple solver instances
    qpSolvers_.resize(numSolvers_);
    solvers_.resize(numSolvers_);
    results_.resize(numSolvers_);
    resultsTensor_.resize(numSolvers_);

    for (int i = 0; i < numSolvers_; ++i) {
        // Configure QP solver
        qpSolvers_[i].setDimension(augmentedVariables.size1(), augmentedConstraints.size1());
        qpSolvers_[i].setVerbosity(false);
        qpSolvers_[i].setWarmStart(true);
        qpSolvers_[i].setAbsoluteTolerance(1e-3);
        qpSolvers_[i].setRelativeTolerance(1e-3);
        qpSolvers_[i].setMaxIteration(10000);

        // Initialize result dictionary
        results_[i] = {
                {"x", DM::zeros(variables.size1())},
                {"f", DM::zeros(1)}
        };

        // Initialize Tensor result dictionary
        resultsTensor_[i] = {
                {"x", torch::zeros({static_cast<int64_t>(variables.size1()), 1})},
                {"f", torch::zeros({1, 1})}
        };
    }

    // Enable CUDA by default
    // setBackend(true);
}

void SQPOptimizationSolver::loadFromFile() {
    // cusadi related parameters
    // Path to load localsystemfunction
    functionFilePath_ = ament_index_cpp::get_package_share_directory("optimal_control_problem") +
                        "/code_gen/localSystemFunction.casadi";
    N_ENVS = 1;
    fn = casadi::Function::load(functionFilePath_);
    if (verbose_)
        std::cout << "Loaded CasADi function: " << fn.name() << "\n";

    // Load function for each solver instance
    for (int i = 0; i < numSolvers_; ++i) {
        solvers_[i] = std::make_unique<CasadiGpuEvaluator>(fn);
    }
}

casadi::Function SQPOptimizationSolver::getSXLocalSystemFunction() const {
    return localSystemFunction_;
}

std::vector<torch::Tensor> SQPOptimizationSolver::getLocalSystem(const DMDict &arg, int solverIndex) {
    // Get boundary conditions
    DM lbx = arg.at("lbx");
    DM ubx = arg.at("ubx");
    DM lbg = arg.at("lbg");
    DM ubg = arg.at("ubg");
    DM p_dm = (arg.find("p") != arg.end()) ? arg.at("p") : DM::zeros(0, 1);
    DM x_dm = results_[solverIndex].at("x");
    DM l_dm = DM::vertcat({p_dm, lbx, lbg});
    DM u_dm = DM::vertcat({p_dm, ubx, ubg});

    // 2. Convert DM to Tensor, ensuring data type is kFloat64
    auto p_tensor = dmToTensor(p_dm).to(torch::kFloat64).to(torch::kCUDA);
    auto x_tensor = dmToTensor(x_dm).to(torch::kFloat64).to(torch::kCUDA);

    auto l_tensor = dmToTensor(l_dm).to(torch::kFloat64).to(torch::kCUDA);
    auto u_tensor = dmToTensor(u_dm).to(torch::kFloat64).to(torch::kCUDA);
    // Ensure input Tensor shapes are correct (2D column vectors)
    p_tensor = p_tensor.view({p_dm.size1(), 1});
    x_tensor = x_tensor.view({x_dm.size1(), 1});
    l_tensor = l_tensor.view({l_dm.size1(), 1});
    u_tensor = u_tensor.view({u_dm.size1(), 1});

    std::vector<torch::Tensor> inputs = {p_tensor, x_tensor, l_tensor, u_tensor};
    solvers_[solverIndex]->compute(inputs);
    std::vector<torch::Tensor> outTensorVector;
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(0).cpu());
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(1).cpu());
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(2).cpu());
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(3).cpu());
    outTensorVector.push_back(solvers_[solverIndex]->getDenseResult(4).cpu());
    return outTensorVector;
}

// 辅助函数：打印向量的前几个元素
std::string SQPOptimizationSolver::printVectorPreview(const DM &vec, int maxElements) {
    std::stringstream ss;
    int size = vec.size1();

    ss << "[";
    for (int i = 0; i < std::min(size, maxElements); ++i) {
        ss << vec(i).scalar();
        if (i < std::min(size, maxElements) - 1) {
            ss << ", ";
        }
    }

    if (size > maxElements) {
        ss << ", ... (" << size - maxElements << " more)";
    }
    ss << "]";

    return ss.str();
}

// 清屏函数
void SQPOptimizationSolver::clearScreen() {
    // 使用ANSI转义序列清屏并将光标移到左上角
    std::cout << "\033[2J\033[1;1H";
}

DMDict SQPOptimizationSolver::getOptimalSolution(const DMDict &arg) {
    // Initialize performance statistics variables
    auto totalStartTime = high_resolution_clock::now();
    double totalQpSolveTime = 0.0;
    double totalLocalSystemTime = 0.0;
    double totalConvertingTime = 0.0;

    // Display initial information only once if verbose mode is enabled
    if (verbose_) {
        std::cout << "SQP Optimization Started (Single Instance)" << std::endl;
        std::cout << "Maximum iterations: " << stepNum_ << ", Step factor: " << alpha_ << std::endl;
    }

    for (int i = 0; i < stepNum_; ++i) {
        // Calculate local system
        auto localSystemStartTime = high_resolution_clock::now();
        std::vector<torch::Tensor> localSystem = getLocalSystem(arg, 0);
        auto localSystemEndTime = high_resolution_clock::now();
        double localSystemTime =
                duration_cast<microseconds>(localSystemEndTime - localSystemStartTime).count() / 1000.0;
        totalLocalSystemTime += localSystemTime;

        // Data type conversion
        auto convertingStartTime = high_resolution_clock::now();
        qpSolvers_[0].setSystem(localSystem);
        auto convertingEndTime = high_resolution_clock::now();
        double convertingTime = duration_cast<microseconds>(convertingEndTime - convertingStartTime).count() / 1000.0;
        totalConvertingTime += convertingTime;

        // Solve QP problem
        auto qpStartTime = high_resolution_clock::now();
        qpSolvers_[0].initSolver();
        qpSolvers_[0].solve();
        auto qpEndTime = high_resolution_clock::now();
        double qpSolveTime = duration_cast<microseconds>(qpEndTime - qpStartTime).count() / 1000.0;
        totalQpSolveTime += qpSolveTime;

        // Get solution and update
        DM solution = qpSolvers_[0].getSolutionAsDM();
        DM oldRes = results_[0].at("x");
        // Update result based on whether reference variable p exists
        if (arg.find("p") == arg.end() || arg.at("p").size1() == 0) {
            results_[0].at("x") += alpha_ * solution;
        } else {
            // Extract variable part from complete solution (excluding reference variable p)
            int pSize = arg.at("p").size1();
            results_[0].at("x") += alpha_ * solution(Slice(pSize, solution.size1()));
        }

        // Calculate and update objective function value
        DM p = (arg.find("p") != arg.end()) ? arg.at("p") : DM::zeros(0, 1);
        results_[0].at("f") = objectiveFunction_(DMVector{p, results_[0].at("x")});

        // Calculate solution change
        DM delta = results_[0].at("x") - oldRes;
        double normDelta = norm_2(delta).scalar();

        // Terminate early if change is very small
        if (normDelta < 1e-6) {
            break;
        }
    }

    auto totalEndTime = high_resolution_clock::now();
    double totalTime = duration_cast<microseconds>(totalEndTime - totalStartTime).count() / 1000.0;

    // Display results only at the end
    if (verbose_) {
        clearScreen();
        std::cout << "\n=============== Optimization Results ===============" << std::endl;
        std::cout << "Objective function value: " << results_[0].at("f") << std::endl;
        std::cout << "Optimal solution: " << printVectorPreview(results_[0].at("x"), 5) << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << totalTime << " ms" << std::endl;
        std::cout << "Performance breakdown:" << std::endl;
        std::cout << "  Local system calculation: " << std::fixed << std::setprecision(2)
                  << totalLocalSystemTime << " ms ("
                  << std::fixed << std::setprecision(1) << (totalLocalSystemTime / totalTime * 100) << "%)" << std::endl;
        std::cout << "  Data type conversion: " << std::fixed << std::setprecision(2)
                  << totalConvertingTime << " ms ("
                  << std::fixed << std::setprecision(1) << (totalConvertingTime / totalTime * 100) << "%)" << std::endl;
        std::cout << "  QP solving: " << std::fixed << std::setprecision(2)
                  << totalQpSolveTime << " ms ("
                  << std::fixed << std::setprecision(1) << (totalQpSolveTime / totalTime * 100) << "%)" << std::endl;
    }

    return results_[0];
}


std::vector<DMDict> SQPOptimizationSolver::getOptimalSolution(const std::vector<DMDict> &args) {
    std::vector<DMDict> outputResults;
    int numInputs = static_cast<int>(args.size());
    if (numInputs > numSolvers_) {
        std::cerr << "Warning: Number of inputs (" << numInputs << ") exceeds number of solvers (" << numSolvers_
                  << "), will only process the first " << numSolvers_ << " inputs." << std::endl;
        numInputs = numSolvers_;
    }

    // Display basic information only once at the beginning
    if (verbose_) {
        std::cout << "SQP Optimization Started (Multiple instances, processing " << numInputs << " inputs)" << std::endl;
        std::cout << "Maximum iterations: " << stepNum_ << ", Step factor: " << alpha_ << std::endl;
    }

    outputResults.resize(numInputs);

    for (int solverIndex = 0; solverIndex < numInputs; ++solverIndex) {
        const auto& arg = args[solverIndex];

        // Initialize performance statistics variables
        auto totalStartTime = high_resolution_clock::now();
        double totalQpSolveTime = 0.0;
        double totalLocalSystemTime = 0.0;
        double totalConvertingTime = 0.0;

        for (int i = 0; i < stepNum_; ++i) {
            // Calculate local system
            auto localSystemStartTime = high_resolution_clock::now();
            std::vector<torch::Tensor> localSystem = getLocalSystem(arg, solverIndex);
            auto localSystemEndTime = high_resolution_clock::now();
            double localSystemTime =
                    duration_cast<microseconds>(localSystemEndTime - localSystemStartTime).count() / 1000.0;
            totalLocalSystemTime += localSystemTime;

            // Data type conversion
            auto convertingStartTime = high_resolution_clock::now();
            qpSolvers_[solverIndex].setSystem(localSystem);
            auto convertingEndTime = high_resolution_clock::now();
            double convertingTime = duration_cast<microseconds>(convertingEndTime - convertingStartTime).count() / 1000.0;
            totalConvertingTime += convertingTime;

            // Solve QP problem
            auto qpStartTime = high_resolution_clock::now();
            qpSolvers_[solverIndex].initSolver();
            qpSolvers_[solverIndex].solve();
            auto qpEndTime = high_resolution_clock::now();
            double qpSolveTime = duration_cast<microseconds>(qpEndTime - qpStartTime).count() / 1000.0;
            totalQpSolveTime += qpSolveTime;

            // Get solution and update
            DM solution = qpSolvers_[solverIndex].getSolutionAsDM();
            DM oldRes = results_[solverIndex].at("x");
            // Update result based on whether reference variable p exists
            if (arg.find("p") == arg.end() || arg.at("p").size1() == 0) {
                results_[solverIndex].at("x") += alpha_ * solution;
            } else {
                // Extract variable part from complete solution (excluding reference variable p)
                int pSize = arg.at("p").size1();
                results_[solverIndex].at("x") += alpha_ * solution(Slice(pSize, solution.size1()));
            }

            // Calculate and update objective function value
            DM p = (arg.find("p") != arg.end()) ? arg.at("p") : DM::zeros(0, 1);
            results_[solverIndex].at("f") = objectiveFunction_(DMVector{p, results_[solverIndex].at("x")});

            // Calculate solution change
            DM delta = results_[solverIndex].at("x") - oldRes;
            double normDelta = norm_2(delta).scalar();

            // Terminate early if change is very small
            if (normDelta < 1e-6) {
                break;
            }
        }

        auto totalEndTime = high_resolution_clock::now();
        double totalTime = duration_cast<microseconds>(totalEndTime - totalStartTime).count() / 1000.0;

        // Display final results for each instance
        if (verbose_) {
            std::cout << "\n=============== Instance " << solverIndex + 1 << " Optimization Results ===============" << std::endl;
            std::cout << "Objective function value: " << results_[solverIndex].at("f") << std::endl;
            std::cout << "Optimal solution: " << printVectorPreview(results_[solverIndex].at("x"), 5) << std::endl;
            std::cout << "Total time: " << std::fixed << std::setprecision(2) << totalTime << " ms" << std::endl;
            std::cout << "Performance breakdown:" << std::endl;
            std::cout << "  Local system calculation: " << std::fixed << std::setprecision(2)
                      << totalLocalSystemTime << " ms ("
                      << std::fixed << std::setprecision(1) << (totalLocalSystemTime / totalTime * 100) << "%)" << std::endl;
            std::cout << "  Data type conversion: " << std::fixed << std::setprecision(2)
                      << totalConvertingTime << " ms ("
                      << std::fixed << std::setprecision(1) << (totalConvertingTime / totalTime * 100) << "%)" << std::endl;
            std::cout << "  QP solving: " << std::fixed << std::setprecision(2)
                      << totalQpSolveTime << " ms ("
                      << std::fixed << std::setprecision(1) << (totalQpSolveTime / totalTime * 100) << "%)" << std::endl;
        }

        // Store results
        outputResults[solverIndex] = results_[solverIndex];
    }

    if (verbose_) {
        std::cout << "\n=============== All Instances Optimization Completed ===============" << std::endl;
    }

    return outputResults;
}

void SQPOptimizationSolver::setVerbose(bool verbose) {
    verbose_ = verbose;
}

/**
* @brief Convert CasADi DM to torch::Tensor
* @param dm CasADi DM matrix or vector
* @return Converted torch::Tensor
*/
torch::Tensor SQPOptimizationSolver::dmToTensor(const casadi::DM &dm) {
    // Get DM dimensions
    int rows = dm.size1();
    int cols = dm.size2();
    // Create torch::Tensor
    torch::Tensor tensor = torch::zeros({rows, cols}, torch::kFloat32);
    // For dense matrices, directly fill all elements
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tensor[i][j] = dm(i, j).scalar();
        }
    }
    return tensor;
}

void SQPOptimizationSolver::setBackend(bool useCUDA) {
    this->useCUDA_ = useCUDA;
}

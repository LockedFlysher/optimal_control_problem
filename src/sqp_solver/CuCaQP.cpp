#include "optimal_control_problem/sqp_solver/CuCaQP.h"
#include <iostream>


CuCaQP::CuCaQP() :
        numOfVariables_(0),
        numOfConstraints_(0),
        isInitialized_(false),
        upperBound(),
        lowerBound(),
        gradient(),
        hessianMatrix(),
        linearConstraintMatrix() {
}

CuCaQP::~CuCaQP() {
    // 确保在析构时清理资源
    if (isInitialized_) {
        solver_.clearSolver();
    }
}

bool CuCaQP::setDimension(int numOfVariables, int numOfConstraints) {
    if (numOfVariables <= 0 || numOfConstraints <= 0) {
        std::cerr << "Error: Invalid dimensions." << std::endl;
        return false;
    }

    // 如果求解器已初始化，先清理
    if (isInitialized_) {
        solver_.clearSolver();
        isInitialized_ = false;
    }

    numOfVariables_ = numOfVariables;
    numOfConstraints_ = numOfConstraints;

    solver_.data()->setNumberOfVariables(numOfVariables_);
    solver_.data()->setNumberOfConstraints(numOfConstraints_);
    return true;
}

bool CuCaQP::setHessianMatrix(const casadi::DM &hessian) {
    // 先清理之前的Hessian矩阵
    solver_.data()->clearHessianMatrix();

    hessianMatrix = casadiDMToEigenSparse(hessian);
    if (hessianMatrix.rows() != numOfVariables_ || hessianMatrix.cols() != numOfVariables_) {
        std::cerr << "Error: Hessian matrix dimensions mismatch. Expected "
                  << numOfVariables_ << "x" << numOfVariables_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setHessianMatrix(hessianMatrix);
    return result;
}

bool CuCaQP::setHessianMatrix(const torch::Tensor &hessian) {
    // 先清理之前的Hessian矩阵
    solver_.data()->clearHessianMatrix();

    hessianMatrix = torchTensorToEigenSparse(hessian);
    if (hessianMatrix.rows() != numOfVariables_ || hessianMatrix.cols() != numOfVariables_) {
        std::cerr << "Error: Hessian matrix dimensions mismatch. Expected "
                  << numOfVariables_ << "x" << numOfVariables_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setHessianMatrix(hessianMatrix);
    return result;
}

bool CuCaQP::setGradient(const casadi::DM &q) {
    gradient = casadiDMToEigenVector(q);

    if (gradient.size() != numOfVariables_) {
        std::cerr << "Error: Gradient vector size mismatch. Expected " << numOfVariables_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setGradient(gradient);
    return result;
}

bool CuCaQP::setGradient(const torch::Tensor &q) {
    // 提前检查尺寸
    if (q.numel() != numOfVariables_) {
        std::cerr << "Error: Gradient vector size mismatch. Expected " << numOfVariables_ << std::endl;
        return false;
    }
    // 转换并直接设置梯度
    gradient = torchTensorToEigenVector(q);
    return solver_.data()->setGradient(gradient);
}


bool CuCaQP::setLinearConstraintsMatrix(const DM &A) {
    // 先清理之前的线性约束矩阵
    solver_.data()->clearLinearConstraintsMatrix();

    linearConstraintMatrix = casadiDMToEigenSparse(A);

    if (linearConstraintMatrix.rows() != numOfConstraints_ || linearConstraintMatrix.cols() != numOfVariables_) {
        std::cerr << "Error: Constraint matrix dimensions mismatch. Expected "
                  << numOfConstraints_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    bool result = solver_.data()->setLinearConstraintsMatrix(linearConstraintMatrix);
    return result;
}

bool CuCaQP::setLinearConstraintsMatrix(const torch::Tensor &A) {
    // 先清理之前的线性约束矩阵
    solver_.data()->clearLinearConstraintsMatrix();

    linearConstraintMatrix = torchTensorToEigenSparse(A);

    if (linearConstraintMatrix.rows() != numOfConstraints_ || linearConstraintMatrix.cols() != numOfVariables_) {
        std::cerr << "Error: Constraint matrix dimensions mismatch. Expected "
                  << numOfConstraints_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    bool result = solver_.data()->setLinearConstraintsMatrix(linearConstraintMatrix);
    return result;
}

bool CuCaQP::setLowerBound(const casadi::DM &l) {
    lowerBound = casadiDMToEigenVector(l);
    if (lowerBound.size() != numOfConstraints_) {
        std::cerr << "Error: Lower bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setLowerBound(lowerBound);
    return result;
}

bool CuCaQP::setLowerBound(const torch::Tensor &l) {
    lowerBound = torchTensorToEigenVector(l);
    if (lowerBound.size() != numOfConstraints_) {
        std::cerr << "Error: Lower bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setLowerBound(lowerBound);
    return result;
}

bool CuCaQP::setUpperBound(const casadi::DM &u) {
    upperBound = casadiDMToEigenVector(u);

    if (upperBound.size() != numOfConstraints_) {
        std::cerr << "Error: Upper bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setUpperBound(upperBound);
    return result;
}

bool CuCaQP::setUpperBound(const torch::Tensor &u) {
    upperBound = torchTensorToEigenVector(u);

    if (upperBound.size() != numOfConstraints_) {
        std::cerr << "Error: Upper bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setUpperBound(upperBound);
    return result;
}

void CuCaQP::setVerbosity(bool verbosity) {
    solver_.settings()->setVerbosity(verbosity);
}

void CuCaQP::setWarmStart(bool warmStart) {
    solver_.settings()->setWarmStart(warmStart);
}

void CuCaQP::setAbsoluteTolerance(OSQPFloat tolerance) {
    solver_.settings()->setAbsoluteTolerance(tolerance);
}

void CuCaQP::setRelativeTolerance(OSQPFloat tolerance) {
    solver_.settings()->setRelativeTolerance(tolerance);
}

void CuCaQP::setMaxIteration(int maxIteration) {
    solver_.settings()->setMaxIteration(maxIteration);
}

bool CuCaQP::initSolver() {
    // 如果求解器已经初始化，先清理
    if (isInitialized_) {
        solver_.clearSolver();
        isInitialized_ = false;
    }
    bool success = solver_.initSolver();
    if (success) {
        isInitialized_ = true;
    } else {
        std::cerr << "Error: Failed to initialize solver." << std::endl;
    }
    return success;
}

bool CuCaQP::solve() {
    if (!isInitialized_) {
        std::cerr << "Error: Solver not initialized. Call initSolver() first." << std::endl;
        return false;
    }

    OsqpEigen::ErrorExitFlag result = solver_.solveProblem();
    if (result != OsqpEigen::ErrorExitFlag::NoError) {
        std::cerr << "Error: Failed to solve problem. Error code: " << static_cast<int>(result) << std::endl;
        return false;
    }
    return true;
}

Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> CuCaQP::getSolution() {
    return solver_.getSolution();
}

casadi::DM CuCaQP::getSolutionAsDM() {
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> solution_float = solver_.getSolution();
    DM solution = DM::zeros(solution_float.rows(), solution_float.cols());
    for (int i = 0; i < solution_float.rows(); ++i) {
        solution(i) = solution_float(i, 0);
    }
    return solution;
}

torch::Tensor CuCaQP::getSolutionAsTensor() {
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> solution_float = solver_.getSolution();

    // 创建一个torch::Tensor
    torch::Tensor solution = torch::zeros({solution_float.rows()}, torch::kFloat32);

    // 填充数据
    for (int i = 0; i < solution_float.rows(); ++i) {
        solution[i] = solution_float(i, 0);
    }

    return solution;
}

void CuCaQP::printSolverData() {
    // 直接访问OSQP内部数据
    const auto *osqpData = solver_.data()->getData();

    // 打印梯度向量q
    std::cout << "内部存储的梯度向量q:\n";
    for (int i = 0; i < osqpData->n; i++) {
        std::cout << osqpData->q[i] << " ";
    }
    std::cout << std::endl;

    // 打印下界l
    std::cout << "内部存储的下界l:\n";
    for (int i = 0; i < osqpData->m; i++) {
        std::cout << osqpData->l[i] << " ";
    }
    std::cout << std::endl;

    // 打印上界u
    std::cout << "内部存储的上界u:\n";
    for (int i = 0; i < osqpData->m; i++) {
        std::cout << osqpData->u[i] << " ";
    }
    std::cout << std::endl;

    // 注意：P和A是CSC格式，直接打印比较复杂
    std::cout << "内部存储的Hessian矩阵P (非零元素):\n";
    for (int j = 0; j < osqpData->n; j++) {
        for (int p = osqpData->P->p[j]; p < osqpData->P->p[j + 1]; p++) {
            int i = osqpData->P->i[p];
            c_float val = osqpData->P->x[p];
            std::cout << "(" << i << "," << j << "): " << val << std::endl;
        }
    }

    std::cout << "内部存储的约束矩阵A (非零元素):\n";
    for (int j = 0; j < osqpData->n; j++) {
        for (int p = osqpData->A->p[j]; p < osqpData->A->p[j + 1]; p++) {
            int i = osqpData->A->i[p];
            c_float val = osqpData->A->x[p];
            std::cout << "(" << i << "," << j << "): " << val << std::endl;
        }
    }
}

void CuCaQP::setSystem(DMVector localSystem) {
    // 如果求解器已初始化，先清理
    if (isInitialized_) {
        solver_.clearSolver();
        isInitialized_ = false;
    }

    // 清理所有矩阵
    solver_.data()->clearHessianMatrix();
    solver_.data()->clearLinearConstraintsMatrix();

    // 设置新的系统参数
    setHessianMatrix(casadiDMToEigenSparse(localSystem[0]));
    setGradient(casadiDMToEigenSparse(localSystem[1]));
    setLinearConstraintsMatrix(casadiDMToEigenSparse(localSystem[2]));
    setLowerBound(casadiDMToEigenSparse(localSystem[3]));
    setUpperBound(casadiDMToEigenSparse(localSystem[4]));

//    // 设置新的系统参数
//    setHessianMatrix(localSystem[0]);
//    setGradient(localSystem[1]);
//    setLinearConstraintsMatrix(localSystem[2]);
//    setLowerBound(localSystem[3]);
//    setUpperBound(localSystem[4]);


}

void CuCaQP::setSystem(const std::vector<torch::Tensor> &torchSystem) {
    // 检查输入向量大小
    if (torchSystem.size() != 5) {
        std::cerr << "Error: Expected 5 tensors in the system vector." << std::endl;
        return;
    }

    // 如果求解器已初始化，先清理
    if (isInitialized_) {
        solver_.clearSolver();
        isInitialized_ = false;
    }

    // 清理所有矩阵
    solver_.data()->clearHessianMatrix();
    solver_.data()->clearLinearConstraintsMatrix();

    // 设置新的系统参数
    setHessianMatrix(torchSystem[0]);
    setGradient(torchSystem[1]);
    setLinearConstraintsMatrix(torchSystem[2]);
    setLowerBound(torchSystem[3]);
    setUpperBound(torchSystem[4]);
}

//##################################### 暂时用不上的函数 ##################################################
bool CuCaQP::setHessianMatrix(const Eigen::SparseMatrix<OSQPFloat> &P) {
    // 先清理之前的Hessian矩阵
    solver_.data()->clearHessianMatrix();

    if (P.rows() != numOfVariables_ || P.cols() != numOfVariables_) {
        std::cerr << "Error: Hessian matrix dimensions mismatch. Expected "
                  << numOfVariables_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    hessianMatrix = P; // 保存到成员变量
    bool result = solver_.data()->setHessianMatrix(P);
    return result;
}

bool CuCaQP::setGradient(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &q) {
    if (q.size() != numOfVariables_) {
        std::cerr << "Error: Gradient vector size mismatch. Expected " << numOfVariables_ << std::endl;
        return false;
    }

    gradient = q; // 保存到成员变量
    bool result = solver_.data()->setGradient(gradient);
    return result;
}

bool CuCaQP::setLinearConstraintsMatrix(const Eigen::SparseMatrix<OSQPFloat> &A) {
    // 先清理之前的线性约束矩阵
    solver_.data()->clearLinearConstraintsMatrix();

    if (A.rows() != numOfConstraints_ || A.cols() != numOfVariables_) {
        std::cerr << "Error: Constraint matrix dimensions mismatch. Expected "
                  << numOfConstraints_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    linearConstraintMatrix = A; // 保存到成员变量
    bool result = solver_.data()->setLinearConstraintsMatrix(A);
    return result;
}

bool CuCaQP::setLowerBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &l) {
    if (l.size() != numOfConstraints_) {
        std::cerr << "Error: Lower bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }

    lowerBound = l; // 保存到成员变量
    bool result = solver_.data()->setLowerBound(lowerBound);
    return result;
}

bool CuCaQP::setUpperBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &u) {
    if (u.size() != numOfConstraints_) {
        std::cerr << "Error: Upper bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }

    upperBound = u; // 保存到成员变量
    bool result = solver_.data()->setUpperBound(upperBound);
    return result;
}


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
        linearConstraintMatrix(),
        tripletBuffer() {
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

    // 预分配内存以减少后续操作的开销
    gradient.resize(numOfVariables_);
    lowerBound.resize(numOfConstraints_);
    upperBound.resize(numOfConstraints_);

    // 预分配稀疏矩阵的空间
    hessianMatrix.resize(numOfVariables_, numOfVariables_);
    linearConstraintMatrix.resize(numOfConstraints_, numOfVariables_);

    // 预分配triplet缓冲区，假设平均每行有5个非零元素
    tripletBuffer.reserve(std::max(numOfVariables_, numOfConstraints_) * 5);

    solver_.data()->setNumberOfVariables(numOfVariables_);
    solver_.data()->setNumberOfConstraints(numOfConstraints_);
    return true;
}

bool CuCaQP::setHessianMatrix(const torch::Tensor &hessian) {
    // 先清理之前的Hessian矩阵
    solver_.data()->clearHessianMatrix();
    // 检查维度
    if (hessian.size(0) != numOfVariables_ || hessian.size(1) != numOfVariables_) {
        // 打印接收到的Hessian矩阵维度
        std::cout << "接收到Hessian矩阵: 维度=[" << hessian.size(0) << "x" << hessian.size(1)
                  << "], 期望维度=[" << numOfVariables_ << "x" << numOfVariables_ << "]" << std::endl;
        std::cerr << "Error: Hessian matrix dimensions mismatch. Expected "
                  << numOfVariables_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    // 使用优化的转换函数
    hessianMatrix = torchTensorToEigenSparse<OSQPFloat>(hessian);

    // 设置到求解器
    bool result = solver_.data()->setHessianMatrix(hessianMatrix);
    return result;
}

bool CuCaQP::setGradient(const torch::Tensor &q) {

    // 检查维度
    if (q.numel() != numOfVariables_) {
        // 打印接收到的梯度向量维度
        std::cout << "接收到梯度向量: 元素数=" << q.numel()
                  << ", 期望元素数=" << numOfVariables_ << std::endl;
        std::cerr << "Error: Gradient vector size mismatch. Expected " << numOfVariables_ << std::endl;
        return false;
    }

    // 使用优化的转换函数
    gradient = torchTensorToEigenVector<OSQPFloat>(q);

    // 设置到求解器
    bool result = solver_.data()->setGradient(gradient);
    return result;
}

bool CuCaQP::setLinearConstraintsMatrix(const torch::Tensor &A) {
    // 先清理之前的线性约束矩阵
    solver_.data()->clearLinearConstraintsMatrix();
    // 检查维度
    if (A.size(0) != numOfConstraints_ || A.size(1) != numOfVariables_) {
        // 打印接收到的约束矩阵维度
        std::cout << "接收到约束矩阵: 维度=[" << A.size(0) << "x" << A.size(1)
                  << "], 期望维度=[" << numOfConstraints_ << "x" << numOfVariables_ << "]" << std::endl;
        std::cerr << "Error: Constraint matrix dimensions mismatch. Expected "
                  << numOfConstraints_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    // 使用优化的转换函数
    linearConstraintMatrix = torchTensorToEigenSparse<OSQPFloat>(A);

    // 设置到求解器
    bool result = solver_.data()->setLinearConstraintsMatrix(linearConstraintMatrix);
    return result;
}

bool CuCaQP::setLowerBound(const torch::Tensor &l) {
    // 打印接收到的下界向量维度

    // 检查维度
    if (l.numel() != numOfConstraints_) {
        std::cout << "接收到下界向量: 元素数=" << l.numel()
                  << ", 期望元素数=" << numOfConstraints_ << std::endl;
        std::cerr << "Error: Lower bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }

    // 使用优化的转换函数
    lowerBound = torchTensorToEigenVector<OSQPFloat>(l);

    // 设置到求解器
    bool result = solver_.data()->setLowerBound(lowerBound);
    return result;
}

bool CuCaQP::setUpperBound(const torch::Tensor &u) {
    // 检查维度
    if (u.numel() != numOfConstraints_) {
        std::cout << "接收到上界向量: 元素数=" << u.numel()
                  << ", 期望元素数=" << numOfConstraints_ << std::endl;
        std::cerr << "Error: Upper bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }

    // 使用优化的转换函数
    upperBound = torchTensorToEigenVector<OSQPFloat>(u);

    // 设置到求解器
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
    // 直接返回求解器的解，避免不必要的复制
    return solver_.getSolution();
}

casadi::DM CuCaQP::getSolutionAsDM() {
    // 获取解决方案
    const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &solution_float = solver_.getSolution();

    // 创建CasADi DM矩阵
    int rows = solution_float.rows();
    DM solution = DM::zeros(rows, 1);

    // 填充数据
    for (int i = 0; i < rows; ++i) {
        solution(i) = static_cast<double>(solution_float(i));
    }

    return solution;
}

torch::Tensor CuCaQP::getSolutionAsTensor() {
    // 获取解决方案
    const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &solution_float = solver_.getSolution();
    int rows = solution_float.rows();

    // 创建torch::Tensor，预分配内存
    torch::Tensor solution = torch::zeros({rows}, torch::kFloat32);

    // 获取数据指针并直接填充
    float *data_ptr = solution.data_ptr<float>();
    for (int i = 0; i < rows; ++i) {
        data_ptr[i] = static_cast<float>(solution_float(i));
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

void CuCaQP::setSystem(const std::vector<torch::Tensor> &torchSystem, uint env) {
    // 检查输入向量大小
    if (torchSystem.size() != 5) {
        std::cerr << "Error: Expected 5 tensors in the system vector." << std::endl;
        return;
    }
    setHessianMatrix(torchSystem[0][env]);
    setGradient(torchSystem[1][env]);
    setLinearConstraintsMatrix(torchSystem[2][env]);
    setLowerBound(torchSystem[3][env]);
    setUpperBound(torchSystem[4][env]);
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

    // 直接使用输入矩阵，避免复制
    hessianMatrix = P;
    bool result = solver_.data()->setHessianMatrix(hessianMatrix);
    return result;
}

bool CuCaQP::setGradient(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &q) {
    if (q.size() != numOfVariables_) {
        std::cerr << "Error: Gradient vector size mismatch. Expected " << numOfVariables_ << std::endl;
        return false;
    }

    // 直接使用输入向量，避免复制
    gradient = q;
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

    // 直接使用输入矩阵，避免复制
    linearConstraintMatrix = A;
    bool result = solver_.data()->setLinearConstraintsMatrix(linearConstraintMatrix);
    return result;
}

bool CuCaQP::setLowerBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &l) {
    if (l.size() != numOfConstraints_) {
        std::cerr << "Error: Lower bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }

    // 直接使用输入向量，避免复制
    lowerBound = l;
    bool result = solver_.data()->setLowerBound(lowerBound);
    return result;
}

bool CuCaQP::setUpperBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &u) {
    if (u.size() != numOfConstraints_) {
        std::cerr << "Error: Upper bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }

    // 直接使用输入向量，避免复制
    upperBound = u;
    bool result = solver_.data()->setUpperBound(upperBound);
    return result;
}


#include "sqp_solver/CuCaQP.h"
#include <iostream>
#include <iomanip> // 用于格式化输出

CuCaQP::CuCaQP() : numOfVariables_(0), numOfConstraints_(0), isInitialized_(false) {
}

CuCaQP::~CuCaQP() {
}

bool CuCaQP::setDimension(int numOfVariables, int numOfConstraints) {
    if (numOfVariables <= 0 || numOfConstraints <= 0) {
        std::cerr << "Error: Invalid dimensions." << std::endl;
        return false;
    }
    numOfVariables_ = numOfVariables;
    numOfConstraints_ = numOfConstraints;

    solver_.data()->setNumberOfVariables(numOfVariables_);
    solver_.data()->setNumberOfConstraints(numOfConstraints_);
    return true;
}

bool CuCaQP::setHessianMatrix(const casadi::DM &hessian) {
    hessianMatrix = casadiDMToEigenSparse<OSQPFloat>(hessian);
    if (hessianMatrix.rows() != numOfVariables_ || hessianMatrix.cols() != numOfVariables_) {
        std::cerr << "Error: Hessian matrix dimensions mismatch. Expected "
                  << numOfVariables_ << "x" << numOfVariables_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setHessianMatrix(hessianMatrix);
    return result;
}
bool CuCaQP::setGradient(const casadi::DM &q) {
    gradient = casadiDMToEigenVector<OSQPFloat>(q);

    if (gradient.size() != numOfVariables_) {
        std::cerr << "Error: Gradient vector size mismatch. Expected " << numOfVariables_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setGradient(gradient);
    return result;
}
bool CuCaQP::setLinearConstraintsMatrix(const DM &A) {
    auto constraintMatrix = casadiDMToEigenSparse<OSQPFloat>(A);

    if (constraintMatrix.rows() != numOfConstraints_ || constraintMatrix.cols() != numOfVariables_) {
        std::cerr << "Error: Constraint matrix dimensions mismatch. Expected "
                  << numOfConstraints_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    bool result = solver_.data()->setLinearConstraintsMatrix(constraintMatrix);
    return result;
}

bool CuCaQP::setLowerBound(casadi::DM &l) {
    lowerBound = casadiDMToEigenVector<OSQPFloat>(l);
    if (lowerBound.size() != numOfConstraints_) {
        std::cerr << "Error: Lower bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setLowerBound(lowerBound);
    return result;
}

bool CuCaQP::setUpperBound(const casadi::DM &u) {
    upperBound = casadiDMToEigenVector<OSQPFloat>(u);

    if (upperBound.size() != numOfConstraints_) {
        std::cerr << "Error: Upper bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    bool result = solver_.data()->setUpperBound(upperBound);
    return result;
}


bool CuCaQP::updateHessianMatrix(const casadi::DM &hessian) {
    hessianMatrix = casadiDMToEigenSparse<OSQPFloat>(hessian);

    if (hessianMatrix.rows() != numOfVariables_ || hessianMatrix.cols() != numOfVariables_) {
        std::cerr << "Error: Hessian matrix dimensions mismatch. Expected "
                  << numOfVariables_ << "x" << numOfVariables_ << std::endl;
        return false;
    }
    bool result = solver_.updateHessianMatrix(hessianMatrix);
    return result;
}
bool CuCaQP::updateGradient(const casadi::DM &q) {
    gradient = casadiDMToEigenVector<OSQPFloat>(q);

    if (gradient.size() != numOfVariables_) {
        std::cerr << "Error: Gradient vector size mismatch. Expected " << numOfVariables_ << std::endl;
        return false;
    }
    bool result = solver_.updateGradient(gradient);
    return result;
}
bool CuCaQP::updateLinearConstraintsMatrix(const DM &A) {
    linearConstraintMatrix = casadiDMToEigenSparse<OSQPFloat>(A);

    if (linearConstraintMatrix.rows() != numOfConstraints_ || linearConstraintMatrix.cols() != numOfVariables_) {
        std::cerr << "Error: Constraint matrix dimensions mismatch. Expected "
                  << numOfConstraints_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    bool result = solver_.updateLinearConstraintsMatrix(linearConstraintMatrix);
    return result;
}

bool CuCaQP::updateLowerBound(casadi::DM &l) {
    lowerBound = casadiDMToEigenVector<OSQPFloat>(l);
    if (lowerBound.size() != numOfConstraints_) {
        std::cerr << "Error: Lower bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    bool result = solver_.updateLowerBound(lowerBound);
    return result;
}

bool CuCaQP::updateUpperBound(const casadi::DM &u) {
    upperBound = casadiDMToEigenVector<OSQPFloat>(u);

    if (upperBound.size() != numOfConstraints_) {
        std::cerr << "Error: Upper bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    bool result = solver_.updateUpperBound(upperBound);
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
    solver_.clearSolver();
    solver_.data()->clearHessianMatrix();
    solver_.data()->clearLinearConstraintsMatrix();
    return solution;
}

void CuCaQP::printSolverData() {
    // 直接访问OSQP内部数据
    const auto* osqpData = solver_.data()->getData();

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
    // 这里只打印非零元素
    std::cout << "内部存储的Hessian矩阵P (非零元素):\n";
    for (int j = 0; j < osqpData->n; j++) {
        for (int p = osqpData->P->p[j]; p < osqpData->P->p[j+1]; p++) {
            int i = osqpData->P->i[p];
            c_float val = osqpData->P->x[p];
            std::cout << "(" << i << "," << j << "): " << val << std::endl;
        }
    }

    std::cout << "内部存储的约束矩阵A (非零元素):\n";
    for (int j = 0; j < osqpData->n; j++) {
        for (int p = osqpData->A->p[j]; p < osqpData->A->p[j+1]; p++) {
            int i = osqpData->A->i[p];
            c_float val = osqpData->A->x[p];
            std::cout << "(" << i << "," << j << "): " << val << std::endl;
        }
    }
}

void CuCaQP::setSystem(DMVector localSystem) {
    static bool firstTime{true};
    if(firstTime){
        setHessianMatrix(localSystem[0]);
        setGradient(localSystem[1]);
        setLinearConstraintsMatrix(localSystem[2]);
        setLowerBound(localSystem[3]);
        setUpperBound(localSystem[4]);
//        std::cout<<"=== 第一次设置得到的参数 === \n";
//        const auto& hessianMatrix = solver_.data()->getData()->P;
//        const auto& gradientVector = solver_.data()->getData()->q;
//        const auto& constraintMatrix = solver_.data()->getData()->A;
//        const auto& lowerBound = solver_.data()->getData()->l;
//        const auto& upperBound = solver_.data()->getData()->u;

//        firstTime = false;
    } else{
        updateHessianMatrix(localSystem[0]);
        updateGradient(localSystem[1]);
        updateLinearConstraintsMatrix(localSystem[2]);
        updateLowerBound(localSystem[3]);
        updateUpperBound(localSystem[4]);

        std::cout<<"=== 更新后测参数 === \n";
        /*
         *
         *     OSQPCscMatrix* P; ///< the upper triangular part of the quadratic objective matrix P (size n x n).
    OSQPCscMatrix* A; ///< linear constraints matrix A (size m x n)
    OSQPFloat* q; ///< dense array for linear part of objective function (size n)
    OSQPFloat* l; ///< dense array for lower bound (size m)
    OSQPFloat* u; ///< dense array for upper bound (size m)*/
        std::cout<<"Hessian : "<<solver_.data()->getData()->P;
        std::cout<<"Gradient : "<<solver_.data()->getData()->q;
        std::cout<<"A : "<<solver_.data()->getData()->A;
        std::cout<<"l : "<<solver_.data()->getData()->l;
        std::cout<<"u : "<<solver_.data()->getData()->u<<std::endl;
    }
}


//##################################### 暂时用不上的函数 ##################################################
bool CuCaQP::setHessianMatrix(const Eigen::SparseMatrix<OSQPFloat> &P) {
    if (P.rows() != numOfVariables_ || P.cols() != numOfVariables_) {
        std::cerr << "Error: Hessian matrix dimensions mismatch. Expected "
                  << numOfVariables_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    bool result = solver_.data()->setHessianMatrix(P);
    return result;
}
bool CuCaQP::setGradient(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &q) {
    if (q.size() != numOfVariables_) {
        std::cerr << "Error: Gradient vector size mismatch. Expected " << numOfVariables_ << std::endl;
        return false;
    }

    Eigen::VectorXf gradientCopy = q;
    bool result = solver_.data()->setGradient(gradientCopy);
    return result;
}

bool CuCaQP::setLinearConstraintsMatrix(const Eigen::SparseMatrix<OSQPFloat> &A) {
    if (A.rows() != numOfConstraints_ || A.cols() != numOfVariables_) {
        std::cerr << "Error: Constraint matrix dimensions mismatch. Expected "
                  << numOfConstraints_ << "x" << numOfVariables_ << std::endl;
        return false;
    }

    bool result = solver_.data()->setLinearConstraintsMatrix(A);
    return result;
}

bool CuCaQP::setLowerBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &l) {
    if (l.size() != numOfConstraints_) {
        std::cerr << "Error: Lower bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }

    Eigen::VectorXf lowerBoundCopy = l;
    bool result = solver_.data()->setLowerBound(lowerBoundCopy);
    return result;
}

bool CuCaQP::setUpperBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &u) {
    if (u.size() != numOfConstraints_) {
        std::cerr << "Error: Upper bound vector size mismatch. Expected " << numOfConstraints_ << std::endl;
        return false;
    }
    Eigen::VectorXf upperBoundCopy = u;
    bool result = solver_.data()->setUpperBound(upperBoundCopy);
    return result;
}
// CuCaQP.h
#ifndef CUCAQP_H
#define CUCAQP_H

#include <OsqpEigen/OsqpEigen.h>
#include "casadi/casadi.hpp"
#include "torch/script.h"
#include <eigen3/Eigen/Dense>

using namespace casadi;

/**
 * 将CasADi DM稀疏矩阵转换为Eigen稀疏矩阵
 * @param casadiMatrix CasADi DM稀疏矩阵
 * @return 转换后的Eigen稀疏矩阵
 */
template<typename T>
Eigen::SparseMatrix<T> casadiDMToEigenSparse(const casadi::DM &casadiMatrix);

/**
 * 将CasADi DM向量转换为Eigen密集向量
 * @param casadiVector CasADi DM向量
 * @return 转换后的Eigen密集向量
 */
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> casadiDMToEigenVector(const casadi::DM &casadiVector);

class CuCaQP {
public:
    CuCaQP();

    ~CuCaQP();

    // 设置问题维度
    bool setDimension(int numOfVariables, int numOfConstraints);

    // 设置优化问题数据
    bool setHessianMatrix(const Eigen::SparseMatrix<OSQPFloat> &P);

    bool setHessianMatrix(const casadi::DM &hessian);

    bool setGradient(const DM &q);

    bool setLinearConstraintsMatrix(const DM &A);

    bool setLowerBound(const DM &l);

    bool setUpperBound(const DM &u);

    bool setGradient(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &q);

    bool setLinearConstraintsMatrix(const Eigen::SparseMatrix<OSQPFloat> &A);

    bool setLowerBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &l);

    bool setUpperBound(const Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> &u);

    // 设置求解器参数
    void setVerbosity(bool verbosity);

    void setWarmStart(bool warmStart);

    void setAbsoluteTolerance(OSQPFloat tolerance);

    void setRelativeTolerance(OSQPFloat tolerance);

    void setMaxIteration(int maxIteration);

    // 初始化并求解
    bool initSolver();

    void printSolverData();

    bool solve();

    // 获取结果
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> getSolution();

    DM getSolutionAsDM();

    void setSystem(DMVector vector1);

private:
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> upperBound;
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> lowerBound;
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> gradient;
    Eigen::SparseMatrix<OSQPFloat> hessianMatrix;
    Eigen::SparseMatrix<OSQPFloat> linearConstraintMatrix;
    OsqpEigen::Solver solver_;
    int numOfVariables_;
    int numOfConstraints_;
    bool isInitialized_;

    bool updateHessianMatrix(const DM &hessian);

    bool updateGradient(const DM &q);

    bool updateLinearConstraintsMatrix(const DM &A);

    bool updateLowerBound(const DM &l);

    bool updateUpperBound(const DM &u);
};

// 模板函数实现放在头文件末尾，避免链接错误
template<typename T>
Eigen::SparseMatrix<T> casadiDMToEigenSparse(const casadi::DM &casadiMatrix) {
    int rows = casadiMatrix.size1();
    int cols = casadiMatrix.size2();

    Eigen::SparseMatrix<T> eigenMatrix(rows, cols);
    std::vector<Eigen::Triplet<T>> triplets;

    // 获取CasADi矩阵的稀疏结构
    casadi::Sparsity sparsity = casadiMatrix.sparsity();

    // 修正：使用正确的方法获取列索引和行索引
    // 获取列指针数组 (CSC格式)
    const casadi_int *colind = sparsity.colind();
    // 获取行索引数组
    const casadi_int *row_indices = sparsity.row();
    // 获取非零元素值
    std::vector<double> data = casadiMatrix.nonzeros();

    // 收集非零元素
    for (casadi_int j = 0; j < sparsity.size2(); ++j) {
        for (casadi_int k = colind[j]; k < colind[j + 1]; ++k) {
            casadi_int i = row_indices[k];
            triplets.push_back(Eigen::Triplet<T>(i, j, static_cast<T>(data[k])));
        }
    }

    // 使用triplets填充稀疏矩阵
    eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    eigenMatrix.makeCompressed();

    return eigenMatrix;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> casadiDMToEigenVector(const casadi::DM &casadiVector) {
    // 获取向量大小
    int size = casadiVector.size1();

    // 创建 Eigen 向量
    Eigen::Matrix<T, Eigen::Dynamic, 1> eigenVector(size);

    // 先使用循环填充数据
    for (int i = 0; i < size; i++) {
        eigenVector(i) = static_cast<T>(casadiVector(i).scalar());
    }
    return eigenVector;
}

#endif // CUCAQP_H

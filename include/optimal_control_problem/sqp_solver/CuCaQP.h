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

/**
 * 将torch::Tensor稀疏矩阵转换为Eigen稀疏矩阵
 * @param torchMatrix torch::Tensor稀疏矩阵
 * @return 转换后的Eigen稀疏矩阵
 */
template<typename T>
Eigen::SparseMatrix<T> torchTensorToEigenSparse(const torch::Tensor &torchMatrix);

/**
 * 将torch::Tensor向量转换为Eigen密集向量
 * @param torchVector torch::Tensor向量
 * @return 转换后的Eigen密集向量
 */
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> torchTensorToEigenVector(const torch::Tensor &torchVector);

class CuCaQP {
public:
    CuCaQP();

    ~CuCaQP();

    // 设置问题维度
    bool setDimension(int numOfVariables, int numOfConstraints);

    // 设置优化问题数据 - CasADi DM版本
    bool setHessianMatrix(const casadi::DM &hessian);
    bool setGradient(const DM &q);
    bool setLinearConstraintsMatrix(const DM &A);
    bool setLowerBound(const DM &l);
    bool setUpperBound(const DM &u);

    // 设置优化问题数据 - torch::Tensor版本
    bool setHessianMatrix(const torch::Tensor &hessian);
    bool setGradient(const torch::Tensor &q);
    bool setLinearConstraintsMatrix(const torch::Tensor &A);
    bool setLowerBound(const torch::Tensor &l);
    bool setUpperBound(const torch::Tensor &u);

    // 设置优化问题数据 - Eigen版本
    bool setHessianMatrix(const Eigen::SparseMatrix<OSQPFloat> &P);
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
    torch::Tensor getSolutionAsTensor();

    // 设置系统 - CasADi版本
    void setSystem(DMVector vector1);

    // 设置系统 - torch::Tensor版本
    void setSystem(const std::vector<torch::Tensor> &torchSystem);

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

    // 更新函数 - CasADi版本
    bool updateHessianMatrix(const DM &hessian);
    bool updateGradient(const DM &q);
    bool updateLinearConstraintsMatrix(const DM &A);
    bool updateLowerBound(const DM &l);
    bool updateUpperBound(const DM &u);

    // 更新函数 - torch::Tensor版本
    bool updateHessianMatrix(const torch::Tensor &hessian);
    bool updateGradient(const torch::Tensor &q);
    bool updateLinearConstraintsMatrix(const torch::Tensor &A);
    bool updateLowerBound(const torch::Tensor &l);
    bool updateUpperBound(const torch::Tensor &u);
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

template<typename T>
Eigen::SparseMatrix<T> torchTensorToEigenSparse(const torch::Tensor &torchMatrix) {
    // 确保输入是CPU张量并且是2D的
    auto cpuTensor = torchMatrix.to(torch::kCPU);

    int rows = cpuTensor.size(0);
    int cols = cpuTensor.size(1);

    Eigen::SparseMatrix<T> eigenMatrix(rows, cols);
    std::vector<Eigen::Triplet<T>> triplets;

    // 如果是稠密张量，转换为稀疏表示
    if (!cpuTensor.is_sparse()) {
        // 对于稠密张量，遍历所有元素，只添加非零元素
        auto accessor = cpuTensor.accessor<float, 2>();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float val = accessor[i][j];
                if (std::abs(val) > 1e-10) { // 设置一个小的阈值来判断非零元素
                    triplets.push_back(Eigen::Triplet<T>(i, j, static_cast<T>(val)));
                }
            }
        }
    } else {
        // 对于稀疏张量，直接使用其稀疏表示
        auto indices = cpuTensor._indices();
        auto values = cpuTensor._values();

        auto indicesAccessor = indices.accessor<int64_t, 2>();
        auto valuesAccessor = values.accessor<float, 1>();

        for (int k = 0; k < values.size(0); ++k) {
            int i = indicesAccessor[0][k];
            int j = indicesAccessor[1][k];
            float val = valuesAccessor[k];
            triplets.push_back(Eigen::Triplet<T>(i, j, static_cast<T>(val)));
        }
    }

    eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    eigenMatrix.makeCompressed();

    return eigenMatrix;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> torchTensorToEigenVector(const torch::Tensor &torchVector) {
    // 确保输入是CPU张量
    auto cpuTensor = torchVector.to(torch::kCPU).contiguous();

    // 确保张量是一维的
    if (cpuTensor.dim() > 1) {
        // 如果是2D张量且一个维度为1，则压缩为1D
        if (cpuTensor.dim() == 2 && (cpuTensor.size(0) == 1 || cpuTensor.size(1) == 1)) {
            cpuTensor = cpuTensor.squeeze();
        } else {
            throw std::runtime_error("Input tensor must be a vector (1D tensor)");
        }
    }

    int size = cpuTensor.size(0);

    // 创建Eigen向量
    Eigen::Matrix<T, Eigen::Dynamic, 1> eigenVector(size);

    // 填充数据
    auto accessor = cpuTensor.accessor<float, 1>();
    for (int i = 0; i < size; ++i) {
        eigenVector(i) = static_cast<T>(accessor[i]);
    }

    return eigenVector;
}

#endif // CUCAQP_H

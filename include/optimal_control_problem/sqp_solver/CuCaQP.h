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

    // 处理稀疏张量
    if (cpuTensor.is_sparse()) {
        auto indices = cpuTensor._indices();
        auto values = cpuTensor._values();

        // 直接获取数据指针，避免使用accessor
        int64_t* indicesPtr = indices.data_ptr<int64_t>();
        float* valuesPtr = values.data_ptr<float>();

        int nnz = values.size(0);
        std::vector<Eigen::Triplet<T>> triplets(nnz);

        for (int k = 0; k < nnz; ++k) {
            int i = indicesPtr[k];
            int j = indicesPtr[k + nnz]; // 第二行索引
            triplets[k] = Eigen::Triplet<T>(i, j, static_cast<T>(valuesPtr[k]));
        }

        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }
        // 处理稠密张量 - 使用更高效的方法
    else {
        // 将稠密张量转换为稀疏格式
        auto sparseTensor = cpuTensor.to_sparse();
        auto indices = sparseTensor._indices();
        auto values = sparseTensor._values();

        // 直接获取数据指针
        int64_t* indicesPtr = indices.data_ptr<int64_t>();
        float* valuesPtr = values.data_ptr<float>();

        int nnz = values.size(0);
        std::vector<Eigen::Triplet<T>> triplets(nnz);

        for (int k = 0; k < nnz; ++k) {
            int i = indicesPtr[k];
            int j = indicesPtr[k + nnz]; // 第二行索引
            triplets[k] = Eigen::Triplet<T>(i, j, static_cast<T>(valuesPtr[k]));
        }

        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    eigenMatrix.makeCompressed();
    return eigenMatrix;
}


template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> torchTensorToEigenVector(const torch::Tensor &torchVector) {
    // 确保输入是CPU张量
    auto cpuTensor = torchVector.to(torch::kCPU).contiguous();

    // 获取向量大小
    int size = cpuTensor.size(0);

    // 创建Eigen向量
    Eigen::Matrix<T, Eigen::Dynamic, 1> eigenVector(size);

    // 直接复制内存
    if (cpuTensor.dtype() == torch::kFloat32) {
        float* dataPtr = cpuTensor.data_ptr<float>();
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> eigenMap(dataPtr, size);
        eigenVector = eigenMap.cast<T>();
    }
    else if (cpuTensor.dtype() == torch::kFloat64) {
        double* dataPtr = cpuTensor.data_ptr<double>();
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> eigenMap(dataPtr, size);
        eigenVector = eigenMap.cast<T>();
    }
    else {
        // 对于其他数据类型，使用逐元素复制
        auto accessor = cpuTensor.accessor<float, 1>();
        for (int i = 0; i < size; ++i) {
            eigenVector(i) = static_cast<T>(accessor[i]);
        }
    }

    return eigenVector;
}

#endif // CUCAQP_H

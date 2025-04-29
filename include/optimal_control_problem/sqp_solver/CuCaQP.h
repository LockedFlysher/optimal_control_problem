// CuCaQP.h
#ifndef CUCAQP_H
#define CUCAQP_H

#include <OsqpEigen/OsqpEigen.h>
#include "casadi/casadi.hpp"
#include "torch/script.h"
#include <eigen3/Eigen/Dense>
#include <type_traits>

using namespace casadi;

/**
 * 将CasADi DM稀疏矩阵转换为Eigen稀疏矩阵
 * @param casadiMatrix CasADi DM稀疏矩阵
 * @return 转换后的Eigen稀疏矩阵
 * @tparam T_Out Eigen矩阵的数据类型
 * @tparam T_In CasADi矩阵的数据类型（默认为double）
 */
template<typename T_Out = OSQPFloat, typename T_In = double>
inline Eigen::SparseMatrix<T_Out> casadiDMToEigenSparse(const casadi::DM &casadiMatrix) {
    int rows = casadiMatrix.size1();
    int cols = casadiMatrix.size2();

    Eigen::SparseMatrix<T_Out> eigenMatrix(rows, cols);
    std::vector<Eigen::Triplet<T_Out>> triplets;

    // 预分配空间以避免重新分配
    triplets.reserve(casadiMatrix.nnz());

    // 获取CasADi矩阵的稀疏结构
    casadi::Sparsity sparsity = casadiMatrix.sparsity();

    // 获取列指针数组 (CSC格式)
    const casadi_int *colind = sparsity.colind();
    // 获取行索引数组
    const casadi_int *row_indices = sparsity.row();
    // 获取非零元素值
    std::vector<T_In> data = casadiMatrix.nonzeros();

    // 收集非零元素
    for (casadi_int j = 0; j < sparsity.size2(); ++j) {
        for (casadi_int k = colind[j]; k < colind[j + 1]; ++k) {
            casadi_int i = row_indices[k];
            triplets.emplace_back(i, j, static_cast<T_Out>(data[k]));
        }
    }

    // 使用triplets填充稀疏矩阵
    eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    eigenMatrix.makeCompressed();

    return eigenMatrix;
}

/**
 * 将CasADi DM向量转换为Eigen密集向量
 * @param casadiVector CasADi DM向量
 * @return 转换后的Eigen密集向量
 * @tparam T_Out Eigen向量的数据类型
 * @tparam T_In CasADi向量的数据类型（默认为double）
 */
template<typename T_Out = OSQPFloat, typename T_In = double>
inline Eigen::Matrix<T_Out, Eigen::Dynamic, 1> casadiDMToEigenVector(const casadi::DM &casadiVector) {
    // 获取向量大小
    int size = casadiVector.size1();

    // 创建 Eigen 向量
    Eigen::Matrix<T_Out, Eigen::Dynamic, 1> eigenVector(size);

    // 获取DM的数据指针（如果可能）
    const std::vector<T_In>& data = casadiVector.nonzeros();

    // 如果是稠密向量，可以直接复制
    if (casadiVector.is_dense()) {
        for (int i = 0; i < size; i++) {
            eigenVector(i) = static_cast<T_Out>(data[i]);
        }
    } else {
        // 稀疏向量需要按索引复制
        casadi::Sparsity sp = casadiVector.sparsity();
        const casadi_int* row = sp.row();
        const casadi_int* colind = sp.colind();

        // 先将所有元素置为0
        eigenVector.setZero();

        // 只复制非零元素
        for (casadi_int j = 0; j < 1; j++) {  // 向量只有一列
            for (casadi_int k = colind[j]; k < colind[j + 1]; k++) {
                casadi_int i = row[k];
                eigenVector(i) = static_cast<T_Out>(data[k]);
            }
        }
    }

    return eigenVector;
}

/**
 * 将torch::Tensor稀疏矩阵转换为Eigen稀疏矩阵
 * @param torchMatrix torch::Tensor稀疏矩阵
 * @return 转换后的Eigen稀疏矩阵
 * @tparam T_Out Eigen矩阵的数据类型
 * @tparam T_In torch张量的数据类型（默认根据张量类型自动推断）
 */
template<typename T_Out = OSQPFloat, typename T_In = float>
inline Eigen::SparseMatrix<T_Out> torchTensorToEigenSparse(const torch::Tensor &torchMatrix) {
    // 确保输入是CPU张量并且是2D的
    auto cpuTensor = torchMatrix.to(torch::kCPU);

    int rows = cpuTensor.size(0);
    int cols = cpuTensor.size(1);

    Eigen::SparseMatrix<T_Out> eigenMatrix(rows, cols);

    // 处理稀疏张量
    if (cpuTensor.is_sparse()) {
        auto indices = cpuTensor._indices();
        auto values = cpuTensor._values();

        // 直接获取数据指针，避免使用accessor
        int64_t* indicesPtr = indices.data_ptr<int64_t>();

        int nnz = values.size(0);
        std::vector<Eigen::Triplet<T_Out>> triplets;
        triplets.reserve(nnz);  // 预分配空间

        // 根据张量类型选择适当的数据指针
        if (values.dtype() == torch::kFloat32) {
            float* valuesPtr = values.data_ptr<float>();
            for (int k = 0; k < nnz; ++k) {
                int i = indicesPtr[k];
                int j = indicesPtr[k + nnz]; // 第二行索引
                triplets.emplace_back(i, j, static_cast<T_Out>(valuesPtr[k]));
            }
        } else if (values.dtype() == torch::kFloat64) {
            double* valuesPtr = values.data_ptr<double>();
            for (int k = 0; k < nnz; ++k) {
                int i = indicesPtr[k];
                int j = indicesPtr[k + nnz]; // 第二行索引
                triplets.emplace_back(i, j, static_cast<T_Out>(valuesPtr[k]));
            }
        }

        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }
        // 处理稠密张量
    else {
        // 将稠密张量转换为稀疏格式
        auto sparseTensor = cpuTensor.to_sparse();
        auto indices = sparseTensor._indices();
        auto values = sparseTensor._values();

        // 直接获取数据指针
        int64_t* indicesPtr = indices.data_ptr<int64_t>();

        int nnz = values.size(0);
        std::vector<Eigen::Triplet<T_Out>> triplets;
        triplets.reserve(nnz);  // 预分配空间

        // 根据张量类型选择适当的数据指针
        if (values.dtype() == torch::kFloat32) {
            float* valuesPtr = values.data_ptr<float>();
            for (int k = 0; k < nnz; ++k) {
                int i = indicesPtr[k];
                int j = indicesPtr[k + nnz]; // 第二行索引
                triplets.emplace_back(i, j, static_cast<T_Out>(valuesPtr[k]));
            }
        } else if (values.dtype() == torch::kFloat64) {
            double* valuesPtr = values.data_ptr<double>();
            for (int k = 0; k < nnz; ++k) {
                int i = indicesPtr[k];
                int j = indicesPtr[k + nnz]; // 第二行索引
                triplets.emplace_back(i, j, static_cast<T_Out>(valuesPtr[k]));
            }
        }

        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    eigenMatrix.makeCompressed();
    return eigenMatrix;
}

/**
 * 将torch::Tensor向量转换为Eigen密集向量
 * @param torchVector torch::Tensor向量
 * @return 转换后的Eigen密集向量
 * @tparam T_Out Eigen向量的数据类型
 */
template<typename T_Out = OSQPFloat>
inline Eigen::Matrix<T_Out, Eigen::Dynamic, 1> torchTensorToEigenVector(const torch::Tensor &torchVector) {
    // 确保输入是CPU张量且内存连续
    auto cpuTensor = torchVector.to(torch::kCPU).contiguous();

    // 获取向量大小
    int size = cpuTensor.size(0);

    // 创建Eigen向量
    Eigen::Matrix<T_Out, Eigen::Dynamic, 1> eigenVector(size);

    // 根据张量类型选择适当的转换方法
    if (cpuTensor.dtype() == torch::kFloat32) {
        float* dataPtr = cpuTensor.data_ptr<float>();
        if (std::is_same<T_Out, float>::value) {
            // 如果输出类型也是float，可以直接映射内存
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> eigenMap(dataPtr, size);
            eigenVector = eigenMap.template cast<T_Out>();
        } else {
            // 需要类型转换
            for (int i = 0; i < size; ++i) {
                eigenVector(i) = static_cast<T_Out>(dataPtr[i]);
            }
        }
    }
    else if (cpuTensor.dtype() == torch::kFloat64) {
        double* dataPtr = cpuTensor.data_ptr<double>();
        if (std::is_same<T_Out, double>::value) {
            // 如果输出类型也是double，可以直接映射内存
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> eigenMap(dataPtr, size);
            eigenVector = eigenMap.template cast<T_Out>();
        } else {
            // 需要类型转换
            for (int i = 0; i < size; ++i) {
                eigenVector(i) = static_cast<T_Out>(dataPtr[i]);
            }
        }
    }
    else {
        // 对于其他数据类型，使用逐元素复制
        auto accessor = cpuTensor.accessor<float, 1>();
        for (int i = 0; i < size; ++i) {
            eigenVector(i) = static_cast<T_Out>(accessor[i]);
        }
    }

    return eigenVector;
}

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
    // 缓存的数据结构，避免重复分配内存
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> upperBound;
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> lowerBound;
    Eigen::Matrix<OSQPFloat, Eigen::Dynamic, 1> gradient;
    Eigen::SparseMatrix<OSQPFloat> hessianMatrix;
    Eigen::SparseMatrix<OSQPFloat> linearConstraintMatrix;

    // 用于稀疏矩阵转换的缓冲区，避免重复分配
    std::vector<Eigen::Triplet<OSQPFloat>> tripletBuffer;

    OsqpEigen::Solver solver_;
    int numOfVariables_;
    int numOfConstraints_;
    bool isInitialized_;
};

#endif // CUCAQP_H

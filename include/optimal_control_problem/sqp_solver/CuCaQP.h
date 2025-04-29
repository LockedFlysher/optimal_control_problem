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
 * 将torch::Tensor稀疏矩阵转换为Eigen稀疏矩阵
 * @param torchMatrix torch::Tensor稀疏矩阵
 * @param n 指定要提取的第n个矩阵（在第一维上的索引）
 * @return 转换后的Eigen稀疏矩阵
 * @tparam T_Out Eigen矩阵的数据类型
 * @tparam T_In torch张量的数据类型（默认根据张量类型自动推断）
 */
template<typename T_Out = OSQPFloat, typename T_In = float>
inline Eigen::SparseMatrix<T_Out> torchTensorToEigenSparse(const torch::Tensor &torchMatrix, int n = 0) {
    // 确保输入是CPU张量
    auto cpuTensor = torchMatrix.to(torch::kCPU);

    // 处理3D张量情况
    if (cpuTensor.dim() == 3) {
        // 确保n在有效范围内
        if (n < 0 || n >= cpuTensor.size(0)) {
            throw std::out_of_range("Index n is out of range for the tensor's first dimension");
        }

        // 提取指定的2D切片
        cpuTensor = cpuTensor.select(0, n);
    } else if (cpuTensor.dim() != 2) {
        throw std::invalid_argument("Input tensor must be 2D or 3D");
    }

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
 * 将torch::Tensor密集矩阵转换为Eigen密集矩阵
 * @param torchMatrix torch::Tensor密集矩阵
 * @param n 指定要提取的第n个矩阵（在第一维上的索引）
 * @return 转换后的Eigen密集矩阵
 * @tparam T_Out Eigen矩阵的数据类型
 */
template<typename T_Out = OSQPFloat>
inline Eigen::Matrix<T_Out, Eigen::Dynamic, Eigen::Dynamic> torchTensorToEigenDense(const torch::Tensor &torchMatrix, int n = 0) {
    // 确保输入是CPU张量且内存连续
    auto cpuTensor = torchMatrix.to(torch::kCPU);

    // 处理3D张量情况
    if (cpuTensor.dim() == 3) {
        // 确保n在有效范围内
        if (n < 0 || n >= cpuTensor.size(0)) {
            throw std::out_of_range("Index n is out of range for the tensor's first dimension");
        }

        // 提取指定的2D切片
        cpuTensor = cpuTensor.select(0, n);
    } else if (cpuTensor.dim() != 2) {
        throw std::invalid_argument("Input tensor must be 2D or 3D");
    }

    cpuTensor = cpuTensor.contiguous();

    // 获取矩阵维度
    int rows = cpuTensor.size(0);
    int cols = cpuTensor.size(1);

    // 创建Eigen矩阵
    Eigen::Matrix<T_Out, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix(rows, cols);

    // 根据张量类型选择适当的转换方法
    if (cpuTensor.dtype() == torch::kFloat32) {
        float* dataPtr = cpuTensor.data_ptr<float>();
        if (std::is_same<T_Out, float>::value) {
            // 如果输出类型也是float，可以直接映射内存
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigenMap(dataPtr, rows, cols);
            eigenMatrix = eigenMap.template cast<T_Out>();
        } else {
            // 需要类型转换
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    eigenMatrix(i, j) = static_cast<T_Out>(dataPtr[i * cols + j]);
                }
            }
        }
    }
    else if (cpuTensor.dtype() == torch::kFloat64) {
        double* dataPtr = cpuTensor.data_ptr<double>();
        if (std::is_same<T_Out, double>::value) {
            // 如果输出类型也是double，可以直接映射内存
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigenMap(dataPtr, rows, cols);
            eigenMatrix = eigenMap.template cast<T_Out>();
        } else {
            // 需要类型转换
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    eigenMatrix(i, j) = static_cast<T_Out>(dataPtr[i * cols + j]);
                }
            }
        }
    }
    else {
        // 对于其他数据类型，使用逐元素复制
        auto accessor = cpuTensor.accessor<float, 2>();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                eigenMatrix(i, j) = static_cast<T_Out>(accessor[i][j]);
            }
        }
    }

    return eigenMatrix;
}
/**
 * 将torch::Tensor向量转换为Eigen密集向量
 * @param torchVector torch::Tensor向量
 * @param n 指定要提取的第n个向量（在第一维上的索引，对于3D张量）
 * @return 转换后的Eigen密集向量
 * @tparam T_Out Eigen向量的数据类型
 */
template<typename T_Out = OSQPFloat>
inline Eigen::Matrix<T_Out, Eigen::Dynamic, 1> torchTensorToEigenVector(const torch::Tensor &torchVector, int n = 0) {
    // 确保输入是CPU张量，并且是密集张量
    auto cpuTensor = torchVector.to(torch::kCPU);

    // 如果是稀疏张量，直接转换为密集张量
    if (cpuTensor.is_sparse()) {
        cpuTensor = cpuTensor.to_dense();
    }

    // 处理多维张量情况
    if (cpuTensor.dim() > 1) {
        // 如果是3D张量，先提取指定的2D切片
        if (cpuTensor.dim() == 3) {
            // 确保n在有效范围内
            if (n < 0 || n >= cpuTensor.size(0)) {
                throw std::out_of_range("Index n is out of range for the tensor's first dimension");
            }

            // 提取指定的2D切片
            cpuTensor = cpuTensor.select(0, n);
        }

        // 如果是2D张量，将其展平为1D
        if (cpuTensor.dim() == 2) {
            cpuTensor = cpuTensor.flatten();
        } else if (cpuTensor.dim() > 2) {
            throw std::invalid_argument("Input tensor must be 1D, 2D or 3D");
        }
    }

    cpuTensor = cpuTensor.contiguous();

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

    // 设置优化问题数据 - torch::Tensor版本
    // 设置系统 - torch::Tensor版本
    void setSystem(const std::vector<torch::Tensor> &torchSystem,uint env=0);

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

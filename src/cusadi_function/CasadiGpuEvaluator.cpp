//
// Created by lock on 25-4-28.
// Updated by Monica on 25-5-8 to remove multi-instance support, rename APIs, and add CPU support.
//
#include "optimal_control_problem/cusadi_function/CasadiGpuEvaluator.h"
#include <iostream>
#include <cuda_runtime.h>  // 用于 cudaDeviceSynchronize 等 CUDA 运行时函数
#include <sstream>  // 用于构建详细的错误信息

// 辅助宏，用于检查CUDA错误并提供详细信息
#define CHECK_CUDA_ERROR(call, msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::stringstream error_ss; \
            error_ss << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: " \
                     << msg << " - " << cudaGetErrorString(err); \
            std::cerr << error_ss.str() << std::endl; \
            return; /* 避免程序崩溃，返回当前函数 */ \
        } \
    } while(0)

/**
 * @brief 构造函数，初始化 CasadiGpuEvaluator 对象
 * @param fn_casadi CasADi 函数对象
 * @param use_cuda 是否使用 CUDA 加速，true 为 GPU，false 为 CPU
 */
CasadiGpuEvaluator::CasadiGpuEvaluator(const casadi::Function &fn_casadi)
        : fn_(fn_casadi), lib_handle_(nullptr), eval_fn_(nullptr) {

    // 加载动态库，无论是 CPU 还是 GPU 模式都需要
    std::string path = ament_index_cpp::get_package_share_directory("optimal_control_problem") +
                       "/cusadi/build/liblocalSystemFunction.so";
    lib_handle_ = dlopen(path.c_str(), RTLD_LAZY);
    if (!lib_handle_) {
        std::cerr << "动态库加载失败: " << dlerror() << std::endl;
        std::cerr << "尝试加载的路径: " << path << std::endl;
        std::exit(1);
    }

    // 获取评估函数指针
    eval_fn_ = reinterpret_cast<EvalFn>(dlsym(lib_handle_, "evaluate"));
    if (!eval_fn_) {
        std::cerr << "获取evaluate函数失败: " << dlerror() << std::endl;
        std::exit(1);
    }

    // 如果使用CUDA，检查GPU是否可用
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA初始化失败: " << cudaGetErrorString(err) << std::endl;
    } else if (deviceCount == 0) {
        std::cerr << "弟弟，你应该买显卡了，上转转！" << std::endl;
    } else {
        // 选择第一个设备并打印设备信息
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        std::cout << "使用GPU: " << deviceProp.name
                  << " (Compute Capability: " << deviceProp.major << "." << deviceProp.minor << ")"
                  << std::endl;
    }
    // 初始化张量和工作空间
    initializeTensors();
}

void CasadiGpuEvaluator::initializeTensors() {
    try {
        int n_in = fn_.n_in(), n_out = fn_.n_out();
        input_tensors_.resize(n_in);
        output_tensors_.resize(n_out);
        output_tensors_dense_.resize(n_out);

        // 根据 use_cuda_ 选择设备
        torch::Device device = torch::kCUDA;

        // 记录张量大小信息，用于调试
        size_t total_input_size = 0;
        size_t total_output_size = 0;

        // 打印输入输出张量的详细维度信息
        std::cout << "\n====== CasadiGpuEvaluator 张量维度信息 ======" << std::endl;

        // 打印输入张量信息
        std::cout << "输入张量 (" << n_in << "):" << std::endl;
        for (int i = 0; i < n_in; ++i) {
            std::cout << "  输入[" << i << "]: 形状 = [" << fn_.size1_in(i) << " x " << fn_.size2_in(i)
                      << "], 非零元素 = " << fn_.nnz_in(i);
            if (fn_.size1_in(i) * fn_.size2_in(i) > 0) {
                double sparsity = 100.0 * fn_.nnz_in(i) / (fn_.size1_in(i) * fn_.size2_in(i));
                std::cout << ", 稀疏度 = " << sparsity << "%";
            }
            std::cout << std::endl;
        }

        // 初始化输入张量
        for (int i = 0; i < n_in; ++i) {
            int nnz = fn_.nnz_in(i);
            total_input_size += nnz * sizeof(double);
            input_tensors_[i] = torch::zeros({nnz}, torch::kFloat64).contiguous().to(device);
        }

        // 打印输出张量信息
        std::cout << "输出张量 (" << n_out << "):" << std::endl;
        for (int i = 0; i < n_out; ++i) {
            std::cout << "  输出[" << i << "]: 形状 = [" << fn_.size1_out(i) << " x " << fn_.size2_out(i)
                      << "], 非零元素 = " << fn_.nnz_out(i);
            if (fn_.size1_out(i) * fn_.size2_out(i) > 0) {
                double sparsity = 100.0 * fn_.nnz_out(i) / (fn_.size1_out(i) * fn_.size2_out(i));
                std::cout << ", 稀疏度 = " << sparsity << "%";
            }
            std::cout << std::endl;
        }
        std::cout << "=============================================" << std::endl;

        // 初始化输出张量
        for (int i = 0; i < n_out; ++i) {
            int nnz = fn_.nnz_out(i);
            total_output_size += nnz * sizeof(double);
            output_tensors_[i] = torch::zeros({nnz}, torch::kFloat64).contiguous().to(device);
            output_tensors_dense_[i] = torch::zeros(
                    {static_cast<int64_t>(fn_.size1_out(i)), static_cast<int64_t>(fn_.size2_out(i))},
                    torch::kFloat64).to(device);
        }

        // 初始化工作空间
        size_t work_size = fn_.sz_w();
        work_tensor_ = torch::zeros({static_cast<int64_t>(work_size)},
                                    torch::kFloat64).contiguous().to(device);

        // 创建存储指针的张量
        d_input_ptrs_ = nullptr;
        d_output_ptrs_ = nullptr;

        // 如果是CUDA模式，预分配设备指针数组
        CHECK_CUDA_ERROR(cudaMalloc(&d_input_ptrs_, n_in * sizeof(void *)),
                         "分配输入指针数组内存失败");
        CHECK_CUDA_ERROR(cudaMalloc(&d_output_ptrs_, n_out * sizeof(void *)),
                         "分配输出指针数组内存失败");

        std::cout << "CusADi推理神经网络初始化完成: "
                  << n_in << " 个Tensor输入, "
                  << n_out << " 个Tensor输出, "
                  << "工作空间大小: " << work_size
                  << " (" << (work_size * sizeof(double) / 1024.0) << " KB)"
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "初始化张量时发生异常: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief 清除输出和工作空间张量
 */
void CasadiGpuEvaluator::resetTensors() {
    for (auto &t: output_tensors_) {
        t.zero_();
    }
    work_tensor_.zero_();
}

/**
 * @brief 准备输入张量
 * @param inputs 输入张量列表
 */
void CasadiGpuEvaluator::formatInputTensors(const std::vector<torch::Tensor> &inputs) {
    const size_t num_inputs = std::min(inputs.size(), input_tensors_.size());

    if (inputs.size() < input_tensors_.size()) {
        std::cerr << "警告: 提供的输入数量 (" << inputs.size()
                  << ") 少于预期 (" << input_tensors_.size() << ")" << std::endl;
    }

    torch::Device device = torch::kCUDA;

    for (size_t i = 0; i < num_inputs; ++i) {
        // 检查输入张量的大小是否符合预期
        if (inputs[i].numel() != input_tensors_[i].numel()) {
            std::cerr << "错误: 输入 " << i << " 的大小不匹配. 预期: "
                      << input_tensors_[i].numel() << ", 实际: "
                      << inputs[i].numel() << std::endl;
            // 继续使用默认的零张量
            continue;
        }

        try {
            input_tensors_[i] = inputs[i].contiguous().to(device);
        } catch (const std::exception &e) {
            std::cerr << "将输入张量 " << i << " 转移到 "
                      << "GPU" << " 时发生错误: "
                      << e.what() << std::endl;
            // 继续使用默认的零张量
        }
    }
}

/**
 * @brief 执行计算
 * @param inputs 输入张量列表
 */
void CasadiGpuEvaluator::compute(const std::vector<torch::Tensor> &inputs) {
    // 清除之前的计算结果
    resetTensors();

    // 准备输入张量
    formatInputTensors(inputs);

    try {
        computeGPU();
    } catch (const std::exception &e) {
        std::cerr << "计算过程中发生异常: " << e.what() << std::endl;
    }
}

/**
 * @brief GPU模式下的计算实现
 */
void CasadiGpuEvaluator::computeGPU() {
    // 获取输入和输出张量数量
    int n_in = input_tensors_.size();
    int n_out = output_tensors_.size();

    // 在主机端准备指针数组
    std::vector<void *> h_input_ptrs(n_in);
    std::vector<void *> h_output_ptrs(n_out);

    // 填充主机端指针数组
    for (int i = 0; i < n_in; ++i) {
        h_input_ptrs[i] = input_tensors_[i].data_ptr();
    }

    for (int i = 0; i < n_out; ++i) {
        h_output_ptrs[i] = output_tensors_[i].data_ptr();
    }

    // 将指针数组复制到设备内存
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_ptrs_, h_input_ptrs.data(), n_in * sizeof(void *),
                                cudaMemcpyHostToDevice),
                     "复制输入指针到GPU失败");

    CHECK_CUDA_ERROR(cudaMemcpy(d_output_ptrs_, h_output_ptrs.data(), n_out * sizeof(void *),
                                cudaMemcpyHostToDevice),
                     "复制输出指针到GPU失败");

    // 获取工作空间指针
    void *d_work = work_tensor_.data_ptr();

    // 调用评估函数
    float execution_time = eval_fn_(
            reinterpret_cast<int64_t *>(d_input_ptrs_),
            reinterpret_cast<double *>(d_work),
            reinterpret_cast<int64_t *>(d_output_ptrs_),
            1  // 单实例
    );

    // 同步并检查错误
    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "CUDA同步失败");
    CHECK_CUDA_ERROR(cudaGetLastError(), "CUDA内核执行错误");
}

torch::Tensor CasadiGpuEvaluator::getDenseResult(int output_index) {
    // 检查输出索引是否有效
    if (output_index < 0 || output_index >= output_tensors_.size()) {
        std::cerr << "错误: 无效的输出索引 " << output_index
                  << ", 有效范围: 0-" << (output_tensors_.size() - 1) << std::endl;
        // 返回空张量
        return torch::Tensor();
    }

    try {
        // 获取稀疏矩阵的三元组表示 (triplet)
        std::vector<casadi_int> rows, cols;
        fn_.sparsity_out(output_index).get_triplet(rows, cols);
        int nnz = static_cast<int>(rows.size());
        int nrows = fn_.size1_out(output_index);
        int ncols = fn_.size2_out(output_index);

        torch::Device device = torch::kCUDA;

        // 如果没有非零元素，直接返回零张量
        if (nnz == 0) {
            auto result = torch::zeros({nrows, ncols}, torch::kFloat64).to(device);
            std::cout << "getDenseResult(" << output_index << "): 返回零张量, 形状 = ["
                      << result.sizes()[0] << " x " << result.sizes()[1] << "]" << std::endl;
            return result;
        }

        // 创建索引张量
        auto row_idx = torch::from_blob(rows.data(), {nnz}, torch::kInt64).to(device);
        auto col_idx = torch::from_blob(cols.data(), {nnz}, torch::kInt64).to(device);

        // 获取值并转换为密集张量
        auto vals = output_tensors_[output_index].reshape(-1);

        // 创建稀疏张量并返回密集表示
        auto result = torch::sparse_coo_tensor(
                torch::stack({row_idx, col_idx}), vals,
                {nrows, ncols}
        ).to_dense().to(device);

        // 打印转换后的结果维度
        std::cout << "getDenseResult(" << output_index << "): 返回张量, 形状 = [";
        for (int i = 0; i < result.dim(); i++) {
            std::cout << result.sizes()[i];
            if (i < result.dim() - 1) std::cout << " x ";
        }
        std::cout << "]" << std::endl;

        // 确保返回的是二维张量
        if (result.dim() == 1 && nrows > 1 && ncols > 1) {
            std::cout << "警告: 结果被意外展平，尝试重塑回 [" << nrows << " x " << ncols << "]" << std::endl;
            return result.reshape({nrows, ncols});
        }

        return result;
    } catch (const std::exception &e) {
        std::cerr << "获取密集结果时发生错误: " << e.what() << std::endl;
        // 返回空张量
        return torch::Tensor();
    }
}

/**
 * @brief 类析构函数，释放CUDA资源
 */
CasadiGpuEvaluator::~CasadiGpuEvaluator() {
    // 释放CUDA资源
    if (d_input_ptrs_) {
        cudaFree(d_input_ptrs_);
        d_input_ptrs_ = nullptr;
    }
    if (d_output_ptrs_) {
        cudaFree(d_output_ptrs_);
        d_output_ptrs_ = nullptr;
    }


    // 释放动态库资源
    if (lib_handle_) {
        dlclose(lib_handle_);
        lib_handle_ = nullptr;
    }
}

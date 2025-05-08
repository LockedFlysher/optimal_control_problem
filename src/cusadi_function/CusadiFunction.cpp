//
// Created by lock on 25-4-28.
// Updated by Monica on 25-5-8 to remove multi-instance support, rename APIs, and add CPU support.
//
#include "optimal_control_problem/cusadi_function/CusadiFunction.h"
#include <iostream>
#include <cuda_runtime.h>  // 用于 cudaDeviceSynchronize 等 CUDA 运行时函数

/**
 * @brief 构造函数，初始化 CasadiGpuEvaluator 对象
 * @param fn_casadi CasADi 函数对象
 * @param use_cuda 是否使用 CUDA 加速，true 为 GPU，false 为 CPU
 */
CasadiGpuEvaluator::CasadiGpuEvaluator(const casadi::Function &fn_casadi, bool use_cuda)
        : fn_(fn_casadi), use_cuda_(use_cuda), lib_handle_(nullptr), eval_fn_(nullptr) {

    // 加载动态库，无论是 CPU 还是 GPU 模式都需要
    std::string path = ament_index_cpp::get_package_share_directory("optimal_control_problem") +
                       "/cusadi/build/liblocalSystemFunction.so";
    lib_handle_ = dlopen(path.c_str(), RTLD_LAZY);
    if (!lib_handle_) {
        std::cerr << "dlopen failed: " << dlerror();
        std::exit(1);
    }

    // 获取评估函数指针
    eval_fn_ = reinterpret_cast<EvalFn>(dlsym(lib_handle_, "evaluate"));
    if (!eval_fn_) {
        std::cerr << "dlsym evaluate failed: " << dlerror();
        std::exit(1);
    }

    // 初始化张量和工作空间
    initializeTensors();
}

/**
 * @brief 析构函数，释放动态库资源
 */
CasadiGpuEvaluator::~CasadiGpuEvaluator() {
    if (lib_handle_) {
        dlclose(lib_handle_);
    }
}

/**
 * @brief 初始化计算所需的张量和工作空间
 */
void CasadiGpuEvaluator::initializeTensors() {
    int n_in = fn_.n_in(), n_out = fn_.n_out();
    input_tensors_.resize(n_in);
    output_tensors_.resize(n_out);
    output_tensors_dense_.resize(n_out);

    // 根据 use_cuda_ 选择设备
    torch::Device device = use_cuda_ ? torch::kCUDA : torch::kCPU;

    // 初始化输入张量
    for (int i = 0; i < n_in; ++i) {
        int nnz = fn_.nnz_in(i);
        input_tensors_[i] = torch::zeros({nnz}, torch::kFloat64).contiguous().to(device);
    }

    // 初始化输出张量
    for (int i = 0; i < n_out; ++i) {
        int nnz = fn_.nnz_out(i);
        output_tensors_[i] = torch::zeros({nnz}, torch::kFloat64).contiguous().to(device);
        output_tensors_dense_[i] = torch::zeros(
                {static_cast<int64_t>(fn_.size1_out(i)), static_cast<int64_t>(fn_.size2_out(i))},
                torch::kFloat64).to(device);
    }

    // 初始化工作空间
    work_tensor_ = torch::zeros({static_cast<int64_t>(fn_.sz_w())},
                                torch::kFloat64).contiguous().to(device);

    // 创建存储指针的张量（在相应设备上）
    input_ptrs_tensor_ = torch::empty({n_in}, torch::kInt64).to(device);
    output_ptrs_tensor_ = torch::empty({n_out}, torch::kInt64).to(device);
}

/**
 * @brief 清除输出和工作空间张量
 */
void CasadiGpuEvaluator::resetTensors() {
    for (auto &t : output_tensors_) {
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
    torch::Device device = use_cuda_ ? torch::kCUDA : torch::kCPU;
    for (size_t i = 0; i < num_inputs; ++i) {
        input_tensors_[i] = inputs[i].contiguous().to(device);
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

    // 在主机端准备输入指针数组
    std::vector<int64_t> h_input_ptrs(input_tensors_.size());
    for (size_t i = 0; i < input_tensors_.size(); ++i) {
        h_input_ptrs[i] = reinterpret_cast<int64_t>(input_tensors_[i].data_ptr());
    }

    // 在主机端准备输出指针数组
    std::vector<int64_t> h_output_ptrs(output_tensors_.size());
    for (size_t i = 0; i < output_tensors_.size(); ++i) {
        h_output_ptrs[i] = reinterpret_cast<int64_t>(output_tensors_[i].data_ptr());
    }

    // 将主机端数据拷贝到设备张量
    torch::Tensor h_input_tensor = torch::from_blob(
            h_input_ptrs.data(), {static_cast<int64_t>(h_input_ptrs.size())}, torch::kInt64);
    input_ptrs_tensor_.copy_(h_input_tensor);

    torch::Tensor h_output_tensor = torch::from_blob(
            h_output_ptrs.data(), {static_cast<int64_t>(h_output_ptrs.size())}, torch::kInt64);
    output_ptrs_tensor_.copy_(h_output_tensor);

    // 获取设备指针
    int64_t *d_input_ptrs = input_ptrs_tensor_.data_ptr<int64_t>();
    double *d_work = work_tensor_.data_ptr<double>();
    int64_t *d_output_ptrs = output_ptrs_tensor_.data_ptr<int64_t>();

    // 调用评估函数，传递 1 作为实例数量（已移除多实例支持）
    float execution_time = eval_fn_(d_input_ptrs, d_work, d_output_ptrs, 1);

    // 如果是 GPU 模式，执行 CUDA 同步和错误检查
    if (use_cuda_) {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after evaluate: " << cudaGetErrorString(err) << std::endl;
            std::exit(1);
        }
    }
}

/**
 * @brief 获取密集格式的输出张量
 * @param output_index 输出索引
 * @return 密集格式的输出张量
 */
torch::Tensor CasadiGpuEvaluator::getDenseResult(int output_index) {
    // 获取稀疏矩阵的三元组表示 (triplet)
    std::vector<casadi_int> rows, cols;
    fn_.sparsity_out(output_index).get_triplet(rows, cols);
    int nnz = static_cast<int>(rows.size());

    torch::Device device = use_cuda_ ? torch::kCUDA : torch::kCPU;

    // 如果没有非零元素，直接返回零张量
    if (nnz == 0) {
        return torch::zeros({fn_.size1_out(output_index), fn_.size2_out(output_index)},
                            torch::kFloat64).to(device);
    }

    // 创建索引张量
    auto row_idx = torch::from_blob(rows.data(), {nnz}, torch::kInt64).to(device);
    auto col_idx = torch::from_blob(cols.data(), {nnz}, torch::kInt64).to(device);

    // 获取值并转换为密集张量
    auto vals = output_tensors_[output_index].reshape(-1);

    // 创建稀疏张量并可能返回密集表示
    return torch::sparse_coo_tensor(
            torch::stack({row_idx, col_idx}), vals,
            {fn_.size1_out(output_index), fn_.size2_out(output_index)}
    ).to_dense().to(device);
}
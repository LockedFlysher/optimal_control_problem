//
// Created by lock on 25-4-28.
//
#include "optimal_control_problem/cusadi_function/CusadiFunction.h"
#include <iostream>
#include <cuda_runtime.h>  // 用于cudaDeviceSynchronize等CUDA运行时函数

/**
 * @brief 构造函数，初始化CusadiFunction对象
 * @param fn_casadi CasADi函数对象
 * @param num_instances 并行实例数量
 */
CusadiFunction::CusadiFunction(const casadi::Function &fn_casadi, int num_instances)
        : fn_(fn_casadi), num_instances_(num_instances) {
    // 加载CUDA共享库，直接加载的就是这么一个函数，不用做任何的路径修改，先暴力地这么做
    std::string path = ament_index_cpp::get_package_share_directory("optimal_control_problem") +
                       "/scripts/cusadi/build/liblocalSystemFunction.so";
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
    // 设置张量和工作空间
    setup();
}

/**
 * @brief 析构函数，释放动态库资源
 */
CusadiFunction::~CusadiFunction() {
    if (lib_handle_) dlclose(lib_handle_);
}

/**
 * @brief 设置计算所需的张量和工作空间
 */
void CusadiFunction::setup() {
    int n_in = fn_.n_in(), n_out = fn_.n_out();
    input_tensors_.resize(n_in);
    output_tensors_.resize(n_out);
    output_tensors_dense_.resize(n_out);

    // 初始化输入张量
    for (int i = 0; i < n_in; ++i) {
        int nnz = fn_.nnz_in(i);
        input_tensors_[i] = torch::zeros({num_instances_, nnz}, torch::kFloat64).contiguous().to(torch::kCUDA);
    }

    // 初始化输出张量
    for (int i = 0; i < n_out; ++i) {
        int nnz = fn_.nnz_out(i);
        output_tensors_[i] = torch::zeros({num_instances_, nnz}, torch::kFloat64).contiguous().to(torch::kCUDA);
        output_tensors_dense_[i] = torch::zeros(
                {num_instances_, static_cast<int64_t>(fn_.size1_out(i)), static_cast<int64_t>(fn_.size2_out(i))},
                torch::kFloat64).to(torch::kCUDA);
    }

    // 初始化工作空间
    work_tensor_ = torch::zeros({num_instances_, static_cast<int64_t>(fn_.sz_w())},
                                torch::kFloat64).contiguous().to(torch::kCUDA);

    // 创建存储指针的GPU张量
    input_ptrs_tensor_ = torch::empty({n_in}, torch::kInt64).to(torch::kCUDA);
    output_ptrs_tensor_ = torch::empty({n_out}, torch::kInt64).to(torch::kCUDA);
}

/**
 * @brief 清除输出和工作空间张量
 */
void CusadiFunction::clearTensors() {
    for (auto &t: output_tensors_) {
        t.zero_();
    }
    work_tensor_.zero_();
}

/**
 * @brief 准备输入张量
 * @param inputs 输入张量列表
 */
void CusadiFunction::prepareInputTensors(const std::vector<torch::Tensor> &inputs) {
    const size_t num_inputs = std::min(inputs.size(), input_tensors_.size());
    for (size_t i = 0; i < num_inputs; ++i) {
        input_tensors_[i] = inputs[i].contiguous().to(torch::kCUDA);
    }
}

/**
 * @brief 执行CUDA计算
 * @param inputs 输入张量列表
 */
void CusadiFunction::evaluate(const std::vector<torch::Tensor> &inputs) {
    // 清除之前的计算结果
    clearTensors();

    // 准备输入张量
    prepareInputTensors(inputs);

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

    // 调用CUDA函数
    float execution_time = eval_fn_(d_input_ptrs, d_work, d_output_ptrs, num_instances_);

    // 错误检查
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after evaluate: " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

/**
 * @brief 获取密集格式的输出张量
 * @param out_idx 输出索引
 * @return 密集格式的输出张量
 */
torch::Tensor CusadiFunction::getDenseOutput(int out_idx) {
    // 获取稀疏矩阵的三元组表示(triplet)
    std::vector<casadi_int> rows, cols;
    fn_.sparsity_out(out_idx).get_triplet(rows, cols);
    int nnz = static_cast<int>(rows.size());

    // 如果没有非零元素，直接返回零张量
    if (nnz == 0) {
        return torch::zeros({num_instances_, fn_.size1_out(out_idx), fn_.size2_out(out_idx)},
                            torch::kFloat64).to(torch::kCUDA);
    }

    // 创建索引张量 - 使用unsqueeze和expand代替repeat以提高性能
    auto env_idx = torch::arange(num_instances_, torch::kInt64)
            .unsqueeze(1).expand({-1, nnz}).reshape(-1).to(torch::kCUDA);

    // 使用from_blob创建行索引张量
    auto row_idx = torch::from_blob(rows.data(), {nnz}, torch::kInt64)
            .expand({num_instances_, -1}).reshape(-1).to(torch::kCUDA);

    // 使用from_blob创建列索引张量
    auto col_idx = torch::from_blob(cols.data(), {nnz}, torch::kInt64)
            .expand({num_instances_, -1}).reshape(-1).to(torch::kCUDA);

    // 获取值并转换为密集张量
    auto vals = output_tensors_[out_idx].reshape(-1);
    return torch::sparse_coo_tensor(
            torch::stack({env_idx, row_idx, col_idx}), vals,
            {num_instances_, fn_.size1_out(out_idx), fn_.size2_out(out_idx)}
    ).to_dense();
}

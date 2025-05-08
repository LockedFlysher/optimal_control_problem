//
// Created by lock on 25-4-28.
// Updated by Monica on 25-5-8 to remove multi-instance support, rename APIs, and add CPU support.
//

#ifndef BUILD_CASADI_GPU_EVALUATOR_H
#define BUILD_CASADI_GPU_EVALUATOR_H

#include <vector>
#include <string>
#include <dlfcn.h>
// 必须先包含 casadi 再包含 torch，否则会报错
#include <casadi/casadi.hpp>
#include <torch/torch.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

/**
 * @class CasadiGpuEvaluator
 * @brief 用于评估 CasADi 函数的类，支持 GPU 和 CPU 计算
 *        该类将 CasADi 函数与 PyTorch 张量结合，可在指定设备上进行计算
 */
class CasadiGpuEvaluator {
public:
    /**
     * @brief 构造函数，初始化 CasADi 函数评估器
     * @param fn_casadi CasADi 函数对象
     * @param use_cuda 是否使用 CUDA 加速，默认为 true；如果为 false，则在 CPU 上进行计算
     */
    CasadiGpuEvaluator(const casadi::Function& fn_casadi);

    /**
     * @brief 析构函数，释放动态库资源和CUDA资源
     */
    ~CasadiGpuEvaluator();

    /**
     * @brief 执行 CasADi 函数计算
     * @param inputs 输入张量列表
     */
    void compute(const std::vector<torch::Tensor>& inputs);

    /**
     * @brief 获取指定输出的密集格式张量
     * @param output_index 输出张量的索引
     * @return 密集格式的输出张量
     */
    torch::Tensor getDenseResult(int output_index);

private:
    /**
     * @brief 初始化张量和工作空间
     */
    void initializeTensors();

    /**
     * @brief 清除输出和工作空间张量
     */
    void resetTensors();

    /**
     * @brief 准备输入张量
     * @param inputs 输入张量列表
     */
    void formatInputTensors(const std::vector<torch::Tensor>& inputs);

    /**
     * @brief GPU模式下的计算实现
     */
    void computeGPU();

    casadi::Function fn_;  // CasADi 函数对象
    void* lib_handle_;     // 动态库句柄

    // 定义评估函数指针类型
    using EvalFn = float(*)(int64_t*, double*, int64_t*, int);
    EvalFn eval_fn_;       // 评估函数指针

    std::vector<torch::Tensor> input_tensors_;  // 输入张量列表
    std::vector<torch::Tensor> output_tensors_; // 输出张量列表
    std::vector<torch::Tensor> output_tensors_dense_; // 密集格式的输出张量列表
    torch::Tensor work_tensor_;                 // 工作空间张量

    // GPU模式下使用的设备内存指针
    void* d_input_ptrs_;   // 设备端输入指针数组
    void* d_output_ptrs_;  // 设备端输出指针数组

};

#endif // BUILD_CASADI_GPU_EVALUATOR_H

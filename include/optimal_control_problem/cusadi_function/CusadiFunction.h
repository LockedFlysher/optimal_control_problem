//
// Created by lock on 25-4-28.
//

#ifndef BUILD_CUSADIFUNCTION_H
#define BUILD_CUSADIFUNCTION_H

#include <vector>
#include <string>
#include <dlfcn.h>
//一定要先casadi再torch，// 否则会报错
#include <casadi/casadi.hpp>
#include <torch/torch.h>


class CusadiFunction {
public:
    CusadiFunction(const casadi::Function& fn_casadi, int num_instances);
    ~CusadiFunction();

    void evaluate(const std::vector<torch::Tensor>& inputs);
    torch::Tensor getDenseOutput(int out_idx);

private:
    void setup();
    void clearTensors();
    void prepareInputTensors(const std::vector<torch::Tensor>& inputs);

    casadi::Function fn_;
    int num_instances_;

    void* lib_handle_;
    using EvalFn = float(*)(int64_t*, double*, int64_t*, int);
    EvalFn eval_fn_;

    std::vector<torch::Tensor> input_tensors_;
    std::vector<torch::Tensor> output_tensors_;
    std::vector<torch::Tensor> output_tensors_dense_;
    torch::Tensor work_tensor_;

    // 指针数组用 GPU上的指针数组 存储
    torch::Tensor input_ptrs_tensor_;  // GPU上的指针数组
    torch::Tensor output_ptrs_tensor_; // GPU上的指针数组

    int64_t* fn_input_ptrs_;
    double* fn_work_ptr_;
    int64_t* fn_output_ptrs_;
};

#endif //BUILD_CUSADIFUNCTION_H

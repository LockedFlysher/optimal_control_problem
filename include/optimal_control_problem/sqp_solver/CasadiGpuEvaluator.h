//
// Created by lock on 25-4-28.
//

#ifndef BUILD_CASADI_GPU_EVALUATOR_H
#define BUILD_CASADI_GPU_EVALUATOR_H

#include <vector>
#include <string>
#include <dlfcn.h>
// Must include casadi before torch to avoid errors
#include <casadi/casadi.hpp>
#include <torch/torch.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

/**
 * @class CasadiGpuEvaluator
 * @brief Class for evaluating CasADi functions, supporting GPU and CPU computation
 *        This class combines CasADi functions with PyTorch tensors for computation on specified devices
 */
class CasadiGpuEvaluator {
public:
    /**
     * @brief Constructor, initializes CasADi function evaluator
     * @param fn_casadi CasADi function object
     */
    CasadiGpuEvaluator(const casadi::Function& fn_casadi);

    /**
     * @brief Destructor, releases dynamic library and CUDA resources
     */
    ~CasadiGpuEvaluator();

    /**
     * @brief Execute CasADi function computation
     * @param inputs Input tensor list
     */
    void compute(const std::vector<torch::Tensor>& inputs);

    /**
     * @brief Get dense format tensor for specified output
     * @param output_index Output tensor index
     * @return Dense format output tensor
     */
    torch::Tensor getDenseResult(int output_index);

private:
    /**
     * @brief Initialize tensors and workspace
     */
    void initializeTensors();

    /**
     * @brief Prepare input tensors
     * @param inputs Input tensor list
     */
    void formatInputTensors(const std::vector<torch::Tensor>& inputs);

    void resetTensors();

    /**
     * @brief GPU mode computation implementation
     */
    void computeGPU();

    casadi::Function fn_;  // CasADi function object
    void* lib_handle_;     // Dynamic library handle

    // Define evaluation function pointer type
    using EvalFn = float(*)(int64_t*, double*, int64_t*, int);
    EvalFn eval_fn_;       // Evaluation function pointer

    std::vector<torch::Tensor> input_tensors_;  // Input tensor list
    std::vector<torch::Tensor> output_tensors_; // Output tensor list
    std::vector<torch::Tensor> output_tensors_dense_; // Dense format output tensor list
    torch::Tensor work_tensor_;                 // Workspace tensor

    // Device memory pointers used in GPU mode
    void* d_input_ptrs_;   // Device input pointer array
    void* d_output_ptrs_;  // Device output pointer array

};

#endif // BUILD_CASADI_GPU_EVALUATOR_H

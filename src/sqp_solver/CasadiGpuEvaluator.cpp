//
// Created by lock on 25-4-28.
// Updated by Monica on 25-5-8 to remove multi-instance support, rename APIs, and add CPU support.
//
#include "optimal_control_problem/sqp_solver/CasadiGpuEvaluator.h"
#include <iostream>
#include <cuda_runtime.h>  // For cudaDeviceSynchronize and other CUDA runtime functions
#include <sstream>  // For building detailed error messages

/**
 * @brief Constructor, initializes CasadiGpuEvaluator object
 * @param fn_casadi CasADi function object
 */
CasadiGpuEvaluator::CasadiGpuEvaluator(const casadi::Function &fn_casadi)
        : fn_(fn_casadi), lib_handle_(nullptr), eval_fn_(nullptr) {

    // Load dynamic library, required for both CPU and GPU modes
    std::string path = ament_index_cpp::get_package_share_directory("optimal_control_problem") +
                       "/cusadi/build/liblocalSystemFunction.so";
    lib_handle_ = dlopen(path.c_str(), RTLD_LAZY);
    if (!lib_handle_) {
        std::cerr << "Failed to load dynamic library: " << dlerror() << std::endl;
        std::cerr << "Attempted path: " << path << std::endl;
        std::exit(1);
    }

    // Get evaluation function pointer
    eval_fn_ = reinterpret_cast<EvalFn>(dlsym(lib_handle_, "evaluate"));
    if (!eval_fn_) {
        std::cerr << "Failed to get evaluate function: " << dlerror() << std::endl;
        std::exit(1);
    }

    // Check if GPU is available
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(err) << std::endl;
    } else if (deviceCount == 0) {
        std::cerr << "No GPU devices found. Consider purchasing a GPU!" << std::endl;
    } else {
        // Select first device and print device information
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        std::cout << "Using GPU: " << deviceProp.name
                  << " (Compute Capability: " << deviceProp.major << "." << deviceProp.minor << ")"
                  << std::endl;
    }
    // Initialize tensors and workspace
    initializeTensors();
}

void CasadiGpuEvaluator::initializeTensors() {
    try {
        int n_in = fn_.n_in(), n_out = fn_.n_out();
        input_tensors_.resize(n_in);
        output_tensors_.resize(n_out);
        output_tensors_dense_.resize(n_out);

        // Select device
        torch::Device device = torch::kCUDA;

        // Record tensor size information for debugging
        size_t total_input_size = 0;
        size_t total_output_size = 0;

        // Print detailed dimension information for input and output tensors
        std::cout << "\n====== CasadiGpuEvaluator Tensor Dimension Information ======" << std::endl;

        // Print input tensor information
        std::cout << "Input tensors (" << n_in << "):" << std::endl;
        for (int i = 0; i < n_in; ++i) {
            std::cout << "  Input[" << i << "]: Shape = [" << fn_.size1_in(i) << " x " << fn_.size2_in(i)
                      << "], Non-zero elements = " << fn_.nnz_in(i);
            if (fn_.size1_in(i) * fn_.size2_in(i) > 0) {
                double sparsity = 100.0 * fn_.nnz_in(i) / (fn_.size1_in(i) * fn_.size2_in(i));
                std::cout << ", Sparsity = " << sparsity << "%";
            }
            std::cout << std::endl;
        }

        // Initialize input tensors
        for (int i = 0; i < n_in; ++i) {
            int nnz = fn_.nnz_in(i);
            total_input_size += nnz * sizeof(double);
            input_tensors_[i] = torch::zeros({nnz}, torch::kFloat64).contiguous().to(device);
        }

        // Print output tensor information
        std::cout << "Output tensors (" << n_out << "):" << std::endl;
        for (int i = 0; i < n_out; ++i) {
            std::cout << "  Output[" << i << "]: Shape = [" << fn_.size1_out(i) << " x " << fn_.size2_out(i)
                      << "], Non-zero elements = " << fn_.nnz_out(i);
            if (fn_.size1_out(i) * fn_.size2_out(i) > 0) {
                double sparsity = 100.0 * fn_.nnz_out(i) / (fn_.size1_out(i) * fn_.size2_out(i));
                std::cout << ", Sparsity = " << sparsity << "%";
            }
            std::cout << std::endl;
        }
        std::cout << "=============================================" << std::endl;

        // Initialize output tensors
        for (int i = 0; i < n_out; ++i) {
            int nnz = fn_.nnz_out(i);
            total_output_size += nnz * sizeof(double);
            output_tensors_[i] = torch::zeros({nnz}, torch::kFloat64).contiguous().to(device);
            output_tensors_dense_[i] = torch::zeros(
                    {static_cast<int64_t>(fn_.size1_out(i)), static_cast<int64_t>(fn_.size2_out(i))},
                    torch::kFloat64).to(device);
        }

        // Initialize workspace
        size_t work_size = fn_.sz_w();
        work_tensor_ = torch::zeros({static_cast<int64_t>(work_size)},
                                    torch::kFloat64).contiguous().to(device);

        // Create tensors to store pointers
        d_input_ptrs_ = nullptr;
        d_output_ptrs_ = nullptr;

        // Pre-allocate device pointer arrays for CUDA mode
        cudaError_t err = cudaMalloc(&d_input_ptrs_, n_in * sizeof(void *));
        if (err != cudaSuccess) {
            std::stringstream error_ss;
            error_ss << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                     << "Failed to allocate input pointer array memory" << " - " << cudaGetErrorString(err);
            std::cerr << error_ss.str() << std::endl;
            return;
        }

        err = cudaMalloc(&d_output_ptrs_, n_out * sizeof(void *));
        if (err != cudaSuccess) {
            std::stringstream error_ss;
            error_ss << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                     << "Failed to allocate output pointer array memory" << " - " << cudaGetErrorString(err);
            std::cerr << error_ss.str() << std::endl;
            return;
        }

        std::cout << "CusADi inference network initialized: "
                  << n_in << " tensor inputs, "
                  << n_out << " tensor outputs, "
                  << "workspace size: " << work_size
                  << " (" << (work_size * sizeof(double) / 1024.0) << " KB)"
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Exception during tensor initialization: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief Clear output and workspace tensors
 */
void CasadiGpuEvaluator::resetTensors() {
    for (auto &t: output_tensors_) {
        t.zero_();
    }
    work_tensor_.zero_();
}

/**
 * @brief Prepare input tensors
 * @param inputs Input tensor list
 */
void CasadiGpuEvaluator::formatInputTensors(const std::vector<torch::Tensor> &inputs) {
    const size_t num_inputs = std::min(inputs.size(), input_tensors_.size());

    if (inputs.size() < input_tensors_.size()) {
        std::cerr << "Warning: Number of provided inputs (" << inputs.size()
                  << ") is less than expected (" << input_tensors_.size() << ")" << std::endl;
    }

    torch::Device device = torch::kCUDA;

    for (size_t i = 0; i < num_inputs; ++i) {
        // Check if input tensor size matches expected size
        if (inputs[i].numel() != input_tensors_[i].numel()) {
            std::cerr << "Error: Size mismatch for input " << i << ". Expected: "
                      << input_tensors_[i].numel() << ", Actual: "
                      << inputs[i].numel() << std::endl;
            // Continue using default zero tensor
            continue;
        }

        try {
            input_tensors_[i] = inputs[i].contiguous().to(device);
        } catch (const std::exception &e) {
            std::cerr << "Error transferring input tensor " << i << " to "
                      << "GPU" << ": "
                      << e.what() << std::endl;
            // Continue using default zero tensor
        }
    }
}

/**
 * @brief Execute computation
 * @param inputs Input tensor list
 */
void CasadiGpuEvaluator::compute(const std::vector<torch::Tensor> &inputs) {
    // Clear previous computation results
    resetTensors();

    // Prepare input tensors
    formatInputTensors(inputs);

    try {
        computeGPU();
    } catch (const std::exception &e) {
        std::cerr << "Exception during computation: " << e.what() << std::endl;
    }
}

/**
 * @brief GPU mode computation implementation
 */
void CasadiGpuEvaluator::computeGPU() {
    // Get number of input and output tensors
    int n_in = input_tensors_.size();
    int n_out = output_tensors_.size();

    // Prepare pointer arrays on host
    std::vector<void *> h_input_ptrs(n_in);
    std::vector<void *> h_output_ptrs(n_out);

    // Fill host pointer arrays
    for (int i = 0; i < n_in; ++i) {
        h_input_ptrs[i] = input_tensors_[i].data_ptr();
    }

    for (int i = 0; i < n_out; ++i) {
        h_output_ptrs[i] = output_tensors_[i].data_ptr();
    }

    // Copy pointer arrays to device memory
    cudaError_t err = cudaMemcpy(d_input_ptrs_, h_input_ptrs.data(), n_in * sizeof(void *),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::stringstream error_ss;
        error_ss << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                 << "Failed to copy input pointers to GPU" << " - " << cudaGetErrorString(err);
        std::cerr << error_ss.str() << std::endl;
        return;
    }

    err = cudaMemcpy(d_output_ptrs_, h_output_ptrs.data(), n_out * sizeof(void *),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::stringstream error_ss;
        error_ss << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                 << "Failed to copy output pointers to GPU" << " - " << cudaGetErrorString(err);
        std::cerr << error_ss.str() << std::endl;
        return;
    }

    // Get workspace pointer
    void *d_work = work_tensor_.data_ptr();

    // Call evaluation function
    float execution_time = eval_fn_(
            reinterpret_cast<int64_t *>(d_input_ptrs_),
            reinterpret_cast<double *>(d_work),
            reinterpret_cast<int64_t *>(d_output_ptrs_),
            1  // Single instance
    );

    // Synchronize and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::stringstream error_ss;
        error_ss << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                 << "CUDA synchronization failed" << " - " << cudaGetErrorString(err);
        std::cerr << error_ss.str() << std::endl;
        return;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::stringstream error_ss;
        error_ss << "CUDA Error [" << __FILE__ << ":" << __LINE__ << "]: "
                 << "CUDA kernel execution error" << " - " << cudaGetErrorString(err);
        std::cerr << error_ss.str() << std::endl;
        return;
    }
}

torch::Tensor CasadiGpuEvaluator::getDenseResult(int output_index) {
    // Check if output index is valid
    if (output_index < 0 || output_index >= output_tensors_.size()) {
        std::cerr << "Error: Invalid output index " << output_index
                  << ", valid range: 0-" << (output_tensors_.size() - 1) << std::endl;
        // Return empty tensor
        return torch::Tensor();
    }

    try {
        // Get triplet representation of sparse matrix
        std::vector<casadi_int> rows, cols;
        fn_.sparsity_out(output_index).get_triplet(rows, cols);
        int nnz = static_cast<int>(rows.size());
        int nrows = fn_.size1_out(output_index);
        int ncols = fn_.size2_out(output_index);

        torch::Device device = torch::kCUDA;

        // If no non-zero elements, return zero tensor directly
        if (nnz == 0) {
            auto result = torch::zeros({nrows, ncols}, torch::kFloat64).to(device);
            return result;
        }

        // Create index tensors
        auto row_idx = torch::from_blob(rows.data(), {nnz}, torch::kInt64).to(device);
        auto col_idx = torch::from_blob(cols.data(), {nnz}, torch::kInt64).to(device);

        // Get values and convert to dense tensor
        auto vals = output_tensors_[output_index].reshape(-1);

        // Create sparse tensor and return dense representation
        auto result = torch::sparse_coo_tensor(
                torch::stack({row_idx, col_idx}), vals,
                {nrows, ncols}
        ).to_dense().to(device);

        // Print dimensions of converted result
        for (int i = 0; i < result.dim(); i++) {
            std::cout << result.sizes()[i];
            if (i < result.dim() - 1) std::cout << " x ";
        }
        std::cout << "]" << std::endl;

        // Ensure returned tensor is 2D
        if (result.dim() == 1 && nrows > 1 && ncols > 1) {
            std::cout << "Warning: Result was unexpectedly flattened, attempting to reshape to ["
                      << nrows << " x " << ncols << "]" << std::endl;
            return result.reshape({nrows, ncols});
        }

        return result;
    } catch (const std::exception &e) {
        std::cerr << "Error getting dense result: " << e.what() << std::endl;
        // Return empty tensor
        return torch::Tensor();
    }
}

/**
 * @brief Class destructor, releases CUDA resources
 */
CasadiGpuEvaluator::~CasadiGpuEvaluator() {
    // Free CUDA resources
    if (d_input_ptrs_) {
        cudaFree(d_input_ptrs_);
        d_input_ptrs_ = nullptr;
    }
    if (d_output_ptrs_) {
        cudaFree(d_output_ptrs_);
        d_output_ptrs_ = nullptr;
    }

    // Free dynamic library resources
    if (lib_handle_) {
        dlclose(lib_handle_);
        lib_handle_ = nullptr;
    }
}

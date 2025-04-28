#include "optimal_control_problem/cusadi_function/CusadiFunction.h"
//一定要先casadi再torch，// 否则会报错
#include <casadi/casadi.hpp>
#include <torch/torch.h>
#include <iostream>

int main() {
    // 1. 加载 CasADi 函数
    //ToDO casadi的方程的路径（后面整合改）
    std::string casadi_file = "/home/lyj/project/NEBULA_ws/src/cusadi_trial/cusadi/src/casadi_functions/QuadraticProblem.casadi";
    casadi::Function fn = casadi::Function::load(casadi_file);
    std::cout<<"Loaded CasADi function: "<<fn.name()<<"\n";

    // 2. 创建 CusadiFunction
    int N_ENVS = 1;
    CusadiFunction solver(fn, N_ENVS);

    // 3. 构造输入 tensors
    auto p = torch::tensor({0.5}, torch::kFloat64).to(torch::kCUDA);
    auto x = torch::tensor({1.0, 1.0}, torch::kFloat64).to(torch::kCUDA);
    auto l = torch::tensor({0.5, -5.0, -5.0, 0.0, -casadi::inf},
                           torch::kFloat64).to(torch::kCUDA);
    auto u = torch::tensor({0.5, 5.0, 5.0, 0.0, 0.0},
                           torch::kFloat64).to(torch::kCUDA);
    std::vector<torch::Tensor> inputs = {p, x, l, u};

    // 4. 求解
    solver.evaluate(inputs);

    // 5. 打印输出
    for (int i = 0; i < fn.n_out(); ++i) {
        auto D = solver.getDenseOutput(i);
        std::cout<<"Output "<<i<<" =\n"<<D<<std::endl;
        // std::cout<<"Output "<<i<<" =\n"<<D.cpu()<<std::endl;
    }
    return 0;
}

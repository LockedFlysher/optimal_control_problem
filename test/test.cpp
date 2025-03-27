//
// Created by lock on 25-3-27.
//
#include "sqp_solver/SQPOptimizationSolver.h"
#include <iostream>

using namespace casadi;

void test_all_qp_cases() {
    // 定义浮点数极限值，用于表示无限制
    const double INF = std::numeric_limits<float>::infinity();
    {
        std::cout << "\n===== 测试案例2：无约束二次规划 =====" << std::endl;
        SX xs = SX::sym("x", 2);
        SX obj = pow(xs(0) - 3, 2) + pow(xs(1) + 2, 2);  // 最小点在(3, -2)
        auto nlp = SXDict{
                {"x", xs},
                {"f", obj},
                {"g", SX()}  ,// 无约束
                {"p", SX()}
        };
        SQPOptimizationSolver solver(nlp);

        auto arg = DMDict{
                {"lbx", {-100, -100}},
                {"ubx", {100, 100}},
                {"lbg", {} },  // 空约束
                {"ubg", {}},  // 空约束
                {"p", {0}}
        };
        std::cout << "预期解：x1=3, x2=-2" << std::endl;
        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
    }

    std::cout << "\n========== 测试完成 ==========\n" << std::endl;
}
int main() {
    test_all_qp_cases();
    return 0;
}

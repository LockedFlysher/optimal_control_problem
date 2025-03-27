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
        std::cout << "\n===== 测试案例1：标准二次规划 =====" << std::endl;
        SX xs = SX::sym("x", 2);
        SX obj = pow(xs(0), 2) + pow(xs(1), 2);  // 最小点在原点(0,0)
        SX g = SX::vertcat({xs(0) + xs(1) - 1});  // 约束: x1 + x2 = 1

        auto nlp = SXDict{
                {"x", xs},
                {"f", obj},
                {"g", g},
                {"p", SX()}
        };
        SQPOptimizationSolver solver(nlp);

        auto arg = DMDict{
                {"lbx", {-50, -100}},
                {"ubx", {50, 100}},
                {"lbg", {-0.00}},
                {"ubg", {0.00}},
                {"p", {}}
        };
        std::cout << "预期解：x1=0.5, x2=0.5" << std::endl;
        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
    }

    {
        std::cout << "\n===== 测试案例2：无约束二次规划 =====" << std::endl;
        SX xs = SX::sym("x", 2);
        SX obj = pow(xs(0) - 3, 2) + pow(xs(1) + 2, 2);  // 最小点在(3, -2)
        auto nlp = SXDict{
                {"x", xs},
                {"f", obj},
                {"g", SX()},  // 无约束
                {"p", SX()}
        };
        SQPOptimizationSolver solver(nlp);

        auto arg = DMDict{
                {"lbx", {-50, -100}},
                {"ubx", {50, 100}},
                {"lbg", {}},  // 空约束
                {"ubg", {}},  // 空约束
                {"p", {}}
        };
        std::cout << "预期解：x1=3, x2=-2" << std::endl;
        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
    }
//
//    {
//        std::cout << "\n===== 测试案例3：带不等式约束的二次规划 =====" << std::endl;
//        SX xs = SX::sym("x", 2);
//        SX obj = pow(xs(0) - 2, 2) + pow(xs(1) - 3, 2);  // 最小点在(2, 3)
//        SX g = SX::vertcat({xs(0) + xs(1) - 1});  // 约束: x1 + x2 >= 1
//
//        auto nlp = SXDict{
//                {"x", xs},
//                {"f", obj},
//                {"g", g},
//                {"p", SX()}
//        };
//        SQPOptimizationSolver solver(nlp);
//
//        auto arg = DMDict{
//                {"lbx", {-100, -100}},
//                {"ubx", {100, 100}},
//                {"lbg", {1}},  // 下界为1
//                {"ubg", {INF}},  // 上界为无穷
//                {"p", {}}
//        };
//        std::cout << "预期解：x1=2, x2=3 (约束不起作用)" << std::endl;
//        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
//    }
//
//    {
//        std::cout << "\n===== 测试案例4：带多个不等式约束的二次规划 =====" << std::endl;
//        SX xs = SX::sym("x", 2);
//        SX obj = pow(xs(0), 2) + pow(xs(1), 2);  // 最小点在原点(0,0)
//        // 约束: x1 >= 1, x2 >= 2
//        SX g = SX::vertcat({xs(0), xs(1)});
//
//        auto nlp = SXDict{
//                {"x", xs},
//                {"f", obj},
//                {"g", g},
//                {"p", SX()}
//        };
//        SQPOptimizationSolver solver(nlp);
//
//        auto arg = DMDict{
//                {"lbx", {-100, -100}},
//                {"ubx", {100, 100}},
//                {"lbg", {1, 2}},  // 下界
//                {"ubg", {INF, INF}},  // 上界为无穷
//                {"p", {}}
//        };
//        std::cout << "预期解：x1=1, x2=2" << std::endl;
//        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
//    }
//
//    {
//        std::cout << "\n===== 测试案例5：带等式和不等式约束的二次规划 =====" << std::endl;
//        SX xs = SX::sym("x", 3);
//        SX obj = pow(xs(0) - 1, 2) + pow(xs(1) - 2, 2) + pow(xs(2) - 3, 2);  // 最小点在(1,2,3)
//        // 约束: x1 + x2 + x3 = 5, x1 >= 0, x2 >= 0, x3 >= 0
//        SX g = SX::vertcat({xs(0) + xs(1) + xs(2) - 5});
//
//        auto nlp = SXDict{
//                {"x", xs},
//                {"f", obj},
//                {"g", g},
//                {"p", SX()}
//        };
//        SQPOptimizationSolver solver(nlp);
//
//        auto arg = DMDict{
//                {"lbx", {0, 0, 0}},  // 变量下界
//                {"ubx", {INF, INF, INF}},  // 变量上界
//                {"lbg", {0}},  // 约束下界
//                {"ubg", {0}},  // 约束上界
//                {"p", {}}
//        };
//        std::cout << "预期解：接近(1,2,2)或其他满足约束的点" << std::endl;
//        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
//    }
//
//    {
//        std::cout << "\n===== 测试案例6：带参数化的二次规划 =====" << std::endl;
//        SX xs = SX::sym("x", 2);
//        SX p = SX::sym("p", 1);  // 参数
//        SX obj = pow(xs(0) - p, 2) + pow(xs(1), 2);  // 最小点在(p,0)
//
//        auto nlp = SXDict{
//                {"x", xs},
//                {"f", obj},
//                {"g", SX()},
//                {"p", p}
//        };
//        SQPOptimizationSolver solver(nlp);
//
//        auto arg = DMDict{
//                {"lbx", {-100, -100}},
//                {"ubx", {100, 100}},
//                {"lbg", {}},
//                {"ubg", {}},
//                {"p", {5}}  // 设置参数p=5
//        };
//        std::cout << "预期解：x1=5, x2=0" << std::endl;
//        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
//    }
//
//    {
//        std::cout << "\n===== 测试案例7：带边界约束的二次规划 =====" << std::endl;
//        SX xs = SX::sym("x", 2);
//        SX obj = pow(xs(0) - 3, 2) + pow(xs(1) - 4, 2);  // 最小点在(3,4)
//
//        auto nlp = SXDict{
//                {"x", xs},
//                {"f", obj},
//                {"g", SX()},
//                {"p", SX()}
//        };
//        SQPOptimizationSolver solver(nlp);
//
//        auto arg = DMDict{
//                {"lbx", {0, 0}},  // 变量下界
//                {"ubx", {2, 3}},  // 变量上界
//                {"lbg", {}},
//                {"ubg", {}},
//                {"p", {}}
//        };
//        std::cout << "预期解：x1=2, x2=3 (受边界约束)" << std::endl;
//        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
//    }

//    {
//        std::cout << "\n===== 测试案例8：非凸二次规划 =====" << std::endl;
//        SX xs = SX::sym("x", 2);
//        // 非凸目标函数，有鞍点
//        SX obj = xs(0)*xs(0) - xs(1)*xs(1);
//        SX g = SX::vertcat({pow(xs(0), 2) + pow(xs(1), 2) - 1});  // 约束: x1^2 + x2^2 <= 1
//
//        auto nlp = SXDict{
//                {"x", xs},
//                {"f", obj},
//                {"g", g},
//                {"p", SX()}
//        };
//        SQPOptimizationSolver solver(nlp);
//
//        auto arg = DMDict{
//                {"lbx", {-100, -100}},
//                {"ubx", {100, 100}},
//                {"lbg", {-INF}},
//                {"ubg", {1}},
//                {"p", {}}
//        };
//        std::cout << "预期解：取决于初始点和求解器特性" << std::endl;
//        std::cout << "实际解：" << solver.getOptimalSolution(arg) << std::endl;
//    }

    std::cout << "\n========== 测试完成 ==========\n" << std::endl;
}

int main() {
    test_all_qp_cases();
    return 0;
}


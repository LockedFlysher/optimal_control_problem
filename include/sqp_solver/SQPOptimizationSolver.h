#pragma once

#include "sqp_solver/AutoDifferentiator.h"
#include "sqp_solver/CuCaQP.h"
#include "ament_index_cpp/get_package_share_directory.hpp"

class SQPOptimizationSolver {
public:
/**
* @brief 
* @note None
* @param None
* @retval None
*/

    explicit SQPOptimizationSolver(::casadi::SXDict nlp);
/**
* @brief SQP求解
* @note 如果是MPC问题初始点是通过限制第一项的lbx和ubx来完成的，和OptimalControlProblem一致
* @param <string,DM>字典，用到了lbx,ubx,lbg,ubg
* @retval DMDict result
*/
    casadi::DMDict getOptimalSolution(const DMDict &arg);

private:
    //    其实我不太确定这里到底需要多少个自动微分器
    std::shared_ptr <AutoDifferentiator> objectiveFunctionAutoDifferentiatorPtr_;
    std::shared_ptr <AutoDifferentiator> constraintsAutoDifferentiator_;
    CuCaQP qpSolver_;

    int stepNum_{10};
//    LineSearch使用到的
    double alpha_{0.2};
    DMDict result_;

    casadi::DM lowerBounds_;
    casadi::DM upperBounds_;

    /**
    * @brief
    * @note  依次序产生1. Hessian，gradient，A, l，u
    * @param [0]point
    * @retval [0]Hessian [1]Gradient [2]A [3]l [4]u
    */
    casadi::Function localSystemFunction_;

private:
    casadi::DMVector getLocalSystem(const DMDict &arg);

};

//
// Created by lock on 2025/2/17.
//
#include "sqp_solver/AutoDifferentiator.h"

AutoDifferentiator::AutoDifferentiator(const casadi::SX& variables, const casadi::SX& expression)
        : dim_(variables.size1())
        , x_(variables)
        , expr_(expression) {
    try {
        // 创建函数对象，使用引用避免拷贝
        const std::vector<casadi::SX> input = {x_};
        const std::vector<casadi::SX> output = {expr_};
        F_ = casadi::Function("F", input, output);

        if(expression.size1()==1){
//            单一函数，可以求梯度和Hessian，Jacobian也是可以求的
            const casadi::SX g = casadi::SX::gradient(expr_, x_);
            G_ = casadi::Function("G", input, {g});
            const casadi::SX h = casadi::SX::hessian(expr_, x_);
            H_ = casadi::Function("H", input, {h});
            const casadi::SX J = casadi::SX::jacobian(expr_, x_);
            J_ = casadi::Function("J", input, {J});
        }if(expression.size1()>1){
//            如果函数多就可以求Jacobian
            const casadi::SX J = casadi::SX::jacobian(expr_, x_);
            J_ = casadi::Function("J", input, {J});
        }
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("构造函数失败: " + std::string(e.what()));
    }
}

void AutoDifferentiator::validateInputDimension(const casadi::DM& point) const {
    if (point.size2() != 1) {
        throw AutoDifferentiatorException("你不能输入多个点");
    }
    if (point.size1() != x_.size1()) {
        throw AutoDifferentiatorException("输入维度不匹配: 期望 " +
                                          std::to_string(x_.size1()) + ", 实际 " + std::to_string(point.size1()));
    }
}

void AutoDifferentiator::validateInputDimension(const casadi::SX& point) const {
    if (point.size2() != 1) {
        throw AutoDifferentiatorException("你不能输入多个点");
    }
    if (point.size1() != x_.size1()) {
        throw AutoDifferentiatorException("输入维度不匹配: 期望 " +
                                          std::to_string(x_.size1()) + ", 实际 " + std::to_string(point.size1()));
    }
}

casadi::SX AutoDifferentiator::getExpression() const {
    try {
        return F_(x_)[0];
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("获取表达式失败: " + std::string(e.what()));
    }
}
casadi::DM AutoDifferentiator::getJacobian(const casadi::DM& point) const {
    if(expr_.size1()==1){
        throw AutoDifferentiatorException("表达式是标量，请使用getGradient");
    }
    try {
        validateInputDimension(point);
        if (verbose_) { std::cout << point << std::endl; }
        return J_(point)[0];
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("计算Jacobian失败: " + std::string(e.what()));
    }
}

casadi::DM AutoDifferentiator::getGradient(const casadi::DM& point) const {
    if(expr_.size1()>1){
        throw AutoDifferentiatorException("表达式是向量，请使用getJacobian");
    }
    try {
        validateInputDimension(point);
        if (verbose_) { std::cout << point << std::endl; }
        return G_(point)[0];
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("计算梯度失败: " + std::string(e.what()));
    }
}
casadi::SX AutoDifferentiator::getJacobian(const casadi::SX& point) const {
    if(expr_.size1()==1){
        throw AutoDifferentiatorException("表达式是标量，请使用getGradient");
    }
    try {
        validateInputDimension(point);
        if (verbose_) { std::cout << point << std::endl; }
        return J_(point)[0];
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("计算Jacobian失败: " + std::string(e.what()));
    }
}

casadi::SX AutoDifferentiator::getGradient(const casadi::SX& point) const {
    if(expr_.size1()>1){
        throw AutoDifferentiatorException("表达式是向量，请使用getJacobian");
    }
    try {
        validateInputDimension(point);
        if (verbose_) { std::cout << point << std::endl; }
        return G_(point)[0];
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("计算梯度失败: " + std::string(e.what()));
    }
}

casadi::SX AutoDifferentiator::getHessian(const casadi::SX& point) const {
    try {
        validateInputDimension(point);
        if (verbose_) { std::cout << point << std::endl; }
        return H_(point)[0];
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("计算Hessian失败: " + std::string(e.what()));
    }
}

casadi::DM AutoDifferentiator::getHessian(const casadi::DM& point) const {
    try {
        validateInputDimension(point);
        if (verbose_) { std::cout << point << std::endl; }
        return H_(point)[0];
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("计算Hessian失败: " + std::string(e.what()));
    }
}

casadi::SXVector AutoDifferentiator::getLinearization(const casadi::SX& point) {
    try {
        casadi::SX jacobian = this->getJacobian(point);
        casadi::SX b = -F_({point})[0];
        return {jacobian, b};
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("线性化失败: " + std::string(e.what()));
    }
}

/*
 * brief :
 * 返回第一个元素使Jacobian或者是梯度
 * 第二个元素是表达式（单一或组合的表达式）在该点的取值
 * */
casadi::DMVector AutoDifferentiator::getLinearization(const casadi::DM& point) {
    try {
        casadi::DM jacobian = this->getJacobian(point);
        casadi::DM b = F_({point})[0];
        return {jacobian, b};
    } catch (const std::exception& e) {
        throw AutoDifferentiatorException("线性化失败: " + std::string(e.what()));
    }
}

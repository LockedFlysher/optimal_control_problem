#pragma once

#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

class AutoDifferentiatorException : public std::runtime_error {
public:
    explicit AutoDifferentiatorException(const std::string& message)
            : std::runtime_error(message) {}
};

class AutoDifferentiator {
protected:
    casadi::SX x_;              // 符号变量
    casadi::SX expr_;           // 函数表达式
    casadi::Function F_;        // 函数对象，计算损失使用
    casadi::Function G_;        // 梯度函数对象，求一阶导的时候用到
    casadi::Function H_;        // Hessian矩阵函数对象
    casadi::Function J_;        // Jacobian矩阵函数对象
    size_t dim_;               // 输入维度
    bool verbose_{false};

    // 新增：验证输入点维度的辅助函数
    void validateInputDimension(const casadi::DM& point) const;
    void validateInputDimension(const casadi::SX& point) const;

public:
    // 构造函数使用初始化列表
    explicit AutoDifferentiator(const casadi::SX& variables, const casadi::SX& expression);

    // 禁用拷贝构造和赋值操作符，避免不必要的深拷贝
    AutoDifferentiator(const AutoDifferentiator&) = delete;
    AutoDifferentiator& operator=(const AutoDifferentiator&) = delete;

    // 允许移动构造和移动赋值
    AutoDifferentiator(AutoDifferentiator&&) noexcept = default;
    AutoDifferentiator& operator=(AutoDifferentiator&&) noexcept = default;

    // 析构函数
    ~AutoDifferentiator() = default;

    // Getter函数
    [[nodiscard]] const casadi::SX& getSymbolicVar() const noexcept { return x_; }
    [[nodiscard]] casadi::SX getExpression() const;
    [[nodiscard]] const casadi::Function& getJacobianFunction() const noexcept { return J_; }
    [[nodiscard]] const casadi::Function& getHessianFunction() const noexcept { return H_; }
    [[nodiscard]] const casadi::Function& getGradientFunction() const noexcept { return G_; }

    // 计算函数 - SX版本
    [[nodiscard]] casadi::SX getJacobian(const casadi::SX& point) const;
    [[nodiscard]] casadi::SX getGradient(const casadi::SX& point) const;
    [[nodiscard]] casadi::SX getHessian(const casadi::SX& point) const;

    // 计算函数 - DM版本
    [[nodiscard]] casadi::DM getJacobian(const casadi::DM& point) const;
    [[nodiscard]] casadi::DM getGradient(const casadi::DM& point) const;
    [[nodiscard]] casadi::DM getHessian(const casadi::DM& point) const;

    // 线性化函数
    [[nodiscard]] casadi::SXVector getLinearization(const casadi::SX& point);
    [[nodiscard]] casadi::DMVector getLinearization(const casadi::DM& point);

    // 调试开关
    void setVerbose(bool verbose) noexcept { verbose_ = verbose; }
    [[nodiscard]] bool isVerbose() const noexcept { return verbose_; }
};


#pragma once

#include <casadi/casadi.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <iostream>
#include <ament_index_cpp/get_package_share_directory.hpp>

/*
 * 负责的内容是构建求解器，求解器的调用应该由子类完成
 * 使用符合规范的YAML文件来初始化
 * 离散的MPC构建器
 * */
class OptimalControlProblem {
private:
    // Frame是用来记录状态和输入变量的结构体，包括总大小、字段名称和偏移量等信息。
    struct Frame {
//        totalSize是表示的状态量或者输入量的全体变量的数量
        int totalSize;
        std::vector<std::pair<std::string, int>> fields;
//        unordered_map表示没有顺序的字典，使用find来查找
        std::unordered_map<std::string, int> fieldOffsets;
    };
    YAML::Node configNode_;
    Frame statusFrame_, inputFrame_;
    casadi::SX statusVariables, inputVariables;
    std::vector<casadi::SX> constraints_;
    std::vector<std::string> constraintNames_;
    std::vector<casadi::DM> constraintLowerBounds_;
    std::vector<casadi::DM> constraintUpperBounds_;
    casadi::SX cost_ = 0;
    int horizon_;
    float dt_;
    ::casadi::DM initialGuess_;
    std::vector<casadi::DM> statusUpperBounds_;
    std::vector<casadi::DM> statusLowerBounds_;
    std::vector<casadi::DM> inputUpperBounds_;
    std::vector<casadi::DM> inputLowerBounds_;
    bool setInitialGuess_{false};
    bool firstTime_{true};
    ::casadi::DM optimalTrajectory_;
    // OCP问题构建和求解的接口
    bool genCode_{false};
    bool loadLib_{false};
    bool verbose_{true};
    std::string packagePath_;
    ::casadi::Function IPOPTSolver_;
    ::casadi::Function SQPSolver_;
    ::casadi::Function libIPOPTSolver_;
    ::casadi::Function libSQPSolver_;

    enum OCPStateMachine {
        UN_INITIALIZED = 0,
        FRAME_INITIALIZED = 1
    };
private:
    //    通过解析YAML文件内的节点完成帧变量的初始化
    void parseOCPBounds();
    /*
     * 直接给定所有数据帧的状态变量的上界和下界
     * */
    void setStatusBounds(const ::casadi::DM& lowerBound, const ::casadi::DM& upperBound);

    //    使用一帧的下界完成所有变量的下界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的下界添加到整个状态变量的下界中
    void coverLowerStatusBounds(const casadi::SX &oneFrameLowerBound) {
        statusLowerBounds_.clear();
        for (int i = 0; i < horizon_; ++i) {
            statusLowerBounds_.emplace_back(oneFrameLowerBound);
        }
    }

//    使用一帧的上界完成所有变量的上界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的上界添加到整个状态变量的上界中
    void coverUpperStatusBounds(const casadi::SX &oneFrameUpperBound) {
        statusUpperBounds_.clear();
        for (int i = 0; i < horizon_; ++i) {
            statusUpperBounds_.emplace_back(oneFrameUpperBound);
        }
    }

//    使用一帧的下界完成所有变量的下界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的下界添加到整个状态变量的下界中
    void coverLowerInputBounds(const casadi::SX &oneFrameLowerBound) {
        inputLowerBounds_.clear();
        for (int i = 0; i < horizon_; ++i) {
            inputLowerBounds_.emplace_back(oneFrameLowerBound);
        }
    }

//    使用一帧的上界完成所有变量的上界的设置，首先会clear掉原来的数据，防止重复添加，然后把一个帧的上界添加到整个状态变量的上界中
    void coverUpperInputBounds(const casadi::SX &oneFrameLowerBound) {
        inputUpperBounds_.clear();
        for (int i = 0; i < horizon_; ++i) {
            inputUpperBounds_.emplace_back(oneFrameLowerBound);
        }
    }

    static void initializeFrameWithYAML(Frame &frame, const YAML::Node &config);

public:
    ::casadi::DM getOptimalInputFirstFrame();
    ::casadi::SX getReference();
    /*
     * 设置初始猜测解
     * */
    void setInitialGuess(const ::casadi::DM&);
    /*
     * 返回最优解变量，不做计算
     * */
    ::casadi::DM getOptimalTrajectory();
    //    求解功能使用到的变量们
    ::casadi::SX reference_;

    /*
     * 根据配置文件决定
     * 1.是否生成c代码和动态链接库
     * 2.是否使用SQP类型的求解器
     * 3.是否从.so加载求解器
     * */
    void genSolver();
    /*
     * 把OCP的当前的状态输入到这里，reference的具体的数值发到这里，就行了
     * */
    void computeOptimalTrajectory(const ::casadi::DM &statusFrame, const ::casadi::DM &reference);
    /*
     * 构造函数
     * */
    explicit OptimalControlProblem(const std::string &configFilePath);

    //    子类在创建变量之间的约束的时候会频繁使用，通过帧的ID和变量的名称拿到SX
    ::casadi::SX getStatusVariable(int frameID, const std::string &fieldName) const;
    //    子类在创建变量之间的约束的时候会频繁使用，通过帧的ID和变量的名称拿到SX
    ::casadi::SX getInputVariable(int frameID, const std::string &fieldName) const;
    /*
     * costFunction是从这里进行
     * */
    void addCost(const casadi::SX &cost);
    /*
     * 添加不等式约束其实是一个通用的函数，是添加等式约束的基础函数
     * */
    void addInequalityConstraint(const std::string &constraintName,
                                 const casadi::DM &lowerBound,
                                 const casadi::SX &expression,
                                 const casadi::DM &upperBound);

    void addEquationConstraint(const std::string &constraintName, const casadi::SX &leftSX, const casadi::SX &rightSX);

    void addEquationConstraint(const std::string &constraintName, const casadi::SX &expression);

    void saveConstraintsToCSV(const std::string &filename);

//    取得所有帧的状态变量
    casadi::SX getStatusVariables() const;
//    取得所有帧的输入变量
    casadi::SX getInputVariables() const;

    /*
     * OCP创建的时候取得损失函数
     * */
    casadi::SX getCostFunction() const;

    /*
     * OCP创建的时候用得到这个限制
     * */
    std::vector<casadi::SX> getConstraints() const;

    /*
     * OCP创建的时候用得到下界和上界
     * */
    std::vector<casadi::DM> getConstraintUpperBounds() const;

    /*
     * OCP创建的时候用得到下界和上界
     * */
    std::vector<casadi::DM> getConstraintLowerBounds() const;

    std::vector<casadi::DM> getStatusLowerBounds() const;

    std::vector<casadi::DM> getStatusUpperBounds() const;

    std::vector<casadi::DM> getInputLowerBounds() const;

    std::vector<casadi::DM> getInputUpperBounds() const;
    /*
     * 子类必须实现这个函数添加约束
     * */
    virtual void deployConstraintsAndAddCost() = 0;
    /*
     * 取得自变量的下界
     * */
    casadi::DM getVariableLowerBounds() const;
    /*
     * 取得自变量的上界
     * */
    casadi::DM getVariableUpperBounds() const;
    /*
     * 取得时间间隔
     * */
    float getDt() const;

/*
 * 取得预测步数
 * */
    int getHorizon() const;

    /*
     *  拿到状态的所有Frame的大小
     * */
    int getStatusFrameSize() const;

    /*
     *  拿到输入的所有Frame的大小
     * */
    int getInputFrameSize() const;

    /*
     * 检查变量的维度
     * */
    bool solverInputCheck(std::map<std::string, ::casadi::DM> arg);

// if you want to define the status and input bound, override this function
    friend std::ostream &operator<<(std::ostream &os, const OptimalControlProblem &ocp);
};

std::ostream &operator<<(std::ostream &os, const OptimalControlProblem &ocp);

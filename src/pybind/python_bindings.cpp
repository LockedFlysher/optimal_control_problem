////
//// Created by lock on 25-4-21.
////
//#include "pybind11/pybind11.h"
//#include "pybind11/stl.h"
//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
//#include <pybind11/eigen.h>
//#include <pybind11/numpy.h>
//#include <pybind11/functional.h>
//#include <pybind11/operators.h>
//#include <pybind11/complex.h>
//#include "optimal_control_problem/OptimalControlProblem.h"
//
//// 为 casadi 类型创建转换
//#include <pybind11/stl_bind.h>
//#include <casadi/casadi.hpp>
//
//namespace py = pybind11;
//
//// 用于转换 casadi::SX 和 casadi::DM 类型
//namespace pybind11 { namespace detail {
//        // casadi::SX 类型转换器
//        template <> struct type_caster<casadi::SX> {
//        public:
//        PYBIND11_TYPE_CASTER(casadi::SX, _("casadi.SX"));
//
//            // Python -> C++ 转换
//            bool load(handle src, bool convert) {
//                // 使用 casadi 的 Python API 转换
//                try {
//                    if (!py::isinstance(src, py::module::import("casadi").attr("SX")))
//                        return false;
//
//                    // 通过 casadi 的 Python API 获取 C++ 对象
//                    value = py::cast<casadi::SX>(src);
//                    return true;
//                } catch (const std::exception &) {
//                    return false;
//                }
//            }
//
//            // C++ -> Python 转换
//            static handle cast(const casadi::SX &src, return_value_policy policy, handle parent) {
//                // 将 C++ casadi::SX 对象转换为 Python casadi.SX 对象
//                py::object casadi_module = py::module::import("casadi");
//                py::object sx_constructor = casadi_module.attr("SX");
//                // 使用 py::cast 创建一个临时 object
//                py::object result = sx_constructor();
//                // 将 C++ 对象的值复制到 Python 对象
//                result.attr("__setstate__")(py::cast(src.get_str()));
//                return result.release();
//            }
//        };
//
//        // casadi::DM 类型转换器
//        template <> struct type_caster<casadi::DM> {
//        public:
//        PYBIND11_TYPE_CASTER(casadi::DM, _("casadi.DM"));
//
//            // Python -> C++ 转换
//            bool load(handle src, bool convert) {
//                try {
//                    if (!py::isinstance(src, py::module::import("casadi").attr("DM")))
//                        return false;
//
//                    value = py::cast<casadi::DM>(src);
//                    return true;
//                } catch (const std::exception &) {
//                    return false;
//                }
//            }
//
//            // C++ -> Python 转换
//            static handle cast(const casadi::DM &src, return_value_policy policy, handle parent) {
//                py::object casadi_module = py::module::import("casadi");
//                py::object dm_constructor = casadi_module.attr("DM");
//                // 创建一个新的 DM 对象
//                py::object result;
//
//                // 获取 src 的维度
//                std::vector<int> dims = src.size();
//                if (dims.size() == 1) {
//                    // 向量
//                    result = dm_constructor(py::cast(dims[0]), 1);
//                } else if (dims.size() == 2) {
//                    // 矩阵
//                    result = dm_constructor(py::cast(dims[0]), py::cast(dims[1]));
//                } else {
//                    // 默认创建空 DM
//                    result = dm_constructor();
//                }
//
//                // 复制数据
//                std::vector<double> data = std::vector<double>(src.nonzeros().begin(), src.nonzeros().end());
//                result.attr("set_nonzeros")(py::cast(data));
//
//                return result.release();
//            }
//        };
//
//        // casadi::SXVector 类型转换器
//        template <> struct type_caster<casadi::SXVector> {
//        public:
//        PYBIND11_TYPE_CASTER(casadi::SXVector, _("List[casadi.SX]"));
//
//            // Python -> C++ 转换
//            bool load(handle src, bool convert) {
//                try {
//                    if (!py::isinstance<py::list>(src))
//                        return false;
//
//                    py::list list = src.cast<py::list>();
//                    value.resize(list.size());
//
//                    for (size_t i = 0; i < list.size(); ++i) {
//                        if (!py::isinstance(list[i], py::module::import("casadi").attr("SX")))
//                            return false;
//                        value[i] = list[i].cast<casadi::SX>();
//                    }
//                    return true;
//                } catch (const std::exception &) {
//                    return false;
//                }
//            }
//
//            // C++ -> Python 转换
//            static handle cast(const casadi::SXVector &src, return_value_policy policy, handle parent) {
//                py::list result;
//                py::object casadi_module = py::module::import("casadi");
//                py::object sx_constructor = casadi_module.attr("SX");
//
//                for (const auto &sx : src) {
//                    // 创建一个新的 SX 对象
//                    py::object sx_obj = sx_constructor();
//                    // 复制数据
//                    sx_obj.attr("__setstate__")(py::cast(sx.get_str()));
//                    result.append(sx_obj);
//                }
//                return result.release();
//            }
//        };
//
//        // casadi::DMVector 类型转换器
//        template <> struct type_caster<casadi::DMVector> {
//        public:
//        PYBIND11_TYPE_CASTER(casadi::DMVector, _("List[casadi.DM]"));
//
//            // Python -> C++ 转换
//            bool load(handle src, bool convert) {
//                try {
//                    if (!py::isinstance<py::list>(src))
//                        return false;
//
//                    py::list list = src.cast<py::list>();
//                    value.resize(list.size());
//
//                    for (size_t i = 0; i < list.size(); ++i) {
//                        if (!py::isinstance(list[i], py::module::import("casadi").attr("DM")))
//                            return false;
//                        value[i] = list[i].cast<casadi::DM>();
//                    }
//                    return true;
//                } catch (const std::exception &) {
//                    return false;
//                }
//            }
//
//            // C++ -> Python 转换
//            static handle cast(const casadi::DMVector &src, return_value_policy policy, handle parent) {
//                py::list result;
//                py::object casadi_module = py::module::import("casadi");
//                py::object dm_constructor = casadi_module.attr("DM");
//
//                for (const auto &dm : src) {
//                    // 创建一个新的 DM 对象
//                    py::object dm_obj;
//
//                    // 获取 dm 的维度
//                    std::vector<int> dims = dm.size();
//                    if (dims.size() == 1) {
//                        // 向量
//                        dm_obj = dm_constructor(py::cast(dims[0]), 1);
//                    } else if (dims.size() == 2) {
//                        // 矩阵
//                        dm_obj = dm_constructor(py::cast(dims[0]), py::cast(dims[1]));
//                    } else {
//                        // 默认创建空 DM
//                        dm_obj = dm_constructor();
//                    }
//
//                    // 复制数据
//                    std::vector<double> data = std::vector<double>(dm.nonzeros().begin(), dm.nonzeros().end());
//                    dm_obj.attr("set_nonzeros")(py::cast(data));
//
//                    result.append(dm_obj);
//                }
//                return result.release();
//            }
//        };
//
//        // casadi::Function 类型转换器
//        template <> struct type_caster<casadi::Function> {
//        public:
//        PYBIND11_TYPE_CASTER(casadi::Function, _("casadi.Function"));
//
//            // Python -> C++ 转换
//            bool load(handle src, bool convert) {
//                try {
//                    if (!py::isinstance(src, py::module::import("casadi").attr("Function")))
//                        return false;
//
//                    value = py::cast<casadi::Function>(src);
//                    return true;
//                } catch (const std::exception &) {
//                    return false;
//                }
//            }
//
//            // C++ -> Python 转换
//            static handle cast(const casadi::Function &src, return_value_policy policy, handle parent) {
//                py::object casadi_module = py::module::import("casadi");
//
//                // 获取函数的名称和其他属性
//                std::string name = src.name();
//                std::vector<casadi::SX> sx_in = src.sx_in();
//                std::vector<casadi::SX> sx_out = src.sx_out();
//
//                // 创建输入和输出列表
//                py::list py_in;
//                py::list py_out;
//
//                // 转换输入
//                for (const auto &sx : sx_in) {
//                    py::object sx_obj = casadi_module.attr("SX")();
//                    sx_obj.attr("__setstate__")(py::cast(sx.get_str()));
//                    py_in.append(sx_obj);
//                }
//
//                // 转换输出
//                for (const auto &sx : sx_out) {
//                    py::object sx_obj = casadi_module.attr("SX")();
//                    sx_obj.attr("__setstate__")(py::cast(sx.get_str()));
//                    py_out.append(sx_obj);
//                }
//
//                // 创建函数对象
//                py::object func = casadi_module.attr("Function")(
//                        py::cast(name),
//                        py_in,
//                        py_out,
//                        py::dict()
//                );
//
//                return func.release();
//            }
//        };
//    }}
//
//// 为 YAML::Node 创建类型转换器
//namespace pybind11 { namespace detail {
//        template <> struct type_caster<YAML::Node> {
//        public:
//        PYBIND11_TYPE_CASTER(YAML::Node, _("dict"));
//
//            // Python -> C++ 转换
//            bool load(handle src, bool convert) {
//                if (!py::isinstance<py::dict>(src))
//                    return false;
//
//                py::dict dict = src.cast<py::dict>();
//
//                // 创建一个空的 YAML::Node
//                value = YAML::Node(YAML::NodeType::Map);
//
//                // 递归转换字典为 YAML::Node
//                for (auto item : dict) {
//                    std::string key = item.first.cast<std::string>();
//                    py::object val = item.second;
//
//                    if (py::isinstance<py::dict>(val)) {
//                        value[key] = handle_dict(val.cast<py::dict>());
//                    } else if (py::isinstance<py::list>(val)) {
//                        value[key] = handle_list(val.cast<py::list>());
//                    } else if (py::isinstance<py::str>(val)) {
//                        value[key] = val.cast<std::string>();
//                    } else if (py::isinstance<py::int_>(val)) {
//                        value[key] = val.cast<int>();
//                    } else if (py::isinstance<py::float_>(val)) {
//                        value[key] = val.cast<double>();
//                    } else if (py::isinstance<py::bool_>(val)) {
//                        value[key] = val.cast<bool>();
//                    }
//                }
//
//                return true;
//            }
//
//            // C++ -> Python 转换
//            static handle cast(const YAML::Node &src, return_value_policy policy, handle parent) {
//                return convert_yaml_to_python(src).release();
//            }
//
//        private:
//            // 辅助函数：将 Python 字典转换为 YAML::Node
//            static YAML::Node handle_dict(const py::dict &dict) {
//                YAML::Node node(YAML::NodeType::Map);
//                for (auto item : dict) {
//                    std::string key = item.first.cast<std::string>();
//                    py::object val = item.second;
//
//                    if (py::isinstance<py::dict>(val)) {
//                        node[key] = handle_dict(val.cast<py::dict>());
//                    } else if (py::isinstance<py::list>(val)) {
//                        node[key] = handle_list(val.cast<py::list>());
//                    } else if (py::isinstance<py::str>(val)) {
//                        node[key] = val.cast<std::string>();
//                    } else if (py::isinstance<py::int_>(val)) {
//                        node[key] = val.cast<int>();
//                    } else if (py::isinstance<py::float_>(val)) {
//                        node[key] = val.cast<double>();
//                    } else if (py::isinstance<py::bool_>(val)) {
//                        node[key] = val.cast<bool>();
//                    }
//                }
//                return node;
//            }
//
//            // 辅助函数：将 Python 列表转换为 YAML::Node
//            static YAML::Node handle_list(const py::list &list) {
//                YAML::Node node(YAML::NodeType::Sequence);
//                for (size_t i = 0; i < list.size(); ++i) {
//                    py::object val = list[i];
//
//                    if (py::isinstance<py::dict>(val)) {
//                        node.push_back(handle_dict(val.cast<py::dict>()));
//                    } else if (py::isinstance<py::list>(val)) {
//                        node.push_back(handle_list(val.cast<py::list>()));
//                    } else if (py::isinstance<py::str>(val)) {
//                        node.push_back(val.cast<std::string>());
//                    } else if (py::isinstance<py::int_>(val)) {
//                        node.push_back(val.cast<int>());
//                    } else if (py::isinstance<py::float_>(val)) {
//                        node.push_back(val.cast<double>());
//                    } else if (py::isinstance<py::bool_>(val)) {
//                        node.push_back(val.cast<bool>());
//                    }
//                }
//                return node;
//            }
//
//            // 辅助函数：将 YAML::Node 转换为 Python 对象
//            static py::object convert_yaml_to_python(const YAML::Node &node) {
//                if (!node.IsDefined()) {
//                    return py::none();
//                }
//
//                if (node.IsMap()) {
//                    py::dict dict;
//                    for (const auto &it : node) {
//                        std::string key = it.first.as<std::string>();
//                        dict[py::str(key)] = convert_yaml_to_python(it.second);
//                    }
//                    return dict;
//                } else if (node.IsSequence()) {
//                    py::list list;
//                    for (size_t i = 0; i < node.size(); ++i) {
//                        list.append(convert_yaml_to_python(node[i]));
//                    }
//                    return list;
//                } else if (node.IsScalar()) {
//                    // 尝试不同的标量类型转换
//                    try {
//                        return py::cast(node.as<int>());
//                    } catch (...) {
//                        try {
//                            return py::cast(node.as<double>());
//                        } catch (...) {
//                            try {
//                                return py::cast(node.as<bool>());
//                            } catch (...) {
//                                return py::cast(node.as<std::string>());
//                            }
//                        }
//                    }
//                }
//
//                return py::none();
//            }
//        };
//    }}
//
//// 创建一个派生类，用于在 Python 中继承 OptimalControlProblem
//class PyOptimalControlProblem : public OptimalControlProblem {
//public:
//    // 使用父类的构造函数
//    using OptimalControlProblem::OptimalControlProblem;
//
//    // 覆盖纯虚函数
//    void deployConstraintsAndAddCost() override {
//        PYBIND11_OVERRIDE_PURE(
//                void,                      // 返回类型
//                OptimalControlProblem,     // 父类
//                deployConstraintsAndAddCost // 方法名
//        );
//    }
//};
//
//PYBIND11_MODULE(ocp_module, m) {
//    m.doc() = "Python bindings for OptimalControlProblem";
//
//    // 绑定 SolverSettings::SolverType 枚举
//    py::enum_<OptimalControlProblem::SolverSettings::SolverType>(m, "SolverType")
//            .value("IPOPT", OptimalControlProblem::SolverSettings::SolverType::IPOPT)
//            .value("SQP", OptimalControlProblem::SolverSettings::SolverType::SQP)
//            .value("CUDA_SQP", OptimalControlProblem::SolverSettings::SolverType::CUDA_SQP)
//            .value("MIXED", OptimalControlProblem::SolverSettings::SolverType::MIXED)
//            .export_values();
//
//    // 绑定 OptimalControlProblem 类
//    py::class_<OptimalControlProblem, PyOptimalControlProblem>(m, "OptimalControlProblem")
//            .def(py::init<YAML::Node>())
//            .def("set_solver_type", &OptimalControlProblem::setSolverType)
//            .def("get_solver_type", &OptimalControlProblem::getSolverType)
//            .def("get_reference", &OptimalControlProblem::getReference)
//            .def("get_optimal_trajectory", &OptimalControlProblem::getOptimalTrajectory)
//            .def("gen_solver", &OptimalControlProblem::genSolver)
//            .def("compute_optimal_trajectory", &OptimalControlProblem::computeOptimalTrajectory)
//            .def("get_optimal_input_first_frame", &OptimalControlProblem::getOptimalInputFirstFrame)
//            .def("set_reference", &OptimalControlProblem::setReference)
//            .def("add_scalar_cost", &OptimalControlProblem::addScalarCost)
//            .def("add_vector_cost", py::overload_cast<const casadi::DM &, const casadi::SX &>(&OptimalControlProblem::addVectorCost))
//            .def("add_vector_cost", py::overload_cast<const std::vector<double> &, const casadi::SX &>(&OptimalControlProblem::addVectorCost))
//            .def("add_inequality_constraint", &OptimalControlProblem::addInequalityConstraint)
//            .def("add_equation_constraint", py::overload_cast<const std::string &, const casadi::SX &, const casadi::SX &>(&OptimalControlProblem::addEquationConstraint))
//            .def("add_equation_constraint", py::overload_cast<const std::string &, const casadi::SX &>(&OptimalControlProblem::addEquationConstraint))
//            .def("get_cost_function", &OptimalControlProblem::getCostFunction)
//            .def("get_constraint_lower_bounds", &OptimalControlProblem::getConstraintLowerBounds)
//            .def("get_constraint_upper_bounds", &OptimalControlProblem::getConstraintUpperBounds)
//            .def("get_constraints", &OptimalControlProblem::getConstraints)
//            .def("deploy_constraints_and_add_cost", &OptimalControlProblem::deployConstraintsAndAddCost)
//            .def("solver_input_check", &OptimalControlProblem::solverInputCheck)
//            .def("gen_code", &OptimalControlProblem::genCode)
//            .def_readwrite("reference_", &OptimalControlProblem::reference_)
//            .def_readwrite("total_cost_", &OptimalControlProblem::totalCost_);
//}

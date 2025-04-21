//
// Created by lock on 25-4-21.
//
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
PYBIND11_MODULE(_my_package,m){
    std::cout<<"绑定成功"<<std::endl;
};
# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.21)
# 定义项目名称
project(optimal_control_problem)
# 查找ament_cmake包（ROS2基础构建系统）
find_package(ament_cmake REQUIRED)
link_directories(/usr/local/cuda/lib64)

##################################################################################################
#################################### 用户配置区开始 ##############################################
##################################################################################################

######################################## 基础构建设置 ###########################################
# 设置构建类型为RelWithDebInfo（带调试信息的发布版本）
set(CMAKE_BUILD_TYPE Debug)
# 导出编译命令，用于IDE和工具的代码补全
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

######################################## 宏定义设置 ###########################################
## 设置配置文件路径宏
#add_definitions(-DCONFIG_FILE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml")
## 设置包路径宏
add_definitions(-DPACKAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

######################################## 依赖项配置 ###########################################
# 设置第三方库依赖
set(dependencies
        CASADI      # CasADi优化库
        Eigen3      # Eigen矩阵运算库
        yaml-cpp    # YAML配置文件解析库
        OsqpEigen
)

# 设置ROS包依赖
set(parent_packages
        ament_index_cpp
)

######################################## 构建目标配置 ########################################
# 定义构建目标列表
# 格式：
# set(TARGET_NAME "目标名称")
# set(${TARGET_NAME}_TYPE "EXECUTABLE|LIBRARY|INTERFACE")
# set(${TARGET_NAME}_SOURCES "源文件列表或目录")

# 示例：定义一个可执行文件目标
set(TARGET1 "optimal_control_problem")
set(${TARGET1}_TYPE "LIBRARY")
aux_source_directory(src ${TARGET1}_SOURCES)
aux_source_directory(src/OCP_config ${TARGET1}_SOURCES)
aux_source_directory(src/sqp_solver ${TARGET1}_SOURCES)

set(TARGET2 "test_sqp")
set(${TARGET2}_TYPE "EXECUTABLE")
aux_source_directory(src ${TARGET2}_SOURCES)
aux_source_directory(src/OCP_config ${TARGET2}_SOURCES)
aux_source_directory(src/sqp_solver ${TARGET2}_SOURCES)
aux_source_directory(test ${TARGET2}_SOURCES)

# 将所有目标名称添加到列表中
set(ALL_TARGETS
        ${TARGET1}
        ${TARGET2}
        # 在此添加更多目标
)

##################################################################################################
#################################### 用户配置区结束 ##############################################
##################################################################################################
######################################## CasADi处理 ##########################################
# 查找CasADi库文件
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(CASADI casadi)
endif()
##################################################################################################
#################################### 用户配置区结束 ##############################################
##################################################################################################

# 查找ament_cmake包（ROS2基础构建系统）
find_package(ament_cmake REQUIRED)

##################################################################################################
#################################### 自动处理区开始 ##############################################
##################################################################################################

######################################## 依赖处理 #############################################
# 查找所有依赖包
foreach (dependency IN LISTS dependencies parent_packages)
    find_package(${dependency} REQUIRED)
endforeach ()

# 设置包含目录
include_directories(include)

# 自动处理依赖包的包含目录（支持大小写变量名）
foreach (dependency IN LISTS dependencies parent_packages)
    string(TOUPPER "${dependency}" DEP_UPPER)
    string(TOLOWER "${dependency}" DEP_LOWER)

    message(STATUS "Checking include directories for ${dependency}")
    if (DEFINED ${DEP_UPPER}_INCLUDE_DIRS OR DEFINED ${DEP_LOWER}_INCLUDE_DIRS)
        if (DEFINED ${DEP_UPPER}_INCLUDE_DIRS)
            include_directories(${${DEP_UPPER}_INCLUDE_DIRS})
            message(STATUS "Using ${DEP_UPPER}_INCLUDE_DIRS")
        else ()
            include_directories(${${DEP_LOWER}_INCLUDE_DIRS})
            message(STATUS "Using ${DEP_LOWER}_INCLUDE_DIRS")
        endif ()
    else ()
        message(STATUS "No include directories found for ${dependency}")
    endif ()
endforeach ()
######################################## 目标构建 ###########################################
# 处理每个目标
foreach (TARGET_NAME IN LISTS ALL_TARGETS)
    message(STATUS "Processing target: ${TARGET_NAME}")

    if (${${TARGET_NAME}_TYPE} STREQUAL "EXECUTABLE")
        # 添加可执行文件目标
        add_executable(${TARGET_NAME} ${${TARGET_NAME}_SOURCES})

        # 设置安装规则
        install(TARGETS ${TARGET_NAME}
                EXPORT export_${PROJECT_NAME}
                RUNTIME DESTINATION lib/${PROJECT_NAME}
        )

    elseif (${${TARGET_NAME}_TYPE} STREQUAL "LIBRARY")
        # 添加库目标
        add_library(${TARGET_NAME} SHARED ${${TARGET_NAME}_SOURCES})

        # 设置安装规则
        install(TARGETS ${TARGET_NAME}
                EXPORT export_${PROJECT_NAME}
                LIBRARY DESTINATION lib
                ARCHIVE DESTINATION lib
        )
        ament_export_libraries(${PROJECT_NAME})

    elseif (${${TARGET_NAME}_TYPE} STREQUAL "INTERFACE")
        # 接口库不需要源文件
        add_library(${TARGET_NAME} INTERFACE)
        message(STATUS "Creating interface-only library: ${TARGET_NAME}")
    endif ()

    # 如果不是接口库，设置依赖
    if (NOT ${${TARGET_NAME}_TYPE} STREQUAL "INTERFACE")
        # 用户定义： 系统库依赖
        target_link_libraries(${TARGET_NAME}
                ${CASADI_LIBRARIES}  # CasADi
                yaml-cpp             # YAML-CPP
                OsqpEigen::OsqpEigen
                /usr/local/cuda/lib64/libcublas.so
                /usr/local/cuda/lib64/libcusparse.so
        )
        ament_target_dependencies(${TARGET_NAME} ${parent_packages})
    endif ()
endforeach ()

######################################## 通用安装配置 ######################################
# 安装头文件 / Install header files
install(DIRECTORY
        include/
        DESTINATION include)

# 获取当前目录下的所有文件夹
file(GLOB SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

# 遍历所有文件夹，排除特定目录
foreach (SUBDIR ${SUBDIRS})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR})
        # 排除 src、include、build 等目录
        if (NOT ${SUBDIR} STREQUAL "src" AND
                NOT ${SUBDIR} STREQUAL "build" AND
                NOT ${SUBDIR} STREQUAL ".git" AND
                NOT ${SUBDIR} STREQUAL "install")
            install(DIRECTORY
                    ${SUBDIR}
                    DESTINATION share/${PROJECT_NAME}/
            )
        endif ()
    endif ()
endforeach ()
######################################## ROS2包配置 ########################################
# 导出包依赖
ament_export_dependencies(${parent_packages})

# 导出包含目录
ament_export_include_directories(include)

# 生成包配置
ament_package()
cmake_minimum_required(VERSION 3.8)
project(graspnet_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraspPattern.msg"
  DEPENDENCIES std_msgs
)

install(DIRECTORY graspnet_ros
  DESTINATION ${PYTHON_INSTALL_DIR}
)

install(DIRECTORY graspnet-baseline
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/graspnet_node DESTINATION lib/${PROJECT_NAME})

install(FILES meta.mat
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)

ament_package()

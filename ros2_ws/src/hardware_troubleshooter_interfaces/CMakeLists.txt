cmake_minimum_required(VERSION 3.8)
project(hardware_troubleshooter_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MeasureVoltage.action"
  "msg/TestPoint.msg"
  "msg/RegistrationState.msg"
  "srv/ListTestPoints.srv"
  "srv/GetRigStatus.srv"
  "srv/GetRegistration.srv"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces action_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()

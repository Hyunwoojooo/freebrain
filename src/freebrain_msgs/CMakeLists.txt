cmake_minimum_required(VERSION 3.8)
project(freebrain_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SafetyStatus.msg"
  "msg/DetectedObject.msg"
  "msg/DetectedObjectArray.msg"
  "msg/SceneGraph.msg"
  "msg/DevelopmentalState.msg"
  "msg/CognitiveGoal.msg"
  "srv/ExecuteSkill.srv"
  "srv/QueryMemory.srv"
  "srv/StoreMemory.srv"
  "srv/DecomposeTask.srv"
  "srv/GenerateReward.srv"
  "action/TrainSkill.action"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs action_msgs
)

ament_package()

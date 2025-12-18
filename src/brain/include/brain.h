#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>
#include <rerun.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <vision_interface/msg/detections.hpp>
#include <vision_interface/msg/line_segments.hpp>
#include <vision_interface/msg/cal_param.hpp>
#include <vision_interface/msg/segmentation_result.hpp>
#include <game_controller_interface/msg/game_control_data.hpp>
#include <booster/robot/b1/b1_api_const.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "booster_interface/msg/odometer.hpp"
#include "booster_interface/msg/low_state.hpp"
#include "booster_interface/msg/raw_bytes_msg.hpp"
#include "booster_interface/msg/remote_controller_state.hpp"

#include "RoboCupGameControlData.h"
#include "team_communication_msg.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdexcept>
#include "brain/msg/kick.hpp"

#include "brain_data.h"
#include "brain_tree.h"
#include "robot_client.h"
#include "brain_config.h"
#include "walk.h"
#include "movehead.h"
#include "locator.h"
#include "brain_log.h"


using namespace std;


class Brain : public rclcpp::Node
{
public:
    std::shared_ptr<RobotClient> client;
    std::shared_ptr<BrainTree> tree;
    std::shared_ptr<BrainConfig> config;
    std::shared_ptr<BrainData> data;
    std::shared_ptr<Locator> locator;
    std::shared_ptr<BrainLog> log;
    
    Brain();
    ~Brain();
    void init();
    void tick();
    double msecsSince(rclcpp::Time time); // 특정 시간(timestamp) 이후 몇 밀리초가 지났는지 계산하는 유틸리티 함수

    void calibrateOdom(double x, double y, double theta);
    void updateFieldPos(GameObject& obj);
    void logDetection(const vector<GameObject>& objects);
    
    bool isBoundingBoxInCenter(BoundingBox boundingBox, double xRatio = 0.5, double yRatio = 0.5);


    // 행동 노드들 등록
    void registerWalkNodes(BT::BehaviorTreeFactory &factory){RegisterWalkNodes(factory, this);}
    void registerMoveHeadNodes(BT::BehaviorTreeFactory &factory){RegisterMoveHeadNodes(factory, this);}
    void registerLocatorNodes(BT::BehaviorTreeFactory &factory){RegisterLocatorNodes(factory, this);}
    
    // ROS callback 함수
    void gameControlCallback(const game_controller_interface::msg::GameControlData &msg);
    void detectionsCallback(const vision_interface::msg::Detections &msg);
    void fieldLineCallback(const vision_interface::msg::LineSegments &msg);
    void odometerCallback(const booster_interface::msg::Odometer &msg);

    
private:
    void loadConfig(); // config 불러오기
    
    // ROS subscription
    rclcpp::Subscription<game_controller_interface::msg::GameControlData>::SharedPtr gameControlSubscription;
    rclcpp::Subscription<vision_interface::msg::Detections>::SharedPtr detectionsSubscription;
    rclcpp::Subscription<vision_interface::msg::LineSegments>::SharedPtr subFieldLine;
    rclcpp::Subscription<booster_interface::msg::Odometer>::SharedPtr odometerSubscription;
    
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

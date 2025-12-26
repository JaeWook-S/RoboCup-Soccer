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

#include "brain_config.h"
#include "brain_data.h"
#include "brain_log.h"
#include "brain_tree.h"
#include "locator.h"
#include "robot_client.h"
#include "walk.h"
#include "movehead.h"
#include "chase.h"
#include "kick.h"
#include "adjust.h"
#include "decision_role.h"

// Forward declaration to avoid circular dependency
class BrainCommunication;

using namespace std;

/*
    해당 파일은 Brain 클래스의 헤더 파일입니다.
    ROS topic들을 처리하는 함수
    BT 행동 노드들을 등록하는 함수
    여러 행동 노드들의 공통 판단을 위한 함수만을 포함하고 있습니다.
*/

class Brain : public rclcpp::Node
{
public:
    // 클래스 객체 변수
    std::shared_ptr<RobotClient> client;
    std::shared_ptr<BrainTree> tree;
    std::shared_ptr<BrainConfig> config;
    std::shared_ptr<BrainData> data;
    std::shared_ptr<Locator> locator;
    std::shared_ptr<BrainLog> log;
    std::shared_ptr<BrainCommunication> communication;
    
    // 생성자, 소멸자
    Brain();
    ~Brain();

    void init();
    void tick();
    double msecsSince(rclcpp::Time time); // 특정 시간(timestamp) 이후 몇 밀리초가 지났는지 계산하는 유틸리티 함수

    void calibrateOdom(double x, double y, double theta);
    void updateFieldPos(GameObject &obj);
    void logDetection(const vector<GameObject> &gameObjects, bool logBoundingBox = true);
    bool isBoundingBoxInCenter(BoundingBox boundingBox, double xRatio = 0.5, double yRatio = 0.5);


    // 행동 노드들 등록
    void registerWalkNodes(BT::BehaviorTreeFactory &factory){RegisterWalkNodes(factory, this);}
    void registerMoveHeadNodes(BT::BehaviorTreeFactory &factory){RegisterMoveHeadNodes(factory, this);}
    void registerLocatorNodes(BT::BehaviorTreeFactory &factory){RegisterLocatorNodes(factory, this);}
    void registerChaseNodes(BT::BehaviorTreeFactory &factory){RegisterChaseNodes(factory, this);}
    void registerKickNodes(BT::BehaviorTreeFactory &factory){RegisterKickNodes(factory, this);}
    void registerAdjustNodes(BT::BehaviorTreeFactory &factory){RegisterAdjustNodes(factory, this);}
    
    // ROS callback 함수
    void gameControlCallback(const game_controller_interface::msg::GameControlData &msg);
    void detectionsCallback(const vision_interface::msg::Detections &msg);
    void fieldLineCallback(const vision_interface::msg::LineSegments &msg);
    void odometerCallback(const booster_interface::msg::Odometer &msg);
    void lowStateCallback(const booster_interface::msg::LowState &msg);
    void headPoseCallback(const geometry_msgs::msg::Pose &msg);
    void recoveryStateCallback(const booster_interface::msg::RawBytesMsg &msg);
    void depthImageCallback(const sensor_msgs::msg::Image &msg);
    void imageCallback(const sensor_msgs::msg::Image &msg);

    void logObstacles();
    void logDepth(int grid_x_count, int grid_y_count, vector<vector<int>> &grid_occupied, vector<rerun::Vec3D> &points_robot);

    /* ----------------------------- role 결정을 위한 함수 ----------------------------- */
    void registerDecisionRoleNodes(BT::BehaviorTreeFactory &factory){RegisterDecisionRoleNodes(factory, this);}
    bool isAngleGood(double goalPostMargin = 0.3, string type = "kick");
    



    /* ----------------------------- 변수 업데이트를 위한 함수들 ----------------------------- */
    void updateRelativePos(GameObject &obj);
    bool isFreekickStartPlacing();

    /*------------------------------- 공통으로 쓰이는 판단 로직 함수 ----------------------------------------------*/
    // ostacle 관련 함수
    /**
     * @brief 목표 각도 기준 충돌까지의 거리 계산
     *
     * @param angle double, 목표 방향 각도
     *
     * @return double, 충돌까지의 거리
     */
    double distToObstacle(double angle);
    vector<double> findSafeDirections(double startAngle, double safeDist, double step=deg2rad(10));
    double calcAvoidDir(double startAngle, double safeDist);
    
    // goalpost 관련 함수
    vector<double> getGoalPostAngles(const double margin = 0.3);

private:
    void loadConfig(); // config 불러오기

    /* ----------------------------- 변수 업데이트를 위한 함수들 ----------------------------- */
    void updateBallMemory();
    void updateObstacleMemory();

    // ROS subscription 변수
    rclcpp::Subscription<game_controller_interface::msg::GameControlData>::SharedPtr gameControlSubscription;
    rclcpp::Subscription<vision_interface::msg::Detections>::SharedPtr detectionsSubscription;
    rclcpp::Subscription<vision_interface::msg::LineSegments>::SharedPtr subFieldLine;
    rclcpp::Subscription<booster_interface::msg::Odometer>::SharedPtr odometerSubscription;
    rclcpp::Subscription<booster_interface::msg::LowState>::SharedPtr lowStateSubscription;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr headPoseSubscription;
    rclcpp::Subscription<booster_interface::msg::RawBytesMsg>::SharedPtr recoveryStateSubscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depthImageSubscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSubscription;
    
    // tf2 broadcaster
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

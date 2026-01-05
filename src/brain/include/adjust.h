#pragma once

#include <behaviortree_cpp/behavior_tree.h>
#include <behaviortree_cpp/bt_factory.h>

class Brain; 
using namespace BT;


void RegisterAdjustNodes(BT::BehaviorTreeFactory &factory, Brain* brain);

class Adjust : public SyncActionNode
{
public:
    Adjust(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("turn_threshold", 3.25, "공의 각도가 이 값보다 크면 먼저 회전하여 공을 정면으로 바라보고 이동은 중단한다"),
            InputPort<double>("vx_limit", 0.05, "Adjust 과정에서 전진/후진 속도 vx의 최대 제한값이다"),
            InputPort<double>("vy_limit", 0.05, "Adjust 과정에서 좌우 이동 속도 vy의 최대 제한값이다"),
            InputPort<double>("vtheta_limit", 0.1, "Adjust 과정에서 회전 속도 vtheta의 최대 제한값이다"),
            InputPort<double>("range", 2.25, "공과 로봇 사이의 목표 거리로 이 값을 유지하려고 한다"),
            InputPort<double>("vtheta_factor", 3.0, "각도 보정 시 vtheta에 곱해지는 계수로 클수록 회전이 빠르다"),
            InputPort<double>("tangential_speed_far", 0.2, "공이 멀 때 각도 보정을 위해 사용하는 접선 방향 이동 속도이다"),
            InputPort<double>("tangential_speed_near", 0.15, "공이 가까울 때 각도 보정을 위해 사용하는 접선 방향 이동 속도이다"),
            InputPort<double>("near_threshold", 0.8, "목표와의 거리가 이 값보다 작으면 near speed를 사용한다"),
            InputPort<double>("no_turn_threshold", 0.1, "각도 오차가 이 값보다 작으면 회전을 수행하지 않는다"),
            InputPort<double>("turn_first_threshold", 0.5, "각도 오차가 이 값보다 크면 이동하지 않고 회전만 먼저 수행한다"),
            InputPort<double>("yaw_offset", 0.13),
        };        
    }

    NodeStatus tick() override;

private:
    Brain *brain;
};

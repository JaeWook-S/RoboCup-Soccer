#pragma once

#include <behaviortree_cpp/behavior_tree.h>
#include <behaviortree_cpp/bt_factory.h>

class Brain; 
using namespace BT;

void RegisterDecisionRoleNodes(BT::BehaviorTreeFactory &factory, Brain* brain);

class StrikerDecide : public SyncActionNode
{
public:
    StrikerDecide(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain) {}

    static PortsList providedPorts()
    {
        return {
            InputPort<double>("chase_threshold", 1.0, "이 거리보다 멀어지면 공을 추격하는 동작을 수행"),
            InputPort<string>("decision_in", "", "이전 틱에서의 decision 값을 읽기 위해 사용"),
            InputPort<string>("position", "offense", "offense | defense, 공을 어느 방향으로 찰지 결정"),
            OutputPort<string>("decision_out")};
    }

    NodeStatus tick() override;

private:
    Brain *brain;
    double lastDeltaDir = 0.0; 
    rclcpp::Time timeLastTick; 
};

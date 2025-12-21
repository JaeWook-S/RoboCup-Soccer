#include "brain.h"
#include "chase.h"
#include "brain_tree.h"

#include <cstdlib>
#include <ctime>

// BehaviorTree Factory에 Test 노드를 생성하는 함수를 등록하는 역할 -> 코드 양 줄일 수 있음
#define REGISTER_CHASE_BUILDER(Name)     \
    factory.registerBuilder<Name>( \
        #Name,                     \
        [brain](const string &name, const NodeConfig &config) { return make_unique<Name>(name, config, brain); });


void RegisterChaseNodes(BT::BehaviorTreeFactory &factory, Brain* brain){
    REGISTER_CHASE_BUILDER(SimpleChase) // obstacle 없이 chase만 
    REGISTER_CHASE_BUILDER(Chase) // obstacle 추가된 chase
}

NodeStatus SimpleChase::tick(){
    double stopDist, stopAngle, vyLimit, vxLimit;
    getInput("stop_dist", stopDist); // 공과의 거리 임계값 -> 멈추기 위해
    getInput("stop_angle", stopAngle); // 공과의 각도 임계값 -> 멈추기 위해
    getInput("vx_limit", vxLimit); // x축 속도 제한
    getInput("vy_limit", vyLimit); // y축 속도 제한

    // 공의 위치를 모를 때 
    if (!brain->tree->getEntry<bool>("ball_location_known")){
        brain->client->setVelocity(0, 0, 0);
        return NodeStatus::SUCCESS;
    }

    // 로봇 기준 공과의 거리 
    // 단순 P 제어
    double vx = brain->data->ball.posToRobot.x; // 공과의 x축 거리
    double vy = brain->data->ball.posToRobot.y; // 공과의 y축 거리
    double vtheta = brain->data->ball.yawToRobot * 4.0; // 공과의 각도

    // 가까워질수록 속도가 줄어들도록
    double linearFactor = 1 / (1 + exp(3 * (brain->data->ball.range * fabs(brain->data->ball.yawToRobot)) - 3)); 
    vx *= linearFactor;
    vy *= linearFactor;

    // 속도 제한
    vx = cap(vx, vxLimit, -1.0);    
    vy = cap(vy, vyLimit, -vyLimit); 

    if (brain->data->ball.range < stopDist){
        vx = 0;
        vy = 0;
    }

    brain->client->setVelocity(vx, vy, vtheta, false, false, false);
    return NodeStatus::SUCCESS;
}

NodeStatus Chase::tick(){
    auto log = [=](string msg) {
        brain->log->setTimeNow();
        brain->log->log("debug/Chase4", rerun::TextLog(msg));
    };
    log("ticked");
    
    double vxLimit, vyLimit, vthetaLimit, dist, safeDist;
    getInput("vx_limit", vxLimit); // 속도 제한 
    getInput("vy_limit", vyLimit);
    getInput("vtheta_limit", vthetaLimit);
    getInput("dist", dist); // 공과의 거리
    getInput("safe_dist", safeDist); // 공과의 안전 거리

    bool avoidObstacle;
    brain->get_parameter("obstacle_avoidance.avoid_during_chase", avoidObstacle); // 장애물 회피
    double oaSafeDist;
    brain->get_parameter("obstacle_avoidance.chase_ao_safe_dist", oaSafeDist); 	// 회피 시 최소 안전 거리 oaSafeDist

    // 속도 clapping
    if ( brain->config->limitNearBallSpeed && brain->data->ball.range < brain->config->nearBallRange){
        vxLimit = min(brain->config->nearBallSpeedLimit, vxLimit);
    }

    double ballRange = brain->data->ball.range;
    double ballYaw = brain->data->ball.yawToRobot;
    double kickDir = brain->data->kickDir;
    // ball → robot 각도 (field)
    double theta_br = atan2(
        brain->data->robotPoseToField.y - brain->data->ball.posToField.y,
        brain->data->robotPoseToField.x - brain->data->ball.posToField.x
    );
    double theta_rb = brain->data->robotBallAngleToField; // robot → ball 각도
    auto ballPos = brain->data->ball.posToField;


    double vx, vy, vtheta;
    Pose2D target_f, target_r; 
    static string targetType = "direct"; 
    static double circleBackDir = 1.0; 
    double dirThreshold = M_PI / 2;
    if (targetType == "direct") dirThreshold *= 1.2;


    // 로봇 → 공 방향이 킥 방향과 충분히 유사
    if (fabs(toPInPI(kickDir - theta_rb)) < dirThreshold) {
        log("targetType = direct");
        targetType = "direct";
        target_f.x = ballPos.x - dist * cos(kickDir);
        target_f.y = ballPos.y - dist * sin(kickDir);
    } 
    // 로봇 → 공 방향이 킥 방향과 충분히 유사하지 않을 때
    // 공을 둘러싸는 원을 그리는 방식 	• 공 주위를 원형으로 돌아서 접근
    else {
        targetType = "circle_back";
        double cbDirThreshold = 0.0; 
        cbDirThreshold -= 0.2 * circleBackDir; 
        circleBackDir = toPInPI(theta_br - kickDir) > cbDirThreshold ? 1.0 : -1.0;
        log(format("targetType = circle_back, circleBackDir = %.1f", circleBackDir));
        double tanTheta = theta_br + circleBackDir * acos(min(1.0, safeDist/max(ballRange, 1e-5))); 
        target_f.x = ballPos.x + safeDist * cos(tanTheta);
        target_f.y = ballPos.y + safeDist * sin(tanTheta);
    }

    target_r = brain->data->field2robot(target_f);
    brain->log->setTimeNow();
    brain->log->logBall("field/chase_target", Point({target_f.x, target_f.y, 0}), 0xFFFFFFFF, false, false);
            
    double targetDir = atan2(target_r.y, target_r.x);
    double distToObstacle = brain->distToObstacle(targetDir);
    // 장애물을 회피할 때
    if (avoidObstacle && distToObstacle < oaSafeDist) {
        log("avoid obstacle");
        auto avoidDir = brain->calcAvoidDir(targetDir, oaSafeDist);
        const double speed = 0.5;
        vx = speed * cos(avoidDir);
        vy = speed * sin(avoidDir);
        vtheta = ballYaw;
    } 
    else {
        vx = min(vxLimit, brain->data->ball.range);
        vy = 0;
        vtheta = targetDir;
        if (fabs(targetDir) < 0.1 && ballRange > 2.0) vtheta = 0.0;
        vx *= sigmoid((fabs(vtheta)), 1, 3); 
    }

    // 승재욱 추가 -> 공과의 위치가 5도 이하면 그대로 멈춤
    if(fabs(vtheta) < 0.09){
        vtheta = 0;
    }

    vx = cap(vx, vxLimit, -vxLimit);
    vy = cap(vy, vyLimit, -vyLimit);
    vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);

    static double smoothVx = 0.0;
    static double smoothVy = 0.0;
    static double smoothVtheta = 0.0;
    smoothVx = smoothVx * 0.7 + vx * 0.3;
    smoothVy = smoothVy * 0.7 + vy * 0.3;
    smoothVtheta = smoothVtheta * 0.7 + vtheta * 0.3;

    // brain->client->setVelocity(smoothVx, smoothVy, smoothVtheta, false, false, false);
    brain->client->setVelocity(vx, vy, vtheta, false, false, false);
    return NodeStatus::SUCCESS;
}
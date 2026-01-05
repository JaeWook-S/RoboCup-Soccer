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

// 원본 Chase
NodeStatus Chase::tick(){
    auto log = [=](string msg) {
        brain->log->setTimeNow();
        brain->log->log("debug/Chase4", rerun::TextLog(msg));
    };
    log("ticked");

    if (brain->tree->getEntry<string>("striker_state") != "chase") return NodeStatus::SUCCESS;
    
    double vxLimit, vyLimit, vthetaLimit, dist, safeDist;
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    getInput("vtheta_limit", vthetaLimit);
    getInput("dist", dist);
    getInput("safe_dist", safeDist);

    bool avoidObstacle;
    brain->get_parameter("obstacle_avoidance.avoid_during_chase", avoidObstacle);
    double oaSafeDist;
    brain->get_parameter("obstacle_avoidance.chase_ao_safe_dist", oaSafeDist);

    if (
        brain->config->limitNearBallSpeed
        && brain->data->ball.range < brain->config->nearBallRange
    ) {
        vxLimit = min(brain->config->nearBallSpeedLimit, vxLimit);
    }

    double ballRange = brain->data->ball.range;
    double ballYaw = brain->data->ball.yawToRobot;
    double kickDir = brain->data->kickDir;

    double theta_br = atan2(
        brain->data->robotPoseToField.y - brain->data->ball.posToField.y,
        brain->data->robotPoseToField.x - brain->data->ball.posToField.x
    );
    double theta_rb = brain->data->robotBallAngleToField;
    auto ballPos = brain->data->ball.posToField;


    double vx, vy, vtheta;
    Pose2D target_f, target_r; 
    static string targetType = "direct"; 
    static double circleBackDir = 1.0; 
    double dirThreshold = M_PI / 2;
    if (targetType == "direct") dirThreshold *= 1.2;


    // calculate target point
    if (fabs(toPInPI(kickDir - theta_rb)) < dirThreshold) {
        log("targetType = direct");
        targetType = "direct";
        target_f.x = ballPos.x - dist * cos(kickDir);
        target_f.y = ballPos.y - dist * sin(kickDir);
    } 
    // else {
    //     targetType = "circle_back";
    //     double cbDirThreshold = 0.0; 
    //     cbDirThreshold -= 0.2 * circleBackDir; 
    //     circleBackDir = toPInPI(theta_br - kickDir) > cbDirThreshold ? 1.0 : -1.0;
    //     log(format("targetType = circle_back, circleBackDir = %.1f", circleBackDir));
    //     double tanTheta = theta_br + circleBackDir * acos(min(1.0, safeDist/max(ballRange, 1e-5))); 
    //     target_f.x = ballPos.x + safeDist * cos(tanTheta);
    //     target_f.y = ballPos.y + safeDist * sin(tanTheta);
    // }
    target_r = brain->data->field2robot(target_f);
    brain->log->setTimeNow();
    brain->log->logBall("field/chase_target", Point({target_f.x, target_f.y, 0}), 0xFFFFFFFF, false, false);
            
    double targetDir = atan2(target_r.y, target_r.x);
    double distToObstacle = brain->distToObstacle(targetDir);
    
    if (avoidObstacle && distToObstacle < oaSafeDist) {
        log("avoid obstacle");
        auto avoidDir = brain->calcAvoidDir(targetDir, oaSafeDist);
        const double speed = 0.5;
        vx = speed * cos(avoidDir);
        vy = speed * sin(avoidDir);
        vtheta = ballYaw;
        log(format("avoidDir = %.2f", avoidDir));
    } 

    else {
        vx = min(vxLimit, brain->data->ball.range);
        vy = 0;
        vtheta = targetDir;
        if (fabs(targetDir) < 0.1 && ballRange > 2.0) vtheta = 0.0;
        vx *= sigmoid((fabs(vtheta)), 1, 3); 
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

    // chase 멈춤 조건
    bool chaseDone = brain->data->ball.range < dist * 1.2 && fabs(toPInPI(kickDir - theta_rb)) < M_PI / 3;
    if (chaseDone){
        brain->tree->setEntry("striker_state", "adjust");
        log("chase -> adjust");
    }
    log(format("distToObstacle = %.2f, targetDir = %.2f", distToObstacle, targetDir));
    
    // brain->client->setVelocity(smoothVx, smoothVy, smoothVtheta, false, false, false);
    brain->client->setVelocity(vx, vy, vtheta, false, false, false);
    return NodeStatus::SUCCESS;
}


// // 승재욱 - 직접 만든 Chase
// NodeStatus Chase::tick(){
//     auto log = [=](string msg) {
//         brain->log->setTimeNow();
//         brain->log->log("debug/Chase4", rerun::TextLog(msg));
//     };
//     log("ticked");
    
//     // [입력 변수 가져오기]
//     double vxLimit, vyLimit, vthetaLimit, dist, safeDist;
//     getInput("vx_limit", vxLimit); 
//     getInput("vy_limit", vyLimit);
//     getInput("vtheta_limit", vthetaLimit);
//     getInput("dist", dist);          // 목표: 공 뒤쪽 거리 (예: 30cm)
//     getInput("safe_dist", safeDist); // 회전 접근 시 안전 거리

//     // 승재욱 추가 -> 킥 준비 false로 초기화
//     brain->tree->setEntry("ready_to_kick", false);
    
//     // [장애물 회피 파라미터]
//     bool avoidObstacle;
//     brain->get_parameter("obstacle_avoidance.avoid_during_chase", avoidObstacle); 
//     double oaSafeDist;
//     brain->get_parameter("obstacle_avoidance.chase_ao_safe_dist", oaSafeDist); 

//     // [속도 제한 (공 근처 감속)]
//     if ( brain->config->limitNearBallSpeed && brain->data->ball.range < brain->config->nearBallRange){
//         vxLimit = min(brain->config->nearBallSpeedLimit, vxLimit);
//     }

//     // [데이터 가져오기]
//     double ballRange = brain->data->ball.range;
//     double ballYaw = brain->data->ball.yawToRobot;
//     auto ballPos = brain->data->ball.posToField;
    
//     // *중요: kickDir은 CalcKickDir에서 절대좌표로 계산된 값을 쓰는 것이 가장 좋습니다.
//     double kickDir = brain->data->kickDir; 
    
//     double theta_rb = brain->data->robotBallAngleToField; // 로봇 -> 공 각도
//     double theta_br = atan2( // 공 -> 로봇 각도
//         brain->data->robotPoseToField.y - ballPos.y,
//         brain->data->robotPoseToField.x - ballPos.x
//     );

//     // =========================================================================
//     // [1. 상태 관리 & 히스테리시스 (State Machine & Hysteresis)]
//     // =========================================================================
    
//     // 상태를 기억하기 위한 static 변수들
//     static string targetType = "circle_back"; // 초기값
//     static double lockedCircleDir = 1.0;      // 회전 방향 기억 (1.0 or -1.0)
//     static double lockedKickDir = 0.0;        // Direct 모드용 킥 방향 기억
//     static bool isTargetLocked = false;       // 타겟 락 여부

//     // 기준 임계값
//     double baseThreshold = M_PI / 2; // 90도
    
//     // 히스테리시스 임계값 설정 (들어갈 땐 빡빡하게, 나올 땐 널널하게)
//     double enterDirectThresh = baseThreshold;       // 예: 90도 이내면 Direct 진입
//     double exitDirectThresh  = baseThreshold + 0.3; // 예: 108도 벗어나야 CircleBack 복귀 (여유폭 0.3 rad)

//     double angleDiff = fabs(toPInPI(kickDir - theta_rb));

//     // [상태 전환 로직]
//     if (targetType == "direct") {
//         log("targetType = direct");
//         // Direct 상태 유지 중: 오차가 exitThreshold를 넘어야만 상태 변경
//         if (angleDiff > exitDirectThresh) {
//             targetType = "circle_back";
//             isTargetLocked = false; // 락 해제
//             log("State Changed: Direct -> CircleBack");
            
//             // CircleBack 진입 시 회전 방향 결정 (여기서 딱 한 번만 정함!)
//             double cbDirThreshold = -0.2 * lockedCircleDir; 
//             lockedCircleDir = toPInPI(theta_br - kickDir) > cbDirThreshold ? 1.0 : -1.0;
//         }
//     } 
//     else { // targetType == "circle_back"
//         // CircleBack 상태 유지 중: 오차가 enterThreshold보다 작아지면 상태 변경
//         if (angleDiff < enterDirectThresh) {
//             targetType = "direct";
//             log("State Changed: CircleBack -> Direct");
            
//             // Direct 진입 시 현재 킥 방향을 기억(Lock)함
//             lockedKickDir = kickDir; 
//             isTargetLocked = true;
//         }
//     }

//     // =========================================================================
//     // [2. 목표 지점(Target) 계산]
//     // =========================================================================
//     Pose2D target_f, target_r;

//     if (targetType == "direct") {
//         // [Direct 모드]
//         // 킥 방향이 미세하게 흔들려도 무시하고, 기억해둔 'lockedKickDir'를 사용
//         // 단, 공이 굴러갈 수 있으니 'ballPos'는 계속 최신값 반영
//         if (!isTargetLocked) { 
//             lockedKickDir = kickDir; 
//             isTargetLocked = true; 
//         }

//         target_f.x = ballPos.x - dist * cos(lockedKickDir);
//         target_f.y = ballPos.y - dist * sin(lockedKickDir);
//     } 
//     else {
//         // [Circle Back 모드]
//         // 회전 방향(lockedCircleDir)을 사용하므로 좌우로 튀지 않음
//         double tanTheta = theta_br + lockedCircleDir * acos(min(1.0, safeDist/max(ballRange, 1e-5))); 
//         target_f.x = ballPos.x + safeDist * cos(tanTheta);
//         target_f.y = ballPos.y + safeDist * sin(tanTheta);
//     }

//     // 로봇 기준 좌표로 변환
//     target_r = brain->data->field2robot(target_f);
    
//     // 디버그 로그 (Rerun)
//     brain->log->setTimeNow();
//     brain->log->logBall("field/chase_target", Point({target_f.x, target_f.y, 0}), 0xFFFFFFFF, false, false);

//     // =========================================================================
//     // [3. 이동 명령 생성 (Movement Generation)]
//     // =========================================================================
    
//     double vx, vy, vtheta;
//     double targetDir = atan2(target_r.y, target_r.x);
//     double distToTarget = norm(target_r.x, target_r.y); // 목표까지 남은 거리
//     double distToObstacle = brain->distToObstacle(targetDir);

//     // (A) 장애물 회피
//     if (avoidObstacle && distToObstacle < oaSafeDist) {
//         log("Avoiding Obstacle");
//         auto avoidDir = brain->calcAvoidDir(targetDir, oaSafeDist);
//         const double speed = 0.5;
//         vx = speed * cos(avoidDir);
//         vy = speed * sin(avoidDir);
//         vtheta = ballYaw;
//     } 
//     // (B) 일반 주행
//     else {
//         double stopTolerance = 0.1; // 10cm 이내 도착 판정
        
//         // [도착 처리] 목표 지점에 거의 다 왔으면
//         if (distToTarget < stopTolerance) {
//             vx = 0.0;
//             vy = 0.0;
            
//             // 도착했으면 이동 방향(targetDir)이 아니라, 공 방향(ballYaw)을 바라봄
//             // 이미 lockedKickDir로 정렬해서 들어왔으므로 ballYaw를 보면 킥 각이 나옴
//             vtheta = ballYaw; 
//             // 각도 오차가 5도(약 0.09 rad) 이내면 회전 모터를 끔 -> 제자리 떨림 방지 핵심
//             if (fabs(vtheta) < 0.2) {
//                 vtheta = 0.0;
//                 // 승재욱 추가 -> 킥 준비
//                 brain->tree->setEntry("ready_to_kick", true);
//             }
//         }
//         // [이동 처리] 아직 가는 중이면
//         else {
//             // 거리에 비례하여 속도 조절 (도착지점에서 부드럽게 감속)
//             vx = min(vxLimit, distToTarget); 
//             vy = 0;
//             vtheta = targetDir;

//             // 먼 거리 직진 보정
//             if (fabs(targetDir) < 0.1 && ballRange > 2.0) vtheta = 0.0;
            
//             // 회전 시 전진 감속 (커브 돌 때 속도 줄임)
//             vx *= sigmoid((fabs(vtheta)), 1, 3); 
//         }
//     }

//     // =========================================================================
//     // [4. 안정화: 데드존 (Deadzone)]
//     // =========================================================================
    
//     // // 각도 오차가 5도(약 0.09 rad) 이내면 회전 모터를 끔 -> 제자리 떨림 방지 핵심
//     // if (fabs(vtheta) < 0.2) {
//     //     vtheta = 0.0;
//     // }

//     // [5. 리미트 적용]
//     vx = cap(vx, vxLimit, -vxLimit);
//     vy = cap(vy, vyLimit, -vyLimit);
//     vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);

//     // [6. 명령 전송 (스무딩 없이 즉시 반응)]
//     brain->client->setVelocity(vx, vy, vtheta, false, false, false);
    
//     return NodeStatus::SUCCESS;
// }

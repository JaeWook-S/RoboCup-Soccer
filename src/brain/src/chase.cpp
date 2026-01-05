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

// // 원본 Chase
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


// Chase.cpp (or wherever your BT node is implemented)
// 목표: target_f 계산 로직은 그대로 유지하면서,
//      1) robotPoseToField.theta 가 kickDir 와 정렬되도록 vtheta를 P제어
//      2) target_r.y(측면 오차)를 이용해 vy도 사용
//      3) 헤딩 정렬이 안 되면(vtheta 오차가 크면) vx/vy를 자동으로 줄여 안정성 확보

// NodeStatus Chase::tick() {
//     // -----------------------------
//     // Logging helper
//     // -----------------------------
//     auto log = [=](string msg) {
//         brain->log->setTimeNow();
//         brain->log->log("debug/Chase4", rerun::TextLog(msg));
//     };
//     log("ticked");

//     // -----------------------------
//     // State guard: striker_state != chase 면 바로 성공 반환(아무것도 안 함)
//     // -----------------------------
//     if (brain->tree->getEntry<string>("striker_state") != "chase") {
//         return NodeStatus::SUCCESS;
//     }

//     // -----------------------------
//     // BT input ports
//     // -----------------------------
//     double vxLimit, vyLimit, vthetaLimit, dist, safeDist;
//     getInput("vx_limit", vxLimit);           // 최대 전진 속도 제한
//     getInput("vy_limit", vyLimit);           // 최대 측면 속도 제한
//     getInput("vtheta_limit", vthetaLimit);   // 최대 회전 속도 제한
//     getInput("dist", dist);                  // 공 뒤에서 유지하고 싶은 거리(=target_f 오프셋)
//     getInput("safe_dist", safeDist);         // (circle_back에서 쓰던) 안전거리

//     // -----------------------------
//     // Obstacle avoidance params
//     // -----------------------------
//     bool avoidObstacle = false;
//     brain->get_parameter("obstacle_avoidance.avoid_during_chase", avoidObstacle);

//     double oaSafeDist = 0.0;
//     brain->get_parameter("obstacle_avoidance.chase_ao_safe_dist", oaSafeDist);

//     // -----------------------------
//     // Near-ball speed limiting (기존 유지)
//     // -----------------------------
//     if (brain->config->limitNearBallSpeed &&
//         brain->data->ball.range < brain->config->nearBallRange) {
//         vxLimit = min(brain->config->nearBallSpeedLimit, vxLimit);
//     }

//     // -----------------------------
//     // Read perception / strategy data
//     // -----------------------------
//     const double ballRange = brain->data->ball.range;
//     const double ballYaw   = brain->data->ball.yawToRobot;          // 로봇 기준 공의 yaw
//     const double kickDir   = brain->data->kickDir;                  // 필드 기준 "킥 방향"
//     const double theta_rb  = brain->data->robotBallAngleToField;    // 필드 기준 "로봇->공" 방향(혹은 공 기준?)
//     const auto   ballPos   = brain->data->ball.posToField;          // 필드 기준 공 위치

//     // 로봇-공 관계 각도(필드 기준): ball -> robot 방향
//     const double theta_br = atan2(
//         brain->data->robotPoseToField.y - brain->data->ball.posToField.y,
//         brain->data->robotPoseToField.x - brain->data->ball.posToField.x
//     );

//     // -----------------------------
//     // Output velocities to send
//     // -----------------------------
//     double vx, vy, vtheta;

//     // -----------------------------
//     // Target point calculation ( 여기까지는 "그대로 유지" 요청 반영)
//     // -----------------------------
//     Pose2D target_f, target_r;

//     static string targetType = "direct";
//     static double circleBackDir = 1.0;  // (현재 circle_back 주석이라 실사용 X)
//     double dirThreshold = M_PI / 2;
//     if (targetType == "direct") dirThreshold *= 1.2;

//     // 기존 로직: kickDir 과 theta_rb 차이가 작으면 "direct"로 공 뒤 타겟을 잡는다
//     if (fabs(toPInPI(kickDir - theta_rb)) < dirThreshold) {
//         log("targetType = direct");
//         targetType = "direct";

//         // 공 위치에서 kickDir 반대 방향으로 dist만큼 떨어진 점 = 공 뒤쪽
//         target_f.x = ballPos.x - dist * cos(kickDir);
//         target_f.y = ballPos.y - dist * sin(kickDir);
//     }
//     // circle_back 부분은 그대로 주석 유지 (원하면 나중에 다시 살릴 수 있음)
//     /*
//     else {
//         targetType = "circle_back";
//         double cbDirThreshold = 0.0;
//         cbDirThreshold -= 0.2 * circleBackDir;
//         circleBackDir = toPInPI(theta_br - kickDir) > cbDirThreshold ? 1.0 : -1.0;
//         log(format("targetType = circle_back, circleBackDir = %.1f", circleBackDir));
//         double tanTheta = theta_br + circleBackDir * acos(min(1.0, safeDist/max(ballRange, 1e-5)));
//         target_f.x = ballPos.x + safeDist * cos(tanTheta);
//         target_f.y = ballPos.y + safeDist * sin(tanTheta);
//     }
//     */

//     // 필드 -> 로봇 좌표로 타겟 변환
//     target_r = brain->data->field2robot(target_f);

//     // Rerun debug: chase target 표시(필드 좌표)
//     brain->log->setTimeNow();
//     brain->log->logBall(
//         "field/chase_target",
//         Point({target_f.x, target_f.y, 0}),
//         0xFFFFFFFF,
//         false,
//         false
//     );

//     // -----------------------------
//     // Useful derived values
//     // -----------------------------
//     // 로봇 좌표계에서 타겟이 있는 방향(로봇 기준 각도)
//     const double targetDir = atan2(target_r.y, target_r.x);

//     // 장애물까지 거리(로봇 기준 targetDir 방향 레이캐스팅 같은 것)
//     const double distToObstacle = brain->distToObstacle(targetDir);

//     // -----------------------------
//     // (A) Obstacle avoidance 우선 처리
//     // -----------------------------
//     if (avoidObstacle && distToObstacle < oaSafeDist) {
//         log("avoid obstacle");

//         // 회피 방향 계산 (로봇 기준 각도 반환 가정)
//         const double avoidDir = brain->calcAvoidDir(targetDir, oaSafeDist);

//         // 회피할 땐 간단히 일정 속도로 피한다(기존 유지)
//         const double speed = 0.5;
//         vx     = speed * cos(avoidDir);
//         vy     = speed * sin(avoidDir);

//         // 회피 중에도 공을 정면으로 두려는 보정 (기존 코드에서는 vtheta=ballYaw)
//         // ballYaw는 로봇 기준 공 방향이므로, vtheta에 그대로 넣으면 "공 바라보기" 회전이 됨
//         vtheta = ballYaw;

//         log(format("avoidDir = %.2f", avoidDir));
//     }
//     // -----------------------------
//     // (B) Normal chase: ✅ 여기서 P제어 + vy 사용
//     // -----------------------------
//     else {
//         // =============================
//         // 1) 헤딩(로봇 θ) 을 kickDir 로 정렬시키는 P 제어
//         // =============================
//         const double robotTheta_f = brain->data->robotPoseToField.theta; // 필드 기준 로봇 헤딩
//         const double thetaErr     = toPInPI(kickDir - robotTheta_f);     // "원하는 헤딩 - 현재 헤딩"

//         // =============================
//         // 2) 타겟까지 로봇좌표 오차(target_r)로 vx/vy 제어
//         //    target_r.x: 전방 오차(+면 앞으로 가야 함)
//         //    target_r.y: 측방 오차(+면 왼쪽/오른쪽은 좌표 정의에 따라 다름)
//         // =============================

//         // ---- P gains (튜닝 포인트) ----
//         // 휴머노이드에서 너무 큰 게인은 흔들림/넘어짐 유발 가능 -> 보수적으로 시작
//         const double Kp_x     = 0.8;  // 전진
//         const double Kp_y     = 1.0;  // 측면
//         const double Kp_theta = 1.5;  // 헤딩 정렬

//         // P 제어 입력(원시값)
//         vx     = Kp_x * target_r.x;
//         vy     = Kp_y * target_r.y;
//         vtheta = Kp_theta * thetaErr;

//         // =============================
//         // 3) 정렬이 안 됐으면(각도 오차 크면) 이동을 줄여 안정성 확보
//         //    - "몸이 돌아가는 중에 크게 전진/측면 이동"을 줄여서 발/상체 안정
//         // =============================
//         // sigmoid(|thetaErr|)가 0~1 스케일을 준다고 가정(기존 코드와 동일 사용)
//         //  - |thetaErr|가 크면 scale이 작아져 vx/vy가 줄어듦
//         const double headingScale = sigmoid(fabs(thetaErr), 1.0, 3.0);
//         vx *= headingScale;
//         vy *= headingScale;

//         // =============================
//         // 4) 추가 안정화 규칙 (선택)
//         //    멀리 있고 각도 오차도 작으면 vtheta 굳이 주지 않기(흔들림 감소)
//         // =============================
//         if (ballRange > 2.0 && fabs(thetaErr) < 0.1) {
//             vtheta = 0.0;
//         }

//         // (선택) 너무 가까울 때 vx를 줄여서 과속 접근 방지
//         // if (ballRange < 0.6) vx *= 0.7;

//         // 디버깅 로그
//         log(format("thetaErr=%.3f rad, target_r=(%.2f, %.2f), headingScale=%.2f",
//                    thetaErr, target_r.x, target_r.y, headingScale));
//     }

//     // -----------------------------
//     // Apply limits (saturation)
//     // -----------------------------
//     vx     = cap(vx,     vxLimit,     -vxLimit);
//     vy     = cap(vy,     vyLimit,     -vyLimit);
//     vtheta = cap(vtheta, vthetaLimit, -vthetaLimit);

//     // -----------------------------
//     // (Optional) smoothing
//     // - 기존 코드에 smoothing 변수가 있었지만 실제 setVelocity에는 raw vx/vy/vtheta를 쓰고 있었음
//     // - 필요하면 아래 smooth 값을 보내도록 변경 가능
//     // -----------------------------
//     static double smoothVx = 0.0;
//     static double smoothVy = 0.0;
//     static double smoothVtheta = 0.0;
//     smoothVx     = smoothVx * 0.7 + vx     * 0.3;
//     smoothVy     = smoothVy * 0.7 + vy     * 0.3;
//     smoothVtheta = smoothVtheta * 0.7 + vtheta * 0.3;

//     // -----------------------------
//     // Chase 종료 조건
//     // 기존: ball.range < dist*1.2 AND |kickDir - theta_rb| < 60deg
//     // 개선: 이제 "로봇 헤딩(robotTheta)과 kickDir 정렬"이 핵심이므로 thetaErr 기준이 더 직접적
//     // -----------------------------
//     {
//         const double robotTheta_f = brain->data->robotPoseToField.theta;
//         const double thetaErr     = toPInPI(kickDir - robotTheta_f);

//         // dist*1.2: 타겟 거리(공 뒤 dist) 근처로 도달했는지
//         const bool closeEnough = (ballRange < dist * 1.2);

//         // kickDir 정렬 정도 (0.2 rad ≈ 11.5도)
//         const bool headingAligned = (fabs(thetaErr) < 0.2);

//         // 필요하면 기존 조건(theta_rb)도 함께 AND/OR로 섞을 수 있음
//         // const bool rbAligned = fabs(toPInPI(kickDir - theta_rb)) < M_PI/3;

//         const bool chaseDone = closeEnough && headingAligned;

//         if (chaseDone) {
//             brain->tree->setEntry("striker_state", "adjust");
//             log("chase -> adjust");
//         }

//         log(format("distToObstacle=%.2f, targetDir=%.2f, closeEnough=%d, headingAligned=%d",
//                    distToObstacle, targetDir, (int)closeEnough, (int)headingAligned));
//     }

//     // -----------------------------
//     // Send velocity
//     // 기본은 raw 출력. 흔들림 있으면 smoothing 사용 권장.
//     // -----------------------------
//     // brain->client->setVelocity(smoothVx, smoothVy, smoothVtheta, false, false, false);
//     brain->client->setVelocity(vx, vy, vtheta, false, false, false);

//     return NodeStatus::SUCCESS;
// }
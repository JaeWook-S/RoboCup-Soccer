#include "brain.h"
#include "brain_tree.h"
#include "kick.h"

// BehaviorTree Factory에 Test 노드를 생성하는 함수를 등록하는 역할 -> 코드 양 줄일 수 있음
#define REGISTER_KICK_BUILDER(Name)     \
    factory.registerBuilder<Name>( \
        #Name,                     \
        [brain](const string &name, const NodeConfig &config) { return make_unique<Name>(name, config, brain); });


void RegisterKickNodes(BT::BehaviorTreeFactory &factory, Brain* brain){
    REGISTER_KICK_BUILDER(CalcKickDir) // obstacle 없이 chase만 
}

// 해당 노드는 반코트용으로 짰음 -> 풀코트 용으로 나중에 수정 필요함 
// 제안 : 반코트, 풀코트 구분하도록 HalfCalcKickDir, FullCalcKickDir로 구분할 수도 있으니 알아서 잘 활용하길 바람
// 추가해야 될 것 -> 수비 상황 판단
NodeStatus CalcKickDir::tick(){
    // 좌·우 골포스트 각도 차가 이 값보다 작으면 → 골대가 “좁게 보인다” = 정면 슛 각이 안 나온다
    double crossThreshold;
    getInput("cross_threshold", crossThreshold);

    string lastKickType = brain->data->kickType;
    if (lastKickType == "cross") crossThreshold += 0.1; // 이전 type이 cross였다면 좀 더 threshold를 높임으로써 cross를 유지할 수 있도록 함

    auto gpAngles = brain->getGoalPostAngles(0.0); // 공과 골대 각도 계산 -> 마진이 0으로 돼있음
    auto thetal = gpAngles[0]; auto thetar = gpAngles[1]; // 공 기준 좌우 골대 각도
    auto bPos = brain->data->ball.posToField; // 공 위치
    auto fd = brain->config->fieldDimensions; // 필드 정보
    auto color = 0xFFFFFFFF;

    // 풀코트용 슛 방향 계산
    // if (thetal - thetar < crossThreshold && brain->data->ball.posToField.x > fd.circleRadius) {
    //     brain->data->kickType = "cross";
    //     color = 0xFF00FFFF;
    //     brain->data->kickDir = atan2(
    //         - bPos.y,
    //         fd.length/2 - fd.penaltyDist/2 - bPos.x
    //     );
    // }
    // else if (brain->isDefensing()) {
    //     brain->data->kickType = "block";
    //     color = 0xFFFF00FF;
    //     brain->data->kickDir = atan2(
    //         bPos.y,
    //         bPos.x + fd.length/2
    //     );
    // } else { 
    //     brain->data->kickType = "shoot";
    //     color = 0x00FF00FF;
    //     brain->data->kickDir = atan2(
    //         - bPos.y,
    //         fd.length/2 - bPos.x
    //     );
    //     if (brain->data->ball.posToField.x > brain->config->fieldDimensions.length / 2) brain->data->kickDir = 0; 
    // }

    // 반코트용 슛 방향 계산 ( 상대 골대가 음수라고 가정)
    // 추가해야 될 것 -> 수비 상황 판단
    // 그리고 항상 왼쪽이 양수일까?도 고민해보기
    if (thetal - thetar < crossThreshold) {
        brain->data->kickType = "cross";
        color = 0xFF00FFFF;
        // atan2( 목표Y - 공Y , 목표X - 공X )
        // 반코트는 우리팀 진영을 상대팀으로 인식해야하므로 - 값이 들어감
        brain->data->kickDir = atan2( 
            0 - bPos.y, 
            - (fd.length/2 - fd.penaltyDist/2) - bPos.x); // 앞에 - 붙였음
    }
    else { 
        brain->data->kickType = "shoot";
        color = 0x00FF00FF;
        brain->data->kickDir = atan2(
            0 - bPos.y,
            - fd.length/2 - bPos.x // 여기도 - 붙였음
        );
        // 이것도 - 로 바꿈 + 0에서 M_PI로 바꿈
        if (brain->data->ball.posToField.x < - (brain->config->fieldDimensions.length / 2)) brain->data->kickDir = M_PI; 
    }

    brain->log->setTimeNow();
    brain->log->log(
        "field/kick_dir",
        rerun::Arrows2D::from_vectors({{10 * cos(brain->data->kickDir), -10 * sin(brain->data->kickDir)}})
            .with_origins({{brain->data->ball.posToField.x, -brain->data->ball.posToField.y}})
            .with_colors({color})
            .with_radii(0.01)
            .with_draw_order(31)
    );

    return NodeStatus::SUCCESS;
}

// 승재욱 - 직접 만든 Kick
tuple<double, double, double> Kick::_calcSpeed() {
    double vx, vy, msecKick;


    double vxLimit, vyLimit;
    getInput("vx_limit", vxLimit);
    getInput("vy_limit", vyLimit);
    int minMSecKick;
    getInput("min_msec_kick", minMSecKick);
    double vxFactor = brain->config->vxFactor;   
    double yawOffset = brain->config->yawOffset; 


    double adjustedYaw = brain->data->ball.yawToRobot + yawOffset;
    double tx = cos(adjustedYaw) * brain->data->ball.range; 
    double ty = sin(adjustedYaw) * brain->data->ball.range;

    if (fabs(ty) < 0.01 && fabs(adjustedYaw) < 0.01){ 
        vx = vxLimit;
        vy = 0.0;
    }
    else{ 
        vy = ty > 0 ? vyLimit : -vyLimit;
        vx = vy / ty * tx * vxFactor;
        if (fabs(vx) > vxLimit){
            vy *= vxLimit / vx;
            vx = vxLimit;
        }
    }

    double speed = norm(vx, vy);
    msecKick = speed > 1e-5 ? minMSecKick + static_cast<int>(brain->data->ball.range / speed * 1000) : minMSecKick;
    
    return make_tuple(vx, vy, msecKick);
}

NodeStatus Kick::onStart()
{
    // =========================================================================
    // [추가됨] 1. 블랙보드 신호 확인 (안전장치)
    // =========================================================================
    // Chase 노드에서 true로 바꿔주지 않았다면, 킥을 실행하지 않고 실패 처리
    if (!brain->tree->getEntry<bool>("ready_to_kick")) {
        // 아직 준비 안 됨 -> FAILURE 리턴하여 다시 Chase로 돌아가게 함
        return NodeStatus::SUCCESS; 
    }
    // =========================================================================

    _minRange = brain->data->ball.range;
    _speed = 0.5;
    _startTime = brain->get_clock()->now();

    // 장애물 회피 로직
    bool avoidPushing;
    double kickAoSafeDist;
    brain->get_parameter("obstacle_avoidance.avoid_during_kick", avoidPushing);
    brain->get_parameter("obstacle_avoidance.kick_ao_safe_dist", kickAoSafeDist);
    // string role = brain->tree->getEntry<string>("player_role");
    
    if (
        avoidPushing
        // && (role != "goal_keeper")
        && brain->data->robotPoseToField.x < brain->config->fieldDimensions.length / 2 - brain->config->fieldDimensions.goalAreaLength
        && brain->distToObstacle(brain->data->ball.yawToRobot) < kickAoSafeDist
    ) {
        brain->client->setVelocity(-0.1, 0, 0);
        
        // [추가됨] 킥 시도했으나 장애물 때문에 중단했으므로 플래그 초기화
        brain->tree->setEntry("ready_to_kick", false);
        
        return NodeStatus::SUCCESS;
    }

    // [원본 그대로] 운동 지령 (게걸음 시작)
    double angle = brain->data->ball.yawToRobot;
    brain->client->crabWalk(angle, _speed);
    
    return NodeStatus::RUNNING;
}

NodeStatus Kick::onRunning()
{
    auto log = [=](string msg) {
        brain->log->setTimeNow();
        brain->log->log("debug/Kick", rerun::TextLog(msg));
    };

    // [원본 그대로] 킥 중단 조건 (공이 너무 많이 움직였거나 놓쳤을 때)
    bool enableAbort;
    brain->get_parameter("strategy.abort_kick_when_ball_moved", enableAbort);
    auto ballRange = brain->data->ball.range;
    const double MOVE_RANGE_THRESHOLD = 0.3;
    const double BALL_LOST_THRESHOLD = 1000;  
    
    if (
        enableAbort 
        && (
            (brain->data->ballDetected && ballRange - _minRange > MOVE_RANGE_THRESHOLD) 
            || brain->msecsSince(brain->data->ball.timePoint) > BALL_LOST_THRESHOLD 
        )
    ) {
        log("ball moved, abort kick");
        
        // [추가됨] 중단 시 플래그 초기화 (다시 Chase부터 하도록)
        brain->tree->setEntry("ready_to_kick", false);
        
        return NodeStatus::SUCCESS;
    }

    if (ballRange < _minRange) _minRange = ballRange;    

    // [원본 그대로] 킥 도중 장애물 감지 시 회피
    bool avoidPushing;
    brain->get_parameter("obstacle_avoidance.avoid_during_kick", avoidPushing);
    double kickAoSafeDist;
    brain->get_parameter("obstacle_avoidance.kick_ao_safe_dist", kickAoSafeDist);
    
    if (
        avoidPushing
        && brain->data->robotPoseToField.x < brain->config->fieldDimensions.length / 2 - brain->config->fieldDimensions.goalAreaLength
        && brain->distToObstacle(brain->data->ball.yawToRobot) < kickAoSafeDist
    ) {
        brain->client->setVelocity(-0.1, 0, 0);
        
        // [추가됨] 중단 시 플래그 초기화
        brain->tree->setEntry("ready_to_kick", false);
        
        return NodeStatus::SUCCESS;
    }

    // [원본 그대로] 시간 체크 및 종료 처리
    double msecs = getInput<double>("min_msec_kick").value();
    double speedLimit = getInput<double>("speed_limit").value(); // speed 변수명 겹침 주의해서 speedLimit으로 변경 권장하나 원본 유지
    
    // 원본 로직: 시간에 거리/속도 항을 더함
    msecs = msecs + brain->data->ball.range / speedLimit * 1000;
    
    if (brain->msecsSince(_startTime) > msecs) { 
        brain->client->setVelocity(0, 0, 0);
        
        // [추가됨] 킥 완료! 플래그 초기화 (중요)
        brain->tree->setEntry("ready_to_kick", false);
        
        return NodeStatus::SUCCESS;
    }

    // [원본 그대로] 가속 로직 (점점 빨라짐)
    if (brain->data->ballDetected) { 
        double angle = brain->data->ball.yawToRobot;
        // _speed는 멤버 변수
        _speed += 0.1; 
        
        // 입력받은 제한 속도와 비교
        double currentCmdSpeed = min(speedLimit, _speed);
        brain->client->crabWalk(angle, currentCmdSpeed);
    }

    return NodeStatus::RUNNING;
}

void Kick::onHalted()
{
    // [추가됨] 강제 중단 시에도 플래그 초기화 안전장치
    brain->tree->setEntry("ready_to_kick", false);
    
    // [원본]
    _startTime -= rclcpp::Duration(100, 0);
}
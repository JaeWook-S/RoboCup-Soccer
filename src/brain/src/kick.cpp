#include "brain.h"
#include "brain_tree.h"
#include "kick.h"

// BehaviorTree Factory에 Test 노드를 생성하는 함수를 등록하는 역할 -> 코드 양 줄일 수 있음
#define REGISTER_KICK_BUILDER(Name)     \
    factory.registerBuilder<Name>( \
        #Name,                     \
        [brain](const string &name, const NodeConfig &config) { return make_unique<Name>(name, config, brain); });


void RegisterKickNodes(BT::BehaviorTreeFactory &factory, Brain* brain){
    REGISTER_KICK_BUILDER(CalcKickDir)
    REGISTER_KICK_BUILDER(CalcKickDirWithGoalkeeper)
    REGISTER_KICK_BUILDER(CalcPassDir)
    REGISTER_KICK_BUILDER(Kick)
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
    // if (thetal - thetar < crossThreshold) {
    //     brain->data->kickType = "cross";
    //     color = 0xFF00FFFF;
    //     // atan2( 목표Y - 공Y , 목표X - 공X )
    //     // 반코트는 우리팀 진영을 상대팀으로 인식해야하므로 - 값이 들어감
    //     brain->data->kickDir = atan2( 
    //         0 - bPos.y, 
    //         - (fd.length/2 - fd.penaltyDist/2) - bPos.x); // 앞에 - 붙였음
    // }
    // else { 
    //     brain->data->kickType = "shoot";
    //     color = 0x00FF00FF;
    //     brain->data->kickDir = atan2(
    //         0 - bPos.y,
    //         - fd.length/2 - bPos.x // 여기도 - 붙였음
    //     );
    //     // 이것도 - 로 바꿈 + 0에서 M_PI로 바꿈
    //     if (brain->data->ball.posToField.x < - (brain->config->fieldDimensions.length / 2)) brain->data->kickDir = M_PI; 
    // }
    brain->data->kickType = "shoot";
    color = 0x00FF00FF;
    brain->data->kickDir = atan2(
            0 - bPos.y,
            - fd.length/2 - bPos.x // 여기도 - 붙였음
        );
        // 이것도 - 로 바꿈 + 0에서 M_PI로 바꿈
    if (brain->data->ball.posToField.x < - (brain->config->fieldDimensions.length / 2)) brain->data->kickDir = M_PI; 

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

NodeStatus CalcKickDirWithGoalkeeper::tick(){
    double crossThreshold;
    double goalkeeperMargin;
    getInput("cross_threshold", crossThreshold);
    getInput("goalkeeper_margin", goalkeeperMargin);

    auto bPos = brain->data->ball.posToField;
    auto fd = brain->config->fieldDimensions;
    auto color = 0xFFFFFFFF;

    // 1. 골키퍼(장애물) 찾기
    vector<GameObject> obstacles = brain->data->getObstacles();
    vector<GameObject> goalkeepers;
    
    // 골대 주변에 있는 장애물만 골키퍼 후보로 간주
    double goalX = (brain->config->fieldDimensions.length / 2);
    goalX = - (brain->config->fieldDimensions.length / 2); 

    for(const auto& obs : obstacles){
        // 골대 근처 (페널티 박스 + margin)에 있는 로봇을 골키퍼로 인식
        if(obs.posToField.x < goalX + fd.goalAreaLength + goalkeeperMargin){goalkeepers.push_back(obs);}
    }

    // 공 -> 왼쪽 포스트 각도(thetal), 공 -> 오른쪽 포스트 각도(thetar), 골키퍼가 이 사이에 있으면 각도를 좁혀야 함.
    double bestKickDir = 0.0;
    bool isBlocked = false;
    double maxOpenAngle = 0.0;
    string kickType = "shoot";

    // 골키퍼 없으면 기존 로직대로 그냥 킥
    if (goalkeepers.empty()) { bestKickDir = atan2(0 - bPos.y, goalX - bPos.x);} 
    // 골키퍼가 있다면        
    else {        
        vector<pair<double, double>> blockedIntervals;
        for(const auto& gk : goalkeepers){
            double dist = norm(gk.posToField.x - bPos.x, gk.posToField.y - bPos.y);
            double angleToGK = atan2(gk.posToField.y - bPos.y, gk.posToField.x - bPos.x);
            double angularWidth = atan2(goalkeeperMargin, dist);
            blockedIntervals.push_back({angleToGK - angularWidth, angleToGK + angularWidth});
        }
        
        double angleLeftPost = atan2(fd.goalWidth/2 - bPos.y, goalX - bPos.x);
        double angleRightPost = atan2(-fd.goalWidth/2 - bPos.y, goalX - bPos.x);
    
        auto gk = goalkeepers[0];
        double gkAngle = atan2(gk.posToField.y - bPos.y, gk.posToField.x - bPos.x);
        double angleToGoalCenter = atan2(0 - bPos.y, goalX - bPos.x);
        
        double diff = gkAngle - angleToGoalCenter;
        while(diff > M_PI) diff -= 2*M_PI;
        while(diff < -M_PI) diff += 2*M_PI;
        
        // 골키퍼가 중심보다 왼쪽에 있음(diff > 0) -> 오른쪽 포스트 쪽으로 슛
        if(diff > 0) {
            bestKickDir = angleRightPost + (angleToGoalCenter - angleRightPost) * 0.5; // 오른쪽 절반의 중간
        } else {
            bestKickDir = angleLeftPost + (angleToGoalCenter - angleLeftPost) * 0.5; // 왼쪽 절반의 중간
        }
        
        // 만약 골키퍼가 너무 중앙이라 양쪽 다 좁다 or thetal - thetar 자체가 작다면 -> 여기서 바로 cross로 가도 좋을듯
    }

    // 최종 결정
    double angleLeftPost = atan2(fd.goalWidth/2 - bPos.y, goalX - bPos.x);
    double angleRightPost = atan2(-fd.goalWidth/2 - bPos.y, goalX - bPos.x);
    
    double angleDiff = angleLeftPost - angleRightPost;
    while(angleDiff > M_PI) angleDiff -= 2*M_PI;
    while(angleDiff < -M_PI) angleDiff += 2*M_PI;
    
    double goalVisibleAngle = fabs(angleDiff);

    brain->data->kickDir = bestKickDir;

    // 만약 골문이 너무 좁거나(crossThreshold), 확실한 슛 각이 안나오면 cross psss가 된다면 pass로 수정
     if (goalVisibleAngle < crossThreshold) {
        kickType = "cross";
        color = 0xFF00FFFF;
     }
    
    brain->data->kickType = kickType;

    // 시각화
    brain->log->setTimeNow();
    brain->log->log(
        "field/kick_dir_gk",
        rerun::Arrows2D::from_vectors({{10 * cos(brain->data->kickDir), -10 * sin(brain->data->kickDir)}})
            .with_origins({{brain->data->ball.posToField.x, -brain->data->ball.posToField.y}})
            .with_colors({color}) 
            .with_radii(0.015) 
            .with_draw_order(32)
    );

    return NodeStatus::SUCCESS;
}

NodeStatus CalcPassDir::tick(){
    double passThreshold;
    getInput("pass_threshold", passThreshold);

    auto bPos = brain->data->ball.posToField; // 공 위치
    int bestTeammateIdx = -1;
    double minDist = 9999.0;

    // 가장 가까운(혹은 적절한) 팀원 찾기
    for(int i=0; i<HL_MAX_NUM_PLAYERS; i++){
        // 나 자신 제외, 살아있는 팀원 확인
        if(i + 1 == brain->config->playerId) continue;
        if(!brain->data->tmStatus[i].isAlive) continue; 
        
        // 팀원 위치
        auto tmPos = brain->data->tmStatus[i].robotPoseToField;
        
        // 거리 계산
        double dist = norm(bPos.x - tmPos.x, bPos.y - tmPos.y);

        // 유효 거리 내에 있고, 가장 가까운 팀원 선택 (단순 거리 기준)
        if(dist < passThreshold && dist < minDist){
            minDist = dist;
            bestTeammateIdx = i;
        }
    }

    if(bestTeammateIdx != -1){
        brain->data->kickType = "pass"; // 킥 타입 설정
        auto tmPos = brain->data->tmStatus[bestTeammateIdx].robotPoseToField;
        
        // 공에서 팀원 방향으로 킥 방향 설정
        brain->data->kickDir = atan2(tmPos.y - bPos.y, tmPos.x - bPos.x);
        
        brain->log->logToScreen("debug/Pass", format("Passing to TM %d at Dist %.2f", bestTeammateIdx+1, minDist), 0x00FF00FF);
    } else {
        // 줄 사람 없으면 그냥 슛 (기존 로직 fallback)
        brain->data->kickType = "shoot"; 
        brain->data->kickDir = 0.0; // 정면
        brain->log->logToScreen("debug/Pass", "No Teammate found, fallback to shoot", 0xFF0000FF);
    }

    // 시각화
    brain->log->setTimeNow();
    brain->log->log(
        "field/pass_dir",
        rerun::Arrows2D::from_vectors({{10 * cos(brain->data->kickDir), -10 * sin(brain->data->kickDir)}})
            .with_origins({{brain->data->ball.posToField.x, -brain->data->ball.posToField.y}})
            .with_colors({0x00FFFFFF}) // Cyan color for pass
            .with_radii(0.01)
    );

    return NodeStatus::SUCCESS;
}

// 승재욱 - 직접 만든 Kick
// 반코트용으로 부호 및 부등호 바뀐 부분 존재 -> 풀코트로 수정 필요
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

NodeStatus Kick::onStart(){

    // _while로 제어되므로 별도의 state check 불필요
    if(brain->tree->getEntry<string>("striker_state") != "kick") return NodeStatus::SUCCESS;

    _minRange = brain->data->ball.range;
    _speed = 0.5;
    _startTime = brain->get_clock()->now();

    // 장애물 회피 로직
    bool avoidPushing;
    double kickAoSafeDist;
    brain->get_parameter("obstacle_avoidance.avoid_during_kick", avoidPushing);
    brain->get_parameter("obstacle_avoidance.kick_ao_safe_dist", kickAoSafeDist);
    string role = brain->tree->getEntry<string>("player_role");
    
    if (
        avoidPushing
        && (role != "goal_keeper")
        && brain->data->robotPoseToField.x > -(brain->config->fieldDimensions.length / 2 - brain->config->fieldDimensions.goalAreaLength) // 음수 및 부등호 방향 변경 -> 반코트용
        && brain->distToObstacle(brain->data->ball.yawToRobot) < kickAoSafeDist
    ) {
        brain->client->setVelocity(-0.1, 0, 0);
        
        return NodeStatus::SUCCESS;
    }

    // [원본 그대로] 운동 지령 (게걸음 시작)
    double angle = brain->data->ball.yawToRobot;
    brain->client->crabWalk(angle, _speed);
    
    return NodeStatus::RUNNING;
}

NodeStatus Kick::onRunning(){
    auto log = [=](string msg) {
        brain->log->setTimeNow();
        brain->log->log("debug/Kick", rerun::TextLog(msg));
    };

    if(brain->tree->getEntry<string>("striker_state") != "kick") return NodeStatus::SUCCESS;

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
        brain->tree->setEntry("striker_state", "chase"); // 상태 변경은 Decide 노드에 맡김
        
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
        && brain->data->robotPoseToField.x > -(brain->config->fieldDimensions.length / 2 - brain->config->fieldDimensions.goalAreaLength) // 음수 및 부등호 방향 변경 -> 반코트용
        && brain->distToObstacle(brain->data->ball.yawToRobot) < kickAoSafeDist
    ) {
        brain->client->setVelocity(-0.1, 0, 0);
        
        return NodeStatus::SUCCESS;
    }

    // [원본 그대로] 시간 체크 및 종료 처리
    double msecs = getInput<double>("min_msec_kick").value();
    double speedLimit = getInput<double>("speed_limit").value(); // speed 변수명 겹침 주의해서 speedLimit으로 변경 권장하나 원본 유지
    
    // 원본 로직: 시간에 거리/속도 항을 더함
    msecs = msecs + brain->data->ball.range / speedLimit * 1000;
    
    if (brain->msecsSince(_startTime) > msecs) { 
        brain->client->setVelocity(0, 0, 0);

        // 승재욱 추가
        brain->tree->setEntry("striker_state", "chase");
        
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

void Kick::onHalted(){
    // [원본]
    brain->tree->setEntry("striker_state", "chase");
    _startTime -= rclcpp::Duration(100, 0);
}

#include "kick.h"
#include "brain.h"
#include "brain_tree.h"

#include <cstdlib>
#include <ctime>

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


#include <cmath>
#include <cstdlib>

#include "brain_tree.h"
#include "locator.h"
#include "brain.h"

#include "utils/math.h"
#include "utils/print.h"
#include "utils/misc.h"

#include "std_msgs/msg/string.hpp"
#include <fstream>
#include <ios>

#define REGISTER_BUILDER(Name)     \
    factory.registerBuilder<Name>( \
        #Name,                     \
        [this](const string &name, const NodeConfig &config) { return make_unique<Name>(name, config, brain); });

void BrainTree::init(){
    BehaviorTreeFactory factory;

    brain->registerWalkNodes(factory); // walk 관련 노드 등록
    brain->registerMoveHeadNodes(factory); // head move 관련 노드 등록
    brain->registerLocatorNodes(factory); // locator 관련 노드 등록
    brain->registerChaseNodes(factory); // chase 관련 노드 등록
    brain->registerKickNodes(factory); // kick 관련 노드 등록
    brain->registerAdjustNodes(factory); // adjust 관련 노드 등록
    brain->registerDecisionRoleNodes(factory); // decision role 관련 노드 등록
    
    
    factory.registerBehaviorTreeFromFile(brain->config->treeFilePath);
    tree = factory.createTree("MainTree");

    // 여기서 블랙보드가 초기화됨
    initEntry();
}

void BrainTree::initEntry(){
    // 여기서 블랙보드를 초기화하면 됨 
    setEntry<bool>("gamecontroller_isKickOff", true);
    setEntry<string>("gc_game_state", ""); // 현재 ready,set,play,end 중 하나일 것 
    setEntry<string>("gc_game_sub_state_type", "NONE");
    setEntry<string>("gc_game_sub_state", "");
    setEntry<bool>("gc_is_kickoff_side", false); // 우리 팀 킥오프인지
    setEntry<bool>("gc_is_sub_state_kickoff_side", false);
    setEntry<bool>("gc_is_under_penalty", false);
    setEntry<int>("control_state", 3); // control_state == 1 이면 단순 걷기로 (1->3 play.xml test)
    setEntry<string>("decision", "");
    setEntry<bool>("wait_for_opponent_kickoff", false);


    // 실제 경기 중 상황 
    // 공 
    setEntry<bool>("ball_out", false); // 공이 필드 밖으로 나갔는지  
    setEntry<bool>("ball_location_known", false); // 공 위치를 알고 있는지
    setEntry<bool>("tm_ball_pos_reliable", false); // 팀원 공 위치 정보가 신뢰할 만한지 확인
    setEntry<double>("ball_range", 0); // 공과의 거리

    //  승재욱 추가 : chase -> adjust -> kick
    setEntry<string>("striker_state", "chase");   
}

void BrainTree::tick(){ tree.tickOnce(); }

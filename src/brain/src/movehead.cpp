#include "brain.h"
#include "movehead.h"
#include "brain_tree.h"

#include <cstdlib>
#include <ctime>

// BehaviorTree Factory에 Test 노드를 생성하는 함수를 등록하는 역할 -> 코드 양 줄일 수 있음
#define REGISTER_MOVEHEAD_BUILDER(Name)     \
    factory.registerBuilder<Name>( \
        #Name,                     \
        [brain](const string &name, const NodeConfig &config) { return make_unique<Name>(name, config, brain); });


void RegisterMoveHeadNodes(BT::BehaviorTreeFactory &factory, Brain* brain){
    REGISTER_MOVEHEAD_BUILDER(MoveHead) // 속도 제어
    REGISTER_MOVEHEAD_BUILDER(CamFindBall) // 카메라로 공 찾기
    REGISTER_MOVEHEAD_BUILDER(CamTrackBall) // 카메라로 공 추적
    
}

NodeStatus MoveHead::tick(){
    double pitch, yaw;
    getInput("pitch", pitch);
    getInput("yaw", yaw);
    brain->client->moveHead(pitch, yaw);
    return NodeStatus::SUCCESS;
}

// 생성자
CamFindBall::CamFindBall(const string &name, const NodeConfig &config, Brain *_brain) : SyncActionNode(name, config), brain(_brain){
    double lowPitch = 1.0;
    double highPitch = 0.45;
    double leftYaw = 1.1;
    double rightYaw = -1.1;

    _cmdSequence[0][0] = lowPitch;  _cmdSequence[0][1] = leftYaw; // 왼쪽 아래
    _cmdSequence[1][0] = lowPitch;  _cmdSequence[1][1] = 0; // 가운데 아래
    _cmdSequence[2][0] = lowPitch;  _cmdSequence[2][1] = rightYaw; // 오른쪽 아래
    _cmdSequence[3][0] = highPitch; _cmdSequence[3][1] = rightYaw; // 오른쪽 위
    _cmdSequence[4][0] = highPitch; _cmdSequence[4][1] = 0; // 가운데 위
    _cmdSequence[5][0] = highPitch; _cmdSequence[5][1] = leftYaw; // 왼쪽 위

    _cmdIndex = 0;
    _cmdIntervalMSec = 800;
    _cmdRestartIntervalMSec = 50000;
    _timeLastCmd = brain->get_clock()->now();
}

NodeStatus CamFindBall::tick(){
    if (brain->data->ballDetected){ return NodeStatus::SUCCESS; }

    auto curTime = brain->get_clock()->now();
    auto timeSinceLastCmd = (curTime - _timeLastCmd).nanoseconds() / 1e6;

    // 아직 움직이기엔 이름
    if (timeSinceLastCmd < _cmdIntervalMSec){ return NodeStatus::SUCCESS; }
    // 너무 오래 됐으면(50초) 다시 처음으로
    else if (timeSinceLastCmd > _cmdRestartIntervalMSec){ _cmdIndex = 0; }
    // 다음 명령어(시퀀스)로
    else{ _cmdIndex = (_cmdIndex + 1) % (sizeof(_cmdSequence) / sizeof(_cmdSequence[0])); }

    brain->client->moveHead(_cmdSequence[_cmdIndex][0], _cmdSequence[_cmdIndex][1]);
    _timeLastCmd = brain->get_clock()->now();
    return NodeStatus::SUCCESS;
}

// 추가해야할 것 -> 팀원 공이 신뢰 가능한지 변수에 저장하고 신뢰가 정말 가능하다면 그 정보 바탕으로 tracking 진행할 수 있도록 고도화 필요
// communication은 주석 처리 해놨음
NodeStatus CamTrackBall::tick(){
    // 공의 위치가 시야 중심과의 픽셀 차이가 이 허용 오차보다 작으면, 시야 중심에 있다고 판단
    double pitch, yaw, ballX, ballY, deltaX, deltaY;
    const double pixToleranceX = brain->config->camPixX * 3 / 10.; // camPixX - 544, pixToleranceX - 163.2
    const double pixToleranceY = brain->config->camPixY * 3 / 10.; // camPixY - 448, pixToleranceY - 134.4
    const double xCenter = brain->config->camPixX / 2; // 272
    const double yCenter = brain->config->camPixY / 2; // 224

    // 로그 기록
    auto log = [=](string msg) {
        brain->log->setTimeNow();
        brain->log->log("debug/CamTrackBall", rerun::TextLog(msg));
    };

    // 추적 박스 로그 기록
    auto logTrackingBox = [=](int color, string label) {
        brain->log->setTimeNow();
        vector<rerun::Vec2D> mins;
        vector<rerun::Vec2D> sizes;
        mins.push_back(rerun::Vec2D{xCenter - pixToleranceX, yCenter - pixToleranceY});
        sizes.push_back(rerun::Vec2D{pixToleranceX * 2, pixToleranceY * 2});
        brain->log->log(
            "image/track_ball",
            rerun::Boxes2D::from_mins_and_sizes(mins, sizes)
                .with_labels({label})
                .with_colors(color)
        );   
    };

    bool iSeeBall = brain->data->ballDetected; // 공이 실제 Vision에 의해 발견되었는지
    bool iKnowBallPos = brain->tree->getEntry<bool>("ball_location_known"); // 공의 위치가 알고 있는지
    // bool tmBallPosReliable = brain->tree->getEntry<bool>("tm_ball_pos_reliable"); // TM에 의해 공의 위치가 신뢰할 수 있는지
    // if (!(iKnowBallPos || tmBallPosReliable))
    //     return NodeStatus::SUCCESS;

    if(!iKnowBallPos){ return NodeStatus::SUCCESS; }

    if (!iSeeBall){ // 내가 공을 못봤을 때
        if (iKnowBallPos) {
            pitch = brain->data->ball.pitchToRobot;
            yaw = brain->data->ball.yawToRobot;
        } 
        // else if (tmBallPosReliable) {
        //     pitch = brain->data->tmBall.pitchToRobot;
        //     yaw = brain->data->tmBall.yawToRobot;
        // } 
        else { log("reached impossible condition"); }
        logTrackingBox(0x000000FF, "ball not detected"); 
    }
     
    // 내가 공을 볼 때 
    else {      
        ballX = mean(brain->data->ball.boundingBox.xmax, brain->data->ball.boundingBox.xmin); // 공의 x 좌표
        ballY = mean(brain->data->ball.boundingBox.ymax, brain->data->ball.boundingBox.ymin); // 공의 y 좌표
        deltaX = ballX - xCenter; // 공의 x 좌표와 시야 중심의 x 좌표의 차이
        deltaY = ballY - yCenter; // 공의 y 좌표와 시야 중심의 y 좌표의 차이
        
        // 공이 시야 중심에 있다고 판단
        if (std::fabs(deltaX) < pixToleranceX && std::fabs(deltaY) < pixToleranceY){
            auto label = format("ballX: %.1f, ballY: %.1f, deltaX: %.1f, deltaY: %.1f", ballX, ballY, deltaX, deltaY);
            logTrackingBox(0x00FF00FF, label); // 초록색 박스로 바뀜
            return NodeStatus::SUCCESS;
        }

        double smoother = 1.5; // 머리를 스무스하게 움직이기 위한 스무스 계수
        double deltaYaw = deltaX / brain->config->camPixX * brain->config->camAngleX / smoother; // 공의 x 좌표와 시야 중심의 x 좌표의 차이를 이용하여 머리의 yaw를 계산
        double deltaPitch = deltaY / brain->config->camPixY * brain->config->camAngleY / smoother; // 공의 y 좌표와 시야 중심의 y 좌표의 차이를 이용하여 머리의 pitch를 계산

        pitch = brain->data->headPitch + deltaPitch; // 머리의 pitch를 공의 pitch와 더함
        yaw = brain->data->headYaw - deltaYaw; // 머리의 yaw를 공의 yaw와 뺌
        auto label = format("ballX: %.1f, ballY: %.1f, deltaX: %.1f, deltaY: %.1f, pitch: %.1f, yaw: %.1f", ballX, ballY, deltaX, deltaY, pitch, yaw);
        logTrackingBox(0xFF0000FF, label);  // 초록색 박스로 바뀜
    }

    brain->client->moveHead(pitch, yaw);
    return NodeStatus::SUCCESS;
}
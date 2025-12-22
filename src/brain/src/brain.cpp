#include <iostream>
#include <string>
#include <fstream> 
#include <yaml-cpp/yaml.h>

#include "brain.h"
#include "detection_utils.h"

#include "utils/print.h"
#include "utils/math.h"
#include "utils/misc.h"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std;
using std::placeholders::_1;

#define SUB_STATE_QUEUE_SIZE 1

Brain::Brain() : rclcpp::Node("brain_node"){
    // tf 브로드캐스터 초기화
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

    // 매개변수는 반드시 여기에서 먼저 선언해야 한다는 점에 유의해야 한다.
    // 그렇지 않으면 프로그램 내부에서도 읽을 수 없다.
    // yaml 파일에 설정된 파라미터는 계층 구조가 있을 경우 점(.) 표기법으로 가져온다.
    declare_parameter<int>("game.team_id", 29); 
    declare_parameter<int>("game.player_id", 1); // 1 | 2 | 3 | 4 | 5
    declare_parameter<string>("game.field_type", ""); // adult_size | kid_size
    declare_parameter<string>("game.player_role", ""); // striker | goal_keeper
    declare_parameter<bool>("game.treat_person_as_robot", false);
    declare_parameter<int>("game.number_of_players", 4);

    // 로봇 제어 관련 파라미터 -> robot_client.cpp 에서 사용
    declare_parameter<double>("robot.vx_factor", 0.5);
    declare_parameter<double>("robot.yaw_offset", 0.0);
    declare_parameter<double>("robot.vx_limit", 1.0);
    declare_parameter<double>("robot.vy_limit", 0.4);
    declare_parameter<double>("robot.vtheta_limit", 1.0);

    declare_parameter<double>("robot.robot_height", 1.0);
    declare_parameter<double>("robot.odom_factor", 1.0);

    // 전략 관련 파라미터
    declare_parameter<double>("strategy.ball_confidence_threshold", 50.0);   // 공 인식 신뢰도 임계값
    declare_parameter<double>("strategy.ball_memory_timeout", 5.0); // 공의 위치를 얼마나 많은 시간동안 기억할지 정하는 파라미터 (공의 위치를 알고있다고 판단하는 시간)

    // 킥 관련 파라미터
    declare_parameter<bool>("strategy.abort_kick_when_ball_moved", false);
    declare_parameter<bool>("obstacle_avoidance.avoid_during_kick", false);
    declare_parameter<double>("obstacle_avoidance.kick_ao_safe_dist", 1.0);

    // chase 관련 파라미터
    declare_parameter<bool>("chase.limit_near_ball_speed", true);
    declare_parameter<double>("chase.near_ball_speed_limit", 0.3);
    declare_parameter<double>("chase.near_ball_range", 4.0);
    declare_parameter<bool>("obstacle_avoidance.avoid_during_chase", false);
    declare_parameter<double>("obstacle_avoidance.chase_ao_safe_dist", 2.0);

    // obstacle 관련 파라미터
    declare_parameter<int>("obstacle_avoidance.occupancy_threshold", 30);
    declare_parameter<double>("obstacle_avoidance.collision_threshold", 0.4);
    declare_parameter<bool>("obstacle_avoidance.enable_freekick_avoid", false);
    declare_parameter<double>("obstacle_avoidance.obstacle_memory_msecs", 500.0);

    // 카메라 관련 파라미터 
    declare_parameter<string>("vision.image_topic", "/camera/camera/color/image_raw");  // RGB 카메라 이미지 토픽
    declare_parameter<string>("vision.depth_image_topic", "/camera/camera/aligned_depth_to_color/image_raw");  // 정렬된 깊이 이미지 토픽
    declare_parameter<double>("vision.cam_pixel_width", 1280);  // 카메라 가로 해상도
    declare_parameter<double>("vision.cam_pixel_height", 720);  // 카메라 세로 해상도
    declare_parameter<double>("vision.cam_fov_x", 90);  // 카메라 FOV X (도)
    declare_parameter<double>("vision.cam_fov_y", 65);  // 카메라 FOV Y (도)

    declare_parameter<string>("vision_config_path", "");
    declare_parameter<string>("vision_config_local_path", "");

    // 장애물을 인식하는 depth callback 관련 파라미터 
    declare_parameter<int>("depth_obstacle_preprocessing.depth_sample_step", 16);
    declare_parameter<double>("depth_obstacle_preprocessing.obstacle_min_height", 0.15);
    declare_parameter<double>("depth_obstacle_preprocessing.grid_size", 0.2);
    declare_parameter<double>("depth_obstacle_preprocessing.max_x", 0.2);
    declare_parameter<double>("depth_obstacle_preprocessing.max_y", 0.2);
    declare_parameter<double>("depth_obstacle_preprocessing.exclusion_x", 0.25);
    declare_parameter<double>("depth_obstacle_preprocessing.exclusion_y", 0.4);
    declare_parameter<double>("depth_obstacle_preprocessing.ball_exclusion_radius", 0.3);
    declare_parameter<double>("depth_obstacle_preprocessing.ball_exclusion_height", 0.3);

    // rerun 관련 파라미터 
    declare_parameter<bool>("rerunLog.enable_tcp", false);  // TCP 로그 전송 활성화 여부
    declare_parameter<string>("rerunLog.server_ip", "");  // 로그 서버 IP
    declare_parameter<bool>("rerunLog.enable_file", false);  // 파일 로그 활성화 여부
    declare_parameter<string>("rerunLog.log_dir", "");  // 로그 디렉터리
    declare_parameter<double>("rerunLog.max_log_file_mins", 5.0);  // 로그 파일 최대 길이 (분)
    declare_parameter<int>("rerunLog.img_interval", 10);  // 이미지 저장 간격
    
    // locator 관련 파라미터
    declare_parameter<int>("locator.min_marker_count", 5);  // 로컬라이저 최소 마커 수
    declare_parameter<double>("locator.max_residual", 0.3);  // 최대 잔차 (오차 허용값)
    
    // BT
    declare_parameter<string>("tree_file_path", "");

    // 게임 컨트롤러 IP 주소
    declare_parameter<string>("game_control_ip", "0.0.0.0");
}

Brain::~Brain(){}

void Brain::init(){   
    config = std::make_shared<BrainConfig>();
    loadConfig(); // config 변수에 등록

    tree = std::make_shared<BrainTree>(this);
    client = std::make_shared<RobotClient>(this);
    data = std::make_shared<BrainData>();
    locator = std::make_shared<Locator>();
    log = std::make_shared<BrainLog>(this);
   
    tree->init();
    client->init();
    locator->init(config->fieldDimensions, config->pfMinMarkerCnt, config->pfMaxResidual);
    log->prepare();
    
    // 초기 시간 스탬프 설정
    data->lastSuccessfulLocalizeTime = get_clock()->now();  // 마지막 위치추정 성공 시각
    data->timeLastDet = get_clock()->now();  // 마지막 객체 인식 시각
    data->timeLastLineDet = get_clock()->now();  // 마지막 필드라인 감지 시각
    data->timeLastGamecontrolMsg = get_clock()->now();  // 마지막 게임컨트롤 메시지 수신 시각
    data->ball.timePoint = get_clock()->now();  // 마지막 공 위치 업데이트 시각

    // 현재 시각
    auto now = get_clock()->now();
    
    // ROS callback 연결
    detectionsSubscription = create_subscription<vision_interface::msg::Detections>("/booster_vision/detection", SUB_STATE_QUEUE_SIZE, bind(&Brain::detectionsCallback, this, _1));
    subFieldLine = create_subscription<vision_interface::msg::LineSegments>("/booster_vision/line_segments", SUB_STATE_QUEUE_SIZE, bind(&Brain::fieldLineCallback, this, _1));
    odometerSubscription = create_subscription<booster_interface::msg::Odometer>( "/odometer_state", SUB_STATE_QUEUE_SIZE, bind(&Brain::odometerCallback, this, _1));
    lowStateSubscription = create_subscription<booster_interface::msg::LowState>("/low_state", SUB_STATE_QUEUE_SIZE, bind(&Brain::lowStateCallback, this, _1));
    headPoseSubscription = create_subscription<geometry_msgs::msg::Pose>("/head_pose", SUB_STATE_QUEUE_SIZE, bind(&Brain::headPoseCallback, this, _1));
    recoveryStateSubscription = create_subscription<booster_interface::msg::RawBytesMsg>("fall_down_recovery_state", SUB_STATE_QUEUE_SIZE, bind(&Brain::recoveryStateCallback, this, _1));
    // depth 이미지
    string depthTopic = get_parameter("vision.depth_image_topic").as_string();
    depthImageSubscription = create_subscription<sensor_msgs::msg::Image>(depthTopic, SUB_STATE_QUEUE_SIZE, bind(&Brain::depthImageCallback, this, _1));

    // rerun 연결 시에만 사용함 -> 이미지 캡처
    if (config->rerunLogEnableFile || config->rerunLogEnableTCP) {
        string imageTopic = get_parameter("vision.image_topic").as_string();
        imageSubscription = create_subscription<sensor_msgs::msg::Image>(imageTopic, SUB_STATE_QUEUE_SIZE, bind(&Brain::imageCallback, this, _1));
    }

}

// config 로드
void Brain::loadConfig(){
    get_parameter("game.team_id", config->teamId);
    get_parameter("game.player_id", config->playerId);
    get_parameter("game.field_type", config->fieldType);
    get_parameter("game.player_role", config->playerRole);
    get_parameter("game.treat_person_as_robot", config->treatPersonAsRobot);
    get_parameter("game.number_of_players", config->numOfPlayers);

    // 로봇 제어 관련 파라미터
    get_parameter("robot.robot_height", config->robotHeight);
    get_parameter("robot.odom_factor", config->robotOdomFactor);

    get_parameter("robot.vx_factor", config->vxFactor);
    get_parameter("robot.yaw_offset", config->yawOffset);
    get_parameter("robot.vx_limit", config->vxLimit);
    get_parameter("robot.vy_limit", config->vyLimit);
    get_parameter("robot.vtheta_limit", config->vthetaLimit);

    // 전략 관련 파라미터 
    get_parameter("strategy.ball_confidence_threshold", config->ballConfidenceThreshold);  // 공 탐지 신뢰도 임계값

    // chase 관련 파라미터
    get_parameter("chase.limit_near_ball_speed", config->limitNearBallSpeed);
    get_parameter("chase.near_ball_speed_limit", config->nearBallSpeedLimit);
    get_parameter("chase.near_ball_range", config->nearBallRange);

    // obstacle 관련 파라미터
    get_parameter("obstacle_avoidance.collision_threshold", config->collisionThreshold);

    // locator 관련 파리미터
    get_parameter("locator.min_marker_count", config->pfMinMarkerCnt);  // 최소 마커 수
    get_parameter("locator.max_residual", config->pfMaxResidual);  // 최대 허용 잔차 (PF 재샘플 기준)

    // rerun 관련 파라미터 
    get_parameter("rerunLog.enable_tcp", config->rerunLogEnableTCP);  // TCP 로그 활성화
    get_parameter("rerunLog.server_ip", config->rerunLogServerIP);  // 로그 서버 IP
    get_parameter("rerunLog.enable_file", config->rerunLogEnableFile);  // 파일 로그 활성화 여부
    get_parameter("rerunLog.log_dir", config->rerunLogLogDir);  // 로그 저장 디렉토리
    get_parameter("rerunLog.max_log_file_mins", config->rerunLogMaxFileMins);  // 최대 로그 파일 시간
    get_parameter("rerunLog.img_interval", config->rerunLogImgInterval);  // 이미지 로그 주기

    // 카메라 관련 파라미터
    get_parameter("vision.cam_pixel_width", config->camPixX);  // 카메라 해상도 X
    get_parameter("vision.cam_pixel_height", config->camPixY);  // 카메라 해상도 Y
    double camDegX, camDegY;  // 시야각 임시 변수 선언
    get_parameter("vision.cam_fov_x", camDegX);  // FOV X (deg)
    get_parameter("vision.cam_fov_y", camDegY);  // FOV Y (deg)
    config->camAngleX = deg2rad(camDegX);  // 라디안으로 변환
    config->camAngleY = deg2rad(camDegY);  // 라디안으로 변환

    // 从视觉 config 中加载相关参数
    string visionConfigPath, visionConfigLocalPath;  // 비전 설정 파일 경로 변수 선언
    get_parameter("vision_config_path", visionConfigPath);  // 글로벌 설정 파일 경로 읽기
    get_parameter("vision_config_local_path", visionConfigLocalPath);  // 로컬 덮어쓰기 설정 파일 경로 읽기
    if (!filesystem::exists(visionConfigPath)) {  // 파일 존재 확인
        // 报错然后退出
        RCLCPP_ERROR(get_logger(), "vision_config_path %s not exists", visionConfigPath.c_str());  // 에러 로그 출력
        exit(1);  // 비전 설정이 없으면 프로그램 종료
    }
    // else
    YAML::Node vConfig = YAML::LoadFile(visionConfigPath);  // YAML 파일 로드 (전역 설정)
    if (filesystem::exists(visionConfigLocalPath)) {  // 로컬 오버라이드 설정 파일이 존재하면
        YAML::Node vConfigLocal = YAML::LoadFile(visionConfigLocalPath);  // 로컬 YAML 로드
        MergeYAML(vConfig, vConfigLocal);  // 전역 + 로컬 설정 병합
    }
    config->camfx = vConfig["camera"]["intrin"]["fx"].as<double>();  // 카메라 내부 파라미터 fx
    config->camfy = vConfig["camera"]["intrin"]["fy"].as<double>();  // fy
    config->camcx = vConfig["camera"]["intrin"]["cx"].as<double>();  // cx
    config->camcy = vConfig["camera"]["intrin"]["cy"].as<double>();  // cy

    auto extrin = vConfig["camera"]["extrin"];  // 외부 파라미터 행렬 노드 가져오기
    for (int i = 0; i < 4; ++i) {  // 4x4 행렬 복사
        for (int j = 0; j < 4; ++j) {
            config->camToHead(i, j) = extrin[i][j].as<double>();  // YAML 값 → 행렬 요소
        }
    }
    prtDebug(format("camfx: %f, camfy: %f, camcx: %f, camcy: %f", config->camfx, config->camfy, config->camcx, config->camcy));  // 내부 파라미터 출력
    string str_cam2head = "camToHead: \n";  // 외부 파라미터 행렬 문자열로 변환
    for (int i = 0; i < 4; ++i) {  // 행 단위 반복
        for (int j = 0; j < 4; ++j) {  // 열 단위 반복
            str_cam2head += format("%.3f ", config->camToHead(i, j));  // 각 원소를 문자열로 추가
        }
        str_cam2head += "\n";  // 행 끝마다 줄바꿈
    }
    prtDebug(str_cam2head);  // camToHead 행렬 출력 (디버그용)

    // BT 관련 파라미터
    get_parameter("tree_file_path", config->treeFilePath);

    config->handle(); // 맵 관련 정보들 초기화

    ostringstream oss;
    config->print(oss);
    prtDebug(oss.str());
}

void Brain::tick(){ 
    updateBallMemory(); // 공 위치 기억 업데이트
    updateObstacleMemory(); // obstacle 기억 업데이트 
    
    tree->tick(); 
}


/* ----------------------------- time 관련 함수 유틸 -------------------------------*/
// 얼마나 시간이 흘렀는지 확인하는 함수
double Brain::msecsSince(rclcpp::Time time){
    auto now = this->get_clock()->now();
    if (time.get_clock_type() != now.get_clock_type()) return 1e18;
    return (now - time).nanoseconds() / 1e6;
}

/* ------------------------- ROS Callback 관련 함수 구현 -------------------------------*/
void Brain::gameControlCallback(const game_controller_interface::msg::GameControlData &msg){
    data->timeLastGamecontrolMsg = get_clock()->now();
    // 处理比赛的一级状态
    auto lastGameState = tree->getEntry<string>("gc_game_state"); // 比赛的一级状态
    vector<string> gameStateMap = {
        "INITIAL", // 초기 상태, 선수는 경기장 밖에서 대기
        "READY",   // 준비 상태, 선수는 경기장으로 진입하고 자신의 시작 위치로 이동
        "SET",     // 정지 상태, 심판기에서 경기 시작 지시가 내려올 때까지 대기
        "PLAY",    // 정상 경기 진행
        "END"      // 경기 종료
    };
    string gameState = gameStateMap[static_cast<int>(msg.state)]; // 현재 무슨 게임 상태인지 확인
    tree->setEntry<string>("gc_game_state", gameState); // 적용

    bool isKickOffSide = (msg.kick_off_team == config->teamId); // 우리 팀이 킥오프(선공) 팀인지 여부
    tree->setEntry<bool>("gc_is_kickoff_side", isKickOffSide); // 적용

    // 경기의 2차(하위) 상태를 처리
    string gameSubStateType;
    switch (static_cast<int>(msg.secondary_state)) {
        case 0:
            gameSubStateType = "NONE";
            data->realGameSubState = "NONE";
            break;
        case 3:
            gameSubStateType = "TIMEOUT"; // 양 팀의 타임아웃과 심판 타임아웃을 포함
            data->realGameSubState = "TIMEOUT";
            break;

        // 현재 다른 상태는 처리하지 않고, TIMEOUT을 제외한 모든 상태를 FREE_KICK으로 처리
        case 4:
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "DIRECT_FREEKICK"; // 간접 프리킥 
            data->isDirectShoot = true;
            break;
        case 5:
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "INDIRECT_FREEKICK"; // 직접 프리킥
            break;
        case 6:
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "PENALTY_KICK"; // 패널티킥
            data->isDirectShoot = true;
            break;
        case 7:
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "CORNER_KICK"; // 코너 킥
            break;
        case 8:
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "GOAL_KICK"; // 골 킥
            data->isDirectShoot = true;
            break;
        case 9:
            gameSubStateType = "FREE_KICK";
            data->realGameSubState = "THROW_IN"; // 스로우 인
            break;
        default:
            gameSubStateType = "FREE_KICK";
            break;
    }
    
    // STOP: 정지; -> GET_READY: 공격 또는 수비 위치로 이동; -> SET: 대기 자세
    vector<string> gameSubStateMap = {"STOP", "GET_READY", "SET"};
    string gameSubState = gameSubStateMap[static_cast<int>(msg.secondary_state_info[1])];
    tree->setEntry<string>("gc_game_sub_state_type", gameSubStateType); // 현재 어떤 프리킥 상황인지
    tree->setEntry<string>("gc_game_sub_state", gameSubState); // 정지, 공격 또는 수비 위치로 이동, 대기 자세인지 확인

    // 프리킥 상황에서 우리 팀이 선공(킥 수행) 팀인지 여부. 
    bool isSubStateKickOffSide = (static_cast<int>(msg.secondary_state_info[0]) == config->teamId); 
    tree->setEntry<bool>("gc_is_sub_state_kickoff_side", isSubStateKickOffSide);

    game_controller_interface::msg::TeamInfo myTeamInfo;
    game_controller_interface::msg::TeamInfo oppoTeamInfo;

    if (msg.teams[0].team_number == config->teamId) // teams[0]이 우리 팀인 경우 -> 팀 정보 입력
    {
        myTeamInfo = msg.teams[0];
        oppoTeamInfo = msg.teams[1];
    }
    else if (msg.teams[1].team_number == config->teamId) // teams[1]이 우리 팀인 경우
    {
        myTeamInfo = msg.teams[1];
        oppoTeamInfo = msg.teams[0];
    }
    else
    {
        // 데이터 패킷에 우리 팀 정보가 포함되어 있지 않음 → 더 이상 처리하지 않음
        prtErr(format("received invalid game controller message team0 %d, team1 %d, teamId %d",
            msg.teams[0].team_number, msg.teams[1].team_number, config->teamId));
        return;
    }

    int liveCount = 0;
    int oppoLiveCount = 0;
    // 판정(페널티) 상태 처리.
    // penalty[playerId - 1] 은 우리 팀 선수의 페널티 상태를 의미하며,
    // 페널티 상태에 있으면 움직일 수 없음
    for (int i = 0; i < HL_MAX_NUM_PLAYERS; i++) {
        // 우리 팀 선수의 페널티 상태 저장
        data->penalty[i] = static_cast<int>(myTeamInfo.players[i].penalty);
        // 레드카드가 있으면 강제 퇴장 상태로 처리
        if (static_cast<int>(myTeamInfo.players[i].red_card_count) > 0) {
            data->penalty[i] = PENALTY_SUBSTITUTE;
        }
        // 페널티가 없는 선수면 경기 가능한 선수 수 증가
        if (data->penalty[i] == PENALTY_NONE) liveCount++;
        // 상대 팀 선수의 페널티 상태 저장
        data->oppoPenalty[i] = static_cast<int>(oppoTeamInfo.players[i].penalty);
        // 상대 팀 선수의 레드카드가 있으면 강제 퇴장 상태로 처리
        if (static_cast<int>(oppoTeamInfo.players[i].red_card_count) > 0) {
            data->oppoPenalty[i] = PENALTY_SUBSTITUTE;
        }
        // 상대 팀 선수의 페널티가 없는 선수면 경기 가능한 선수 수 증가
        if (data->oppoPenalty[i] == PENALTY_NONE) oppoLiveCount++;
    }
    // 경기 가능한 선수 수 저장
    data->liveCount = liveCount;
    data->oppoLiveCount = oppoLiveCount;

   
    bool lastIsUnderPenalty = tree->getEntry<bool>("gc_is_under_penalty");
    bool isUnderPenalty = (data->penalty[config->playerId - 1] != PENALTY_NONE); // 현재 로봇이 페널티(반칙)를 받는 중인지 여부
    tree->setEntry<bool>("gc_is_under_penalty", isUnderPenalty);

    // 로봇이 페널티를 받으면 경기장 밖으로 나갔다가 다시 들어와야 합니다. 
    // 이때 로봇의 현재 위치 정보(Odometry)가 틀어질 수 있으므로, odom_calibrated(오도메트리 보정 여부)를 false로 설정하여 위치 추정(Localization)을 초기화하겠다는 의미입니다.
    // if (isUnderPenalty && !lastIsUnderPenalty) tree->setEntry<bool>("odom_calibrated", false); 

    // 점수 기록
    data->score = static_cast<int>(myTeamInfo.score);
    data->oppoScore = static_cast<int>(oppoTeamInfo.score);
}

void Brain::detectionsCallback(const vision_interface::msg::Detections &msg){

    // data->camConnected = true;
    // time 관련 변수
    // time 관련 변수
    auto timePoint = detection_utils::timePointFromHeader(msg.header);
    auto now = get_clock()->now();
    data->timeLastDet = timePoint; // 디버깅 시 지연 시간 정보를 출력하기 위해 사용

    auto gameObjects = detection_utils::detectionsToGameObjects(msg, config, data); // 감지된 객체 리스트 GameObject 객체로 변환

    vector<GameObject> balls, goalposts, persons, robots, obstacles, markings;
    for (int i = 0; i < gameObjects.size(); i++){
        const auto &obj = gameObjects[i];
        if (obj.label == "Ball")
            balls.push_back(obj);

        if (obj.label == "Goalpost")
            goalposts.push_back(obj);

        if (obj.label == "Person"){
            persons.push_back(obj);

            if (config->treatPersonAsRobot) // 사람도 로봇으로 다룰건지 
                robots.push_back(obj);
        }
        if (obj.label == "Opponent")
            robots.push_back(obj);
        if (obj.label == "LCross" || obj.label == "TCross" || obj.label == "XCross" || obj.label == "PenaltyPoint")
            markings.push_back(obj);
    }

    // 객체 데이터들 전처리
    detection_utils::detectProcessBalls(balls, config, data, tree);
    detection_utils::detectProcessGoalposts(goalposts, data, log);
    detection_utils::detectProcessMarkings(markings, data, config, log);
    detection_utils::detectProcessRobots(robots, data);

    // Store persons for avoidance even if not treated as robots
    data->setPersons(persons);

    // 处理并记录视野信息
    detection_utils::detectProcessVisionBox(msg, data);

    // 로그 기록
    logDetection(gameObjects);
}

void Brain::fieldLineCallback(const vision_interface::msg::LineSegments &msg){ // 필드 라인 감지 콜백
    auto timePoint = detection_utils::timePointFromHeader(msg.header); // 메시지의 타임스탬프 변환

    auto now = get_clock()->now(); // 현재 시간
    data->timeLastLineDet = timePoint; // 디버깅용 라인 감지 시각 기록

    vector<FieldLine> lines = {}; // 필드 라인 벡터 초기화
    FieldLine line; // 단일 라인 구조체 선언

    double x0, y0, x1, y1, __; // __ is a placeholder for transformations (좌표 변환용 임시 변수)
    for (int i = 0; i < msg.coordinates.size() / 4; i++) { // 한 라인은 4개의 좌표값(x0,y0,x1,y1)으로 구성
        int index = i * 4; // 각 라인의 시작 인덱스 계산
        line.posToRobot.x0 = msg.coordinates[index]; line.posOnCam.x0 = msg.coordinates_uv[index]; // 시작점 좌표 (로봇, 카메라 좌표)
        line.posToRobot.y0 = msg.coordinates[index + 1]; line.posOnCam.y0 = msg.coordinates_uv[index + 1]; // y0 값
        line.posToRobot.x1 = msg.coordinates[index + 2]; line.posOnCam.x1 = msg.coordinates_uv[index + 2]; // 끝점 x1
        line.posToRobot.y1 = msg.coordinates[index + 3]; line.posOnCam.y1 = msg.coordinates_uv[index + 3]; // 끝점 y1
        detection_utils::updateLinePosToField(line, data); // 필드 좌표계로 변환
        line.timePoint = timePoint; // 감지 시점 기록
        // TODO infer line dir and id
        lines.push_back(line); // 결과 벡터에 추가
    }
    lines = detection_utils::processFieldLines(lines, config, data, tree); // 감지된 라인 병합 및 식별 처리
    data->setFieldLines(lines); // 데이터 객체에 저장
}

void Brain::odometerCallback(const booster_interface::msg::Odometer &msg){

    data->robotPoseToOdom.x = msg.x * config->robotOdomFactor;
    data->robotPoseToOdom.y = msg.y * config->robotOdomFactor;
    data->robotPoseToOdom.theta = msg.theta;

    // Odom 정보를 기반으로 Field 좌표계에서 로봇 위치 업데이트
    transCoord(
        data->robotPoseToOdom.x, data->robotPoseToOdom.y, data->robotPoseToOdom.theta,
        data->odomToField.x, data->odomToField.y, data->odomToField.theta,
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta);

    // tf 변환을 퍼블리시
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = this->get_clock()->now();
    transform.header.frame_id = "odom";
    transform.child_frame_id = "base_link";
    
    // 평행 이동(translation) 설정
    transform.transform.translation.x = data->robotPoseToOdom.x;
    transform.transform.translation.y = data->robotPoseToOdom.y;
    transform.transform.translation.z = 0.0;
    
    // 회전(rotation) 설정 (오일러 각을 쿼터니언으로 변환)
    tf2::Quaternion q;
    q.setRPY(0, 0, data->robotPoseToOdom.theta);
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    transform.transform.rotation.w = q.w();

    log->setTimeNow();
    log->log("debug/odom_callback", rerun::TextLog(format("x: %.1f, y: %.1f, z: %.1f", data->robotPoseToOdom.x, data->robotPoseToOdom.y, data->robotPoseToOdom.theta)));
    
    // tf 변환 브로드캐스트
    tf_broadcaster_->sendTransform(transform);

    // Odom 정보 로그 출력

    log->setTimeNow();
    auto color = 0x00FF00FF;
    if (!data->tmImAlive) color = 0x006600FF;
    else if (!data->tmImLead) color = 0x00CC00FF;
    string label = format("Cost: %.1f", data->tmMyCost);
    log->logRobot("field/robot", data->robotPoseToField, color, label, true);
}

void Brain::lowStateCallback(const booster_interface::msg::LowState &msg){
    data->headYaw = msg.motor_state_serial[0].q;
    data->headPitch = msg.motor_state_serial[1].q;
    log->log("debug/head_angles", rerun::TextLog(format("pitch: %.1f, yaw: %.1f", data->headYaw, data->headPitch)));
}

void Brain::headPoseCallback(const geometry_msgs::msg::Pose& msg){
    //  head → base 변환 행렬을 계산
    Eigen::Matrix4d headToBase = Eigen::Matrix4d::Identity();
    
    // 쿼터니언으로부터 회전 행렬을 얻는다
    Eigen::Quaterniond q(
        msg.orientation.w,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z
    );
    headToBase.block<3,3>(0,0) = q.toRotationMatrix();
    
    // 이동(평행이동) 벡터를 설정한다
    headToBase.block<3,1>(0,3) = Eigen::Vector3d(
        msg.position.x,
        msg.position.y,
        msg.position.z
    );

    // cam → base 변환 행렬을 계산하여 저장한다
    data->camToRobot = headToBase * config->camToHead;
}

void Brain::recoveryStateCallback(const booster_interface::msg::RawBytesMsg &msg){
    // uint8_t state; // IS_READY = 0, IS_FALLING = 1, HAS_FALLEN = 2, IS_GETTING_UP = 3,  
    // uint8_t is_recovery_available; // 1 for available, 0 for not available
    // 使用 RobotRecoveryState 结构，将msg里面的msg转换为RobotRecoveryState
    try
    {
        const std::vector<unsigned char>& buffer = msg.msg;
        RobotRecoveryStateData recoveryState;
        memcpy(&recoveryState, buffer.data(), buffer.size());

        vector<RobotRecoveryState> recoveryStateMap = {
            RobotRecoveryState::IS_READY, // 정상
            RobotRecoveryState::IS_FALLING, // 넘어지는 중 
            RobotRecoveryState::HAS_FALLEN, // 완전히 넘어짐
            RobotRecoveryState::IS_GETTING_UP // 일어나는 중
        };
        this->data->recoveryState = recoveryStateMap[static_cast<int>(recoveryState.state)];
        this->data->isRecoveryAvailable = static_cast<bool>(recoveryState.is_recovery_available);
        this->data->currentRobotModeIndex = static_cast<int>(recoveryState.current_planner_index);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}
/*
    depthImageCallback()은 Depth 이미지(깊이맵)를 받아서 3D 포인트로 복원 
    → 로봇 좌표계로 변환 
    → 2D 그리드로 투영해 장애물 점유도를 만들고 
    → GameObject 장애물 목록을 갱신/정리 
    → Rerun으로 시각화 로그까지 남기는 콜백
*/

// 로그 부분 주석 처리함
void Brain::depthImageCallback(const sensor_msgs::msg::Image &msg){
    try {
        // 이미지 데이터가 유효한지 확인
        if (msg.data.empty() || msg.height == 0 || msg.width == 0) { // 비어있으면 종료
            RCLCPP_WARN(get_logger(), "Received empty depth image");
            return;
        }

        // depth 이미지 생성 및 변환
        cv::Mat depthFloat;
        // 이미지 인코딩 형식에 따른 처리
        if (msg.encoding == "16UC1" || msg.encoding == "mono16") {
            size_t expected = (size_t)msg.width * msg.height * sizeof(uint16_t);
            if (msg.data.size() < expected) {
                RCLCPP_ERROR(get_logger(), "Depth mono16 size mismatch");
                return;
            }
            // 16-bit unsigned depth = 보통 mm 단위 
            cv::Mat depthRaw(msg.height, msg.width, CV_16UC1, const_cast<uint8_t*>(msg.data.data()));
            depthRaw.convertTo(depthFloat, CV_32FC1, 1.0 / 1000.0); //  1/1000 곱해서 meter로 변환
        } 
        else if (msg.encoding == "32FC1") {
            // 데이터 크기가 올바른지 확인
            size_t expected_size = msg.height * msg.width * sizeof(float);
            if (msg.data.size() != expected_size) {
                RCLCPP_ERROR(get_logger(), "Depth image size mismatch: expected %zu, got %zu", 
                    expected_size, msg.data.size());
                return;
            }
            // 32-bit float depth = 보통 meter 단위
            depthFloat = cv::Mat(msg.height, msg.width, CV_32FC1, 
                const_cast<float*>(reinterpret_cast<const float*>(msg.data.data()))).clone();
            
        } 
        else {
            RCLCPP_ERROR(get_logger(), "Unsupported depth image encoding: %s", msg.encoding.c_str());
            return;
        }

        vector<rerun::Vec3D> points_robot;  // 로그를 위한 변수

        const double fx = config->camfx;
        const double fy = config->camfy;
        const double cx = config->camcx;
        const double cy = config->camcy;
        
        /* 
            로봇 기준 좌표계에서
	        앞쪽(x): 0m ~ max_x 까지만
	        좌우(y): -max_y ~ +max_y 범위만
            이 영역을 grid_size 간격으로 잘라서 점유 카운팅할 격자를 만든다
         */ 
        const double grid_size = get_parameter("depth_obstacle_preprocessing.grid_size").as_double();  // 그리드 크기
        const double x_min = 0.0, x_max = get_parameter("depth_obstacle_preprocessing.max_x").as_double();
        const double y_min = -get_parameter("depth_obstacle_preprocessing.max_y").as_double();
        const double y_max = -y_min;
        const int grid_x_count = static_cast<int>((x_max - x_min) / grid_size);
        const int grid_y_count = static_cast<int>((y_max - y_min) / grid_size);
        
        // 그리드 점유 배열 생성
        // 각 셀에 “몇 개의 depth 포인트가 들어왔는지” 카운트
	    // 0이면 비어있음, 값이 크면 해당 셀에 포인트가 많이 찍힘(장애물 가능성↑)
        vector<vector<int>> grid_occupied(grid_x_count, vector<int>(grid_y_count, 0));
        
        // 모든 픽셀을 다 쓰면 너무 무거워서 step 간격으로 다운샘플링
        const int sampleStep = get_parameter("depth_obstacle_preprocessing.depth_sample_step").as_int();
        for (int y = 0; y < msg.height; y += sampleStep)
        {
            for (int x = 0; x < msg.width; x += sampleStep)
            {
                float depth = depthFloat.at<float>(y, x);
                if (depth > 0)
                {
                    // 카메라 좌표계로 변환
                    double x_cam = (x - cx) * depth / fx;
                    double y_cam = (y - cy) * depth / fy;
                    double z_cam = depth;

                    // 로봇 좌표계로 변환
                    Eigen::Vector4d point_cam(x_cam, y_cam, z_cam, 1.0);
                    Eigen::Vector4d point_robot = data->camToRobot * point_cam;
                    
                    // 시각화를 위해 사용 -> 0 : 로봇 기준 x, 1 : 로봇 기준 y, 2 : 높이 z
                    points_robot.push_back(rerun::Vec3D{point_robot(0), point_robot(1), point_robot(2)});
                    
                    // 여기부터 장애물로 볼 것인지 필터링을 함
                    const double Z_THRESHOLD = get_parameter("depth_obstacle_preprocessing.obstacle_min_height").as_double();
                    const double EXCLUDE_MAX_X = get_parameter("depth_obstacle_preprocessing.exclusion_x").as_double(); // 로봇 자신의 몸체를 제외
                    const double EXCLUDE_MIN_X = -EXCLUDE_MAX_X;
                    const double EXCLUDE_MAX_Y = get_parameter("depth_obstacle_preprocessing.exclusion_y").as_double(); // 로봇 자신의 몸체를 제외
                    const double EXCLUDE_MIN_Y = -EXCLUDE_MAX_Y;

                    auto isInRange = [&]() { // 장애물 감지 범위 내에 있는지
                        return point_robot(0) >= x_min && point_robot(0) < x_max
                            && point_robot(1) >= y_min && point_robot(1) < y_max;
                    };
                    auto isSelfBody = [&]() { // 로봇 자신의 몸체 영역인지
                        return point_robot(0) >= EXCLUDE_MIN_X && point_robot(0) <= EXCLUDE_MAX_X
                            && point_robot(1) >= EXCLUDE_MIN_Y && point_robot(1) <= EXCLUDE_MAX_Y;
                    };
                    auto isBall = [&]() { // 공의 위치 영역인지
                        double r = get_parameter("depth_obstacle_preprocessing.ball_exclusion_radius").as_double();
                        double h = get_parameter("depth_obstacle_preprocessing.ball_exclusion_height").as_double();
                        return fabs(point_robot(0) - data->ball.posToRobot.x) < r 
                            && fabs(point_robot(1) - data->ball.posToRobot.y) < r
                            && point_robot(2) < h;
                    };
                    
                    // 높이 임계값보다 크고 장애물 감지 범위 내에 있으며, 로봇 자신의 몸이 아니고, 공이 아닐 때 장애물로 간주
                    if ( point_robot(2) > Z_THRESHOLD && isInRange() &&!isSelfBody() &&!isBall()){
                        int grid_x = static_cast<int>((point_robot(0) - x_min) / grid_size);
                        int grid_y = static_cast<int>((point_robot(1) - y_min) / grid_size);
                        grid_occupied[grid_x][grid_y] += 1;
                    }
                }
            }
        }

        auto obs_old = data->getObstacles(); // 이전 장애물
        vector<GameObject> obs_new = {};

        // 이번에 확인된 장애물 기록 
        for (int i = 0; i < grid_x_count; i++) {
            for (int j = 0; j < grid_y_count; j++) {
                if (grid_occupied[i][j] > 0) { // 장애물이면 
                    GameObject obj;
                    obj.label = "Obstacle";
                    obj.timePoint = get_clock()->now();
                    obj.posToRobot.x = x_min + (i + 0.5) * grid_size;
                    obj.posToRobot.y = y_min + (j + 0.5) * grid_size;
                    obj.confidence = grid_occupied[i][j]; // 포인트가 많이 찍힌 셀 = 더 강한 장애물 후보
                    updateFieldPos(obj); // 로봇 좌표(posToRobot)를 필드 좌표(posToField)로 변환
                    obs_new.push_back(obj);
                }
            }
        }

        // 기존 장애물(obs_old) 정리 및 병합 로직
        for (int i = 0; i < obs_old.size(); i++) {
           //   현재 시야 범위 내의 기존 장애물을 먼저 제거. 각도는 대략적으로 계산되었으며, 오프셋을 통해 범위를 적절히 확장
           /*   카메라가 바라보는 FOV 안에 기존 장애물이 있었는데
                이번 프레임 obs_new에 안 잡혔다면
                “없어진 걸로 보고 지워야” 장애물이 계속 누적되지 않음
                단, 여기 각도 계산은 “대략적”이어서 offset으로 범위를 키워서 보수적으로 처리하고 있음.
            */
            double visionLeft = data->headYaw + config->camAngleX / 2;
            double visionRight = data->headYaw - config->camAngleX / 2;
            auto obs = obs_old[i];
            const double offset = 0.20;
            double obsYawLeft = atan2(obs.posToRobot.y - offset, obs.posToRobot.x + offset);
            double obsYawRight = atan2(obs.posToRobot.y + offset, obs.posToRobot.x + offset);
            if (obsYawLeft < visionLeft && obsYawRight > visionRight) continue; 

            // 기존 관측값(obs)과 새로운 관측값이 너무 근접한 경우, 기존 관측값이 더 이상 존재하지 않는 것으로 간주하여 경계 상황에서 관측값이 누적되는 것을 방지
            bool found = false;
            for (int j = 0; j < obs_new.size(); j++) {
                auto obs_n = obs_new[j];
                double dist = norm(obs.posToRobot.x - obs_n.posToRobot.x, obs.posToRobot.y - obs_n.posToRobot.y);
                if (dist < 0.5 * grid_size) { // 시야 밖의 옛 장애물은 “너무 가까운 새 장애물”이 있으면 제거
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // 그 외는 유지
            obs_new.push_back(obs);
        }

        
        data->setObstacles(obs_new); // note :여기서는 시간 초과된 오래된 장애물을 지우지 않고, 틱(tick)에서 정리
        
        log->setTimeSeconds(detection_utils::timePointFromHeader(msg.header).seconds());
        
        // Log depth grid and obstacles
        logDepth(grid_x_count, grid_y_count, grid_occupied, points_robot);
        logObstacles();

    } 
    catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Exception in depth image callback: %s", e.what());
    }
}

// 단순 rerun 시각화용
void Brain::imageCallback(const sensor_msgs::msg::Image &msg){
    static int counter = 0;
    counter++;
    if (counter % config->rerunLogImgInterval == 0)
    {
        // 카메라 연결 상태가 안 좋아서 자동으로 해상도가 낮아지는 경우를 방지하기 위해,
        // CamTrackBall 계산에 영향을 주지 않도록 현재 해상도를 설정값으로 갱신한다
        config->camPixX = msg.width;
        config->camPixY = msg.height;
        log->log("debug/imageCallback", rerun::TextLog(format("img width: %.d, img height: %.d", msg.width, msg.height)));

        cv::Mat image;
        if (msg.encoding == "nv12" || msg.encoding == "NV12") {
            // NV12: Y plane (H x W) + interleaved UV (H/2 x W)
            size_t expected = (size_t)(msg.width * msg.height * 3 / 2);
            if (msg.data.size() < expected) {
                prtErr(format("NV12 buffer too small. got %zu expect >= %zu", msg.data.size(), expected));
                return;
            }
            cv::Mat yuv(msg.height + msg.height / 2, msg.width, CV_8UC1, const_cast<uint8_t*>(msg.data.data()));
            cv::cvtColor(yuv, image, cv::COLOR_YUV2BGR_NV12);
        } else if (msg.encoding == "bgra8") {
            // 创建 OpenCV Mat 对象，处理 BGRA 格式图像
            image = cv::Mat(msg.height, msg.width, CV_8UC4, const_cast<uint8_t *>(msg.data.data()));
            cv::Mat imageBGR;
            // 将 BGRA 转换为 BGR，忽略 Alpha 通道
            cv::cvtColor(image, imageBGR, cv::COLOR_BGRA2BGR);
            image = imageBGR;
        } else if (msg.encoding == "bgr8") {
            // 原有 BGR8 处理逻辑
            image = cv::Mat(msg.height, msg.width, CV_8UC3, const_cast<uint8_t *>(msg.data.data()));
        } else if (msg.encoding == "rgb8") {
            // 原有 RGB8 处理逻辑
            image = cv::Mat(msg.height, msg.width, CV_8UC3, const_cast<uint8_t *>(msg.data.data()));
            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        } else {
            // 处理其他编码格式，或者记录错误日志
            prtErr(format("Unsupported image encoding: %s", msg.encoding.c_str()));
            return;
        }

        std::vector<uint8_t> compressed_image;
        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 10}; // 10 表示压缩质量，可以根据需要调整
        cv::imencode(".jpg", image, compressed_image, compression_params);


        log->setTimeSeconds(detection_utils::timePointFromHeader(msg.header).seconds());
        log->log("image/img", rerun::EncodedImage::from_bytes(compressed_image));
    }
}


/* ----------------------------------------- 변수 업데이트를 위한 함수들 ----------------------------------------- */

void Brain::updateRelativePos(GameObject &obj) {
    Pose2D pf;
    pf.x = obj.posToField.x;
    pf.y = obj.posToField.y;
    pf.theta = 0;
    Pose2D pr = data->field2robot(pf);
    obj.posToRobot.x = pr.x;
    obj.posToRobot.y = pr.y;
    obj.range = norm(obj.posToRobot.x, obj.posToRobot.y);
    obj.yawToRobot = atan2(obj.posToRobot.y, obj.posToRobot.x);
    obj.pitchToRobot = asin(config->robotHeight / obj.range);
}

// 공의 위치를 얼마나 많은 시간동안 기억할지 정하는 함수 
// 팀원 정보를 처리하는 부분 주석 처리됨 -> 나중에 고도화 필요
void Brain::updateBallMemory(){

    double secs = msecsSince(data->ball.timePoint) / 1000; 
    
    double ballMemTimeout;
    get_parameter("strategy.ball_memory_timeout", ballMemTimeout);

    // 시간이 지나면 공의 위치를 기억하지 않음
    if (secs > ballMemTimeout) { 
        tree->setEntry<bool>("ball_location_known", false);
        tree->setEntry<bool>("ball_out", false); 
    }

    updateRelativePos(data->ball);
    // updateRelativePos(data->tmBall);

    tree->setEntry<double>("ball_range", data->ball.range);

    // 로그 기록
    log->setTimeNow();
    log->logBall(
        "field/ball", 
        data->ball.posToField, 
        data->ballDetected ? 0x00FF00FF : 0x006600FF,
        data->ballDetected,
        tree->getEntry<bool>("ball_location_known")
        );
    // log->logBall(
    //     "field/tmBall", 
    //     data->tmBall.posToField, 
    //     0xFFFF00FF,
    //     tree->getEntry<bool>("tm_ball_pos_reliable"),
    //     tree->getEntry<bool>("tm_ball_pos_reliable")
    //     );
}
// 장애물 리스트를 매 주기마다 정리하고 BrainData에 다시 저장하는 함수
// 장애물 리스트 업데이트 시 특수 상황 인지를 위한 함수 (공을 장애물로 인식할지 말지를 결정)
bool Brain::isFreekickStartPlacing() {
    return (tree->getEntry<string>("gc_game_sub_state_type") == "FREE_KICK" && tree->getEntry<string>("gc_game_state") == "PLAY" && tree->getEntry<string>("gc_game_sub_state") == "GET_READY");
}
// 장애물 리스트 업데이트 함수
// 로그는 주석 처리됨
void Brain::updateObstacleMemory() {
   
    auto obstacles = data->getObstacles();
    vector<GameObject> obs_new = {};

    const double OBS_EXPIRE_TIME = get_parameter("obstacle_avoidance.obstacle_memory_msecs").as_double();
    for (int i = 0; i < obstacles.size(); i++) {
        auto obs = obstacles[i];
        if (obs.label == "Ball") continue; 
        // 오래된 장애물 제거
        if (msecsSince(obs.timePoint) > OBS_EXPIRE_TIME)  continue; 

        updateRelativePos(obs);
        obs_new.push_back(obs);
    }

    /* 
        특수 상황
        freekick 시작 상황에서 공 근처로 접근하면 안되니까 공을 장애물처럼 취급 
        READY 상태에도 공을 장애물처럼 취급
    */ 
    if (
        (get_parameter("obstacle_avoidance.enable_freekick_avoid").as_bool() && isFreekickStartPlacing())
        || tree->getEntry<string>("gc_game_state") == "READY"
    ) {
        obs_new.push_back(data->ball);
    }

    data->setObstacles(obs_new);
    // logObstacles();
}


/* ------------------------- 나중에 따로 뺄 거임 ----------------------------------*/
void Brain::calibrateOdom(double x, double y, double theta){

    double x_or, y_or, theta_or; // or = odom to robot
    x_or = -cos(data->robotPoseToOdom.theta) * data->robotPoseToOdom.x - sin(data->robotPoseToOdom.theta) * data->robotPoseToOdom.y;
    y_or = sin(data->robotPoseToOdom.theta) * data->robotPoseToOdom.x - cos(data->robotPoseToOdom.theta) * data->robotPoseToOdom.y;
    theta_or = -data->robotPoseToOdom.theta;

    
    transCoord(x_or, y_or, theta_or,
               x, y, theta,
               data->odomToField.x, data->odomToField.y, data->odomToField.theta);


    transCoord(
        data->robotPoseToOdom.x, data->robotPoseToOdom.y, data->robotPoseToOdom.theta,
        data->odomToField.x, data->odomToField.y, data->odomToField.theta,
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta);


    double placeHolder;
    // ball
    transCoord(
        data->ball.posToRobot.x, data->ball.posToRobot.y, 0,
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta,
        data->ball.posToField.x, data->ball.posToField.y, placeHolder 
    );

    // robots
    auto robots = data->getRobots();
    for (int i = 0; i < robots.size(); i++) {
        updateFieldPos(robots[i]);
    }
    data->setRobots(robots);

    // goalposts
    auto goalposts = data->getGoalposts();
    for (int i = 0; i < goalposts.size(); i++) {
        updateFieldPos(goalposts[i]);
    }
    
    // markers
    auto markings = data->getMarkings();
    for (int i = 0; i < markings.size(); i++) {
        updateFieldPos(markings[i]);
    }

    // relog
    log->setTimeNow();
    // logVisionBox(get_clock()->now());
    vector<GameObject> gameObjects = {};
    if(data->ballDetected) gameObjects.push_back(data->ball);
    for (int i = 0; i < markings.size(); i++) gameObjects.push_back(markings[i]);
    for (int i = 0; i < robots.size(); i++) gameObjects.push_back(robots[i]);
    for (int i = 0; i < goalposts.size(); i++) gameObjects.push_back(goalposts[i]);
    logDetection(gameObjects);
}

bool Brain::isBoundingBoxInCenter(BoundingBox boundingBox, double xRatio, double yRatio) {
    double x = (boundingBox.xmin + boundingBox.xmax) / 2.0;
    double y = (boundingBox.ymin + boundingBox.ymax) / 2.0;

    return (x  > config->camPixX * (1 - xRatio) / 2)
        && (x < config->camPixX * (1 + xRatio) / 2)
        && (y > config->camPixY * (1 - yRatio) / 2)
        && (y < config->camPixY * (1 + yRatio) / 2);
}

void Brain::updateFieldPos(GameObject &obj) {
    Pose2D pr;
    pr.x = obj.posToRobot.x;
    pr.y = obj.posToRobot.y;
    pr.theta = 0;
    Pose2D pf = data->robot2field(pr);
    obj.posToField.x = pf.x;
    obj.posToField.y = pf.y;
    obj.range = norm(obj.posToRobot.x, obj.posToRobot.y);
    obj.yawToRobot = atan2(obj.posToRobot.y, obj.posToRobot.x);
    obj.pitchToRobot = asin(config->robotHeight / obj.range);
}

void Brain::logDetection(const vector<GameObject> &gameObjects, bool logBoundingBox) {
    if (gameObjects.size() == 0) {
        if (logBoundingBox) log->log("image/detection_boxes", rerun::Clear::FLAT);
        log->log("field/detection_points", rerun::Clear::FLAT);
        // log->log("robotframe/detection_points", rerun::Clear::FLAT);
        return;
    }
    
    // else 
    rclcpp::Time timePoint = gameObjects[0].timePoint;
    log->setTimeSeconds(timePoint.seconds());

    map<std::string, rerun::Color> detectColorMap = {
        {"LCross", rerun::Color(0xFFFF00FF)},
        {"TCross", rerun::Color(0x00FF00FF)},
        {"XCross", rerun::Color(0x0000FFFF)},
        {"Person", rerun::Color(0xFF00FFFF)},
        {"Goalpost", rerun::Color(0x00FFFFFF)},
        {"Opponent", rerun::Color(0xFF0000FF)},
        {"PenaltyPoint", rerun::Color(0xFF9900FF)},
    };

    // for logging boundingBoxes
    vector<rerun::Vec2D> mins;
    vector<rerun::Vec2D> sizes;
    vector<rerun::Text> labels;
    vector<rerun::Color> colors;

    // for logging marker points in robot frame
    vector<rerun::Vec2D> points;
    vector<rerun::Vec2D> points_r; // robot frame
    vector<double> radiis;

    for (int i = 0; i < gameObjects.size(); i++)
    {
        auto obj = gameObjects[i];
        auto label = obj.label;
        labels.push_back(rerun::Text(
            format("%s x:%.2f y:%.2f c:%.1f", 
                label == "Opponent" || label == "Person" ? (label + "[" + obj.color + "]").c_str() : label.c_str(), 
                obj.posToRobot.x, 
                obj.posToRobot.y, 
                obj.confidence)
            )
        );
        points.push_back(rerun::Vec2D{obj.posToField.x, -obj.posToField.y}); // y 取反是因为 rerun Viewer 的坐标系是左手系。转一下看起来更方便。
        points_r.push_back(rerun::Vec2D{obj.posToRobot.x, -obj.posToRobot.y});
        mins.push_back(rerun::Vec2D{obj.boundingBox.xmin, obj.boundingBox.ymin});
        sizes.push_back(rerun::Vec2D{obj.boundingBox.xmax - obj.boundingBox.xmin, obj.boundingBox.ymax - obj.boundingBox.ymin});

        // if (obj.label == "Opponent") radiis.push_back(0.5);
        radiis.push_back(0.1);

        auto color = rerun::Color(0xFFFFFFFF);

        auto it = detectColorMap.find(label);
        if (it != detectColorMap.end())
        {
            color = detectColorMap[label];
        }
        else
        {
            // do nothing, use default
            // colors.push_back(rerun::Color(0xFFFFFFFF));
        }
        if (label == "Ball" && detection_utils::isBallOut(0.2, 10.0, config, data))
            color = rerun::Color(0x000000FF);
        if (label == "Ball" && obj.confidence < config->ballConfidenceThreshold)
            color = rerun::Color(0xAAAAAAFF);
        colors.push_back(color);
    }

    
    if (logBoundingBox) log->log("image/detection_boxes",
             rerun::Boxes2D::from_mins_and_sizes(mins, sizes)
                 .with_labels(labels)
                 .with_colors(colors));

    log->log("field/detection_points",
             rerun::Points2D(points)
                 .with_colors(colors)
                 .with_radii(radiis)
             // .with_labels(labels)
    );
}

/*-------------------------------------------- 공통으로 쓰이는 판단 로직 함수 --------------------------------------------*/
// obstacle과 관련된 로직 함수들 
double Brain::distToObstacle(double angle) {
    auto obs = data->getObstacles();
    double minDist = 1e9;
    // double obstacleThreshold = static_cast<double>(config->obstacleThreshold);
    double obstacleThreshold = static_cast<double>(get_parameter("obstacle_avoidance.occupancy_threshold").as_int());

    double collisionThreshold = config->collisionThreshold;

    for (int i = 0; i < obs.size(); i++) {
        if (obs[i].confidence < obstacleThreshold) continue;

        auto o = obs[i];
        Line line = {
            0, 0,
            cos(angle) * 100, sin(angle) * 100
        };
        double perpDist = fabs(pointPerpDistToLine(Point2D{o.posToRobot.x, o.posToRobot.y}, line));
        if (perpDist < collisionThreshold) {
            double dist = innerProduct(vector<double>{o.posToRobot.x, o.posToRobot.y}, vector<double>{cos(angle), sin(angle)});
            if (dist > 0 && dist < minDist) {
                minDist = dist;
            }
        }
    }

    // Include Robots (Opponents) in avoidance
    auto robots = data->getRobots();
    for (const auto& r : robots) {
        Line line = { 0, 0, cos(angle) * 100, sin(angle) * 100 };
        double perpDist = fabs(pointPerpDistToLine(Point2D{r.posToRobot.x, r.posToRobot.y}, line));
        
        // Robots have 20cm radius approx, plus safety margin
        if (perpDist < collisionThreshold + 0.2) { 
             double dist = innerProduct(vector<double>{r.posToRobot.x, r.posToRobot.y}, vector<double>{cos(angle), sin(angle)});
             if (dist > 0 && dist < minDist) {
                 minDist = dist;
             }
        }
    }

    // Include Persons in avoidance
    auto persons = data->getPersons();
    for (const auto& p : persons) {
        Line line = { 0, 0, cos(angle) * 100, sin(angle) * 100 };
        double perpDist = fabs(pointPerpDistToLine(Point2D{p.posToRobot.x, p.posToRobot.y}, line));
        
        // Persons are larger, give them more margin
        if (perpDist < collisionThreshold + 0.3) { 
             double dist = innerProduct(vector<double>{p.posToRobot.x, p.posToRobot.y}, vector<double>{cos(angle), sin(angle)});
             if (dist > 0 && dist < minDist) {
                 minDist = dist;
             }
        }
    }
    
    return minDist;
}

vector<double> Brain::findSafeDirections(double startAngle, double safeDist, double step) {
    double safeAngleLeft = startAngle;
    double safeAngleRight = startAngle;
    double leftFound = 0;
    double rightFound = 0;
    for (double angle = startAngle; angle < startAngle + M_PI; angle += step) {
        if (distToObstacle(angle) > safeDist) {
            safeAngleLeft = angle;
            leftFound = 1;
            break;
        }
    }
    for (double angle = startAngle; angle > startAngle - M_PI; angle -= step) {
        if (distToObstacle(angle) > safeDist) {
            safeAngleRight = angle;
            rightFound = 1;
            break;
        }
    }

    return vector<double>{leftFound, toPInPI(safeAngleLeft), rightFound, toPInPI(safeAngleRight)};
}

double Brain::calcAvoidDir(double startAngle, double safeDist) {
    auto res = findSafeDirections(startAngle, safeDist);
    bool leftFound = res[0] > 0.5;
    bool rightFound = res[2] > 0.5;
    double angleLeft = res[1];
    double angleRight = res[3]; 
    double determinedAngle = 0;
    if (leftFound && rightFound) {
        determinedAngle = fabs(angleLeft) < fabs(angleRight) ? angleLeft : angleRight;
    } else if (leftFound) {
        determinedAngle = angleLeft;
    } else if (rightFound) {
        determinedAngle = angleRight;
    } else {
        return 0;
    }
    return toPInPI(determinedAngle);
}

// kick과 관련된 로직 함수들
// 이것도 반코트로 로직 수정했음 -> 나중에 풀코트로 수정 필요
vector<double> Brain::getGoalPostAngles(const double margin){ // 공(ball) 위치를 기준으로 왼쪽·오른쪽 골포스트가 이루는 각도(θ)를 계산해서 반환
    
    double leftX, leftY, rightX, rightY; 

    // 해당 로직이 풀코트로 작성됐음 -> + 방향이 상대방이므로 골대의 위치가 양수로 돼있음 
    // leftX = config->fieldDimensions.length / 2;
    // leftY = config->fieldDimensions.goalWidth / 2;
    // rightX = config->fieldDimensions.length / 2;
    // rightY = -config->fieldDimensions.goalWidth / 2;

    // 반코트로 수정
    leftX = -config->fieldDimensions.length / 2; // 음수로 변경
    leftY = config->fieldDimensions.goalWidth / 2;
    rightX = -config->fieldDimensions.length / 2; // 음수로 변경
    rightY = -config->fieldDimensions.goalWidth / 2;

    auto goalposts = data->getGoalposts();

    // 현재 상대방 골대로 라벨링돼있는 골대만 가져오는 중 -> 반코트로 수정
    // for (int i = 0; i < goalposts.size(); i++){
    //     auto post = goalposts[i];

    //     if (post.name == "OL"){ // OL은 opponent goal의 left를 의미
    //         leftX = post.posToField.x;
    //         leftY = post.posToField.y;
    //     }
    //     else if (post.name == "OR"){ // OR은 opponent goal의 right를 의미
    //         rightX = post.posToField.x;
    //         rightY = post.posToField.y;
    //     }
    // }

    // 반코트로 수정 
    // for (int i = 0; i < goalposts.size(); i++){
    //     auto post = goalposts[i];

    //     if (post.name == "SL"){ // SL은 self goal의 left를 의미
    //         leftX = post.posToField.x;
    //         leftY = post.posToField.y;
    //     }
    //     else if (post.name == "SR"){ // SR은 self goal의 right를 의미
    //         rightX = post.posToField.x;
    //         rightY = post.posToField.y;
    //     }
    // }

    // 공 기준에서 골대 방향 각도 계산
    // 결과 : 공 -> 골대 방향
    // margin은 골대를 안쪽으로 살짝 좁힘 -> 포스트 맞고 튀는 슛 방지를 위해 
    // 반코트로 수정했기 때문에 left, right 크기가 바뀌는걸 고민해보기
    const double theta_l = atan2(leftY - margin - data->ball.posToField.y, leftX - data->ball.posToField.x);
    const double theta_r = atan2(rightY + margin - data->ball.posToField.y, rightX - data->ball.posToField.x);

    vector<double> vec = {theta_l, theta_r};
    return vec;
}

/* ---------------------------------------------------------------------------- 로그 관련 함수 -------------------------------------------------------------------- */
void Brain::logObstacles() {
    // log->setTimeNow();
    // time is set on the outside
    
    // 장애물(즉, 점유된 그리드)을 기록함
    auto obs = data->getObstacles();
    vector<rerun::Vec2D> points;
    vector<rerun::Color> colors;
    vector<rerun::Text> labels;
    const int occThreshold = get_parameter("obstacle_avoidance.occupancy_threshold").as_int();
    for (int i = 0; i < obs.size(); i++) {
        auto o = obs[i];

        if (o.confidence < occThreshold) continue; // 이 로직이 아래 로직을 덮어쓰기 때문에, 주석 처리하면 서로 다른 신뢰도(confidence)를 다른 색으로 로그할 수 있음

        points.push_back(rerun::Vec2D{o.posToField.x, -o.posToField.y});
        double mem_msecs = get_parameter("obstacle_avoidance.obstacle_memory_msecs").as_double();
        auto age = msecsSince(o.timePoint);
        uint8_t alpha = static_cast<uint8_t>(0xFF - 0xFF * age / mem_msecs);
        uint32_t color = (o.confidence > occThreshold) ? (0xFF000000 | alpha) : (0xFFFF0000 | alpha);
        colors.push_back(rerun::Color(color));

        labels.push_back(rerun::Text(format("count: %.0f age: %.0fms", o.confidence, age)));
    }
    log->log(
        "field/obstacles", 
        rerun::Points2D(points)
        .with_colors(colors)
        .with_labels(labels)
        .with_radii(0.1)
    );
}

void Brain::logDepth(int grid_x_count, int grid_y_count, vector<vector<int>> &grid_occupied, vector<rerun::Vec3D> &points_robot) {
    // time is set on the outside
    const double grid_size = get_parameter("depth_obstacle_preprocessing.grid_size").as_double();  // 网格大小
    const double x_min = 0.0, x_max = get_parameter("depth_obstacle_preprocessing.max_x").as_double();
    const double y_min = -get_parameter("depth_obstacle_preprocessing.max_y").as_double();
    const double y_max = -y_min;

    // 记录原始点云和网格
    vector<rerun::Position3D> vertices;
    vector<rerun::Color> vertex_colors;
    vector<array<uint32_t, 3>> triangle_indices;
    const int OCCUPANCY_THRESHOLD = get_parameter("obstacle_avoidance.occupancy_threshold").as_int(); // 设置一个显示用的阈值

    for (int i = 0; i < grid_x_count; i++) {
        for (int j = 0; j < grid_y_count; j++) {
            if (grid_occupied[i][j] > 0) {
                // 计算有障碍网格的四个顶点坐标
                double x0 = x_min + i * grid_size;
                double y0 = y_min + j * grid_size;
                double x1 = x0 + grid_size;
                double y1 = y0 + grid_size;

                // 添加四个顶点
                uint32_t base_index = vertices.size();
                vertices.push_back({x0, y0, 0.0});
                vertices.push_back({x1, y0, 0.0});
                vertices.push_back({x1, y1, 0.0});
                vertices.push_back({x0, y1, 0.0});

                // 设置颜色：根据占用情况设置不同的红色
                rerun::Color color;
                if (grid_occupied[i][j] > OCCUPANCY_THRESHOLD) {
                    color = rerun::Color(255, 0, 0, 255);  // RGBA, 红色
                } else {
                    color = rerun::Color(255, 255, 0, 255);  // RGBA, 黄色
                }
                vertex_colors.push_back(color);
                vertex_colors.push_back(color);
                vertex_colors.push_back(color);
                vertex_colors.push_back(color);

                // 添加两个三角形面
                triangle_indices.push_back({base_index, base_index + 1, base_index + 2});
                triangle_indices.push_back({base_index, base_index + 2, base_index + 3});
            }
        }
    }

    vector<uint32_t> point_colors;
    for (auto &point : points_robot) {
        float z_val = std::clamp(point.z(), 0.0f, 1.0f);
        const double obstacleMinHeight = get_parameter("depth_obstacle_preprocessing.obstacle_min_height").as_double();
        if (z_val < obstacleMinHeight) {
            point_colors.push_back(0x0000FFFF);
        } else {
            uint8_t r = static_cast<uint8_t>(z_val * 255);
            uint8_t g = static_cast<uint8_t>((1 - z_val) * 255);
            point_colors.push_back((r << 24) | (g << 16) | 0xFF);
        }
    }
    
    log->log("depth/depth_points",
            rerun::Points3D(points_robot)
                .with_radii(0.01)
                .with_colors(point_colors)
    );
    
    log->log("depth/grid_mesh",
            rerun::Mesh3D(vertices)
                .with_vertex_colors(vertex_colors)
                .with_triangle_indices(triangle_indices)
    );

    // ball exclusion box
    double r = get_parameter("depth_obstacle_preprocessing.ball_exclusion_radius").as_double();
    double h = get_parameter("depth_obstacle_preprocessing.ball_exclusion_height").as_double();
    log->log(
        "depth/ball_exclusion_box",
        rerun::Boxes3D::from_centers_and_half_sizes(
            {{ data->ball.posToRobot.x, data->ball.posToRobot.y, h/2}},
            {{ r, r, h/2}})
        .with_colors(0x00FF0044)     // 半透明绿色 
    );
}
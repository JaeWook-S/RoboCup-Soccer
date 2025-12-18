#pragma once

#include <string>
#include <ostream>

#include "types.h"
#include "utils/math.h"
#include "RoboCupGameControlData.h"


using namespace std;


class BrainConfig
{
public:
    // ------------ 팀원, 필드 관련 정보 ----------------
    int teamId;                
    int playerId;               
    string fieldType;           
    string playerRole;          

    bool treatPersonAsRobot = false;  
    int numOfPlayers = 3;

    // CAM 관련 변수
    double camPixX = 1280;
    double camPixY = 720;
    double camAngleX = deg2rad(90);
    double camAngleY = deg2rad(65);

    double camfx = 643.898;
    double camfy = 643.216;
    double camcx = 649.038;
    double camcy = 357.21;
    Eigen::Matrix4d camToHead;

    // --------- 경기장 크기 저장 변수 ----------------
    FieldDimensions fieldDimensions;
    vector<FieldLine> mapLines;   // 필드 라인 지도
    vector<MapMarking> mapMarkings; // 필드 마크 지도
    void calcMapLines();
    void calcMapMarkings();

    // locator 관련 변수
    int pfMinMarkerCnt = 5;
    double pfMaxResidual = 0.3;
    
    // ----------- BT 관련 변수 --------------------
    string treeFilePath; // behavior tree 파일 경로 -> 런치에서 지정


    // ---- 로봇 제어를 위한 파라미터 (robot_client에서 사용) ----------
    double headYawLimitLeft = 1.1; // 머리 제어
    double headYawLimitRight = -1.1; 
    double headPitchLimitUp = 0.45; 
    double vxLimit = 0.8; // 속도 제한
    double vyLimit = 0.4;
    double vthetaLimit = 1.0;

    double vxFactor; //crabWalk에 사용
    double yawOffset;    

    double robotOdomFactor; // 로봇 odometer에 사용

    // --------- 공 감지 관련 파라미터 ----------------
    double ballConfidenceThreshold;
    double ballOutThreshold = 2.0; // 공이 밖으로 나갔는지 판단을 위한 임계값


    // --------- rerun 로그 관련 파라미터 ----------------
    bool rerunLogEnableTCP;    
    string rerunLogServerIP;    
    bool rerunLogEnableFile;    
    string rerunLogLogDir;     
    double rerunLogMaxFileMins;         

    int rerunLogImgInterval;   

    // ---------------- 멤버 함수 ----------------
    void handle();
};
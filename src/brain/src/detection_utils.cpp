#include "detection_utils.h"
#include "utils/math.h" // norm, transCoord 등
#include <cmath>

namespace detection_utils {

rclcpp::Time timePointFromHeader(const std_msgs::msg::Header &header) {
    auto stamp = header.stamp;
    auto sec = stamp.sec;
    auto nanosec = stamp.nanosec;
    if (sec <= 0 || nanosec <= 0) {
        sec = 1;
        nanosec = 1;
        prtErr(format("Negative time: sec: %d nanosec: %d"));  // 잘못된 시간 로그
    }
    return rclcpp::Time(sec, nanosec, RCL_ROS_TIME);
}

std::vector<GameObject> detectionsToGameObjects(const vision_interface::msg::Detections &detections, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data){
    std::vector<GameObject> res;
    auto timePoint = timePointFromHeader(detections.header);

    for (size_t i = 0; i < detections.detected_objects.size(); ++i) {
        auto obj = detections.detected_objects[i];
        GameObject gObj;

        gObj.timePoint = timePoint;
        gObj.label = obj.label;
        gObj.color = obj.color;

        if (obj.target_uv.size() == 2) {
            gObj.precisePixelPoint.x = static_cast<double>(obj.target_uv[0]);
            gObj.precisePixelPoint.y = static_cast<double>(obj.target_uv[1]);
        }

        // 객체 바운딩 박스 및 신뢰도
        gObj.boundingBox.xmax = obj.xmax;
        gObj.boundingBox.xmin = obj.xmin;
        gObj.boundingBox.ymax = obj.ymax;
        gObj.boundingBox.ymin = obj.ymin;
        gObj.confidence = obj.confidence;

        // 프로젝션 좌표
        gObj.posToRobot.x = obj.position_projection[0];
        gObj.posToRobot.y = obj.position_projection[1];

        // 거리 및 각도 계산
        gObj.range = norm(gObj.posToRobot.x, gObj.posToRobot.y);
        gObj.yawToRobot = atan2(gObj.posToRobot.y, gObj.posToRobot.x);
        gObj.pitchToRobot = atan2(config->robotHeight, gObj.range);

        // 좌표 변환
        transCoord(
            gObj.posToRobot.x, gObj.posToRobot.y, 0,
            data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta,
            gObj.posToField.x, gObj.posToField.y, gObj.posToField.z
        );

        res.push_back(gObj);
    }

    return res;
}

void detectProcessBalls(const std::vector<GameObject> &ballObjs, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data, const std::shared_ptr<BrainTree> &tree){ // 감지된 공 객체들을 처리
    double bestConfidence = 0; // 최고 신뢰도 초기화
    int indexRealBall = -1;  // 진짜 공이라고 판단한 후보의 index

    for (int i = 0; i < ballObjs.size(); i++){ // 감지된 모든 공 순회
        auto ballObj = ballObjs[i]; // 현재 공 객체
        auto oldBall = data->ball; // 이전 프레임의 공 상태 저장

        if (ballObj.posToRobot.x < -0.5 || ballObj.posToRobot.x > 15.0) // 감지 위치가 비정상적이면
            continue; // 무시

        if (ballObj.confidence < config->ballConfidenceThreshold) // 신뢰도 임계값 이하
            continue; // 무시

    /* 앞으로 추가하려는 기능:
	•	로봇 몸 위에 있는 것 제거
	•	필드 밖
	•	갑자기 위치가 크게 튀는 공 제거
	•	가려졌을 때(occlusion):
	•	갑자기 사라졌지만 가려진 경우 → 오래 믿기
	•	가림도 없는데 사라짐 → 빨리 버리기 */

        // 남은 공 중, 신뢰도가 가장 높은 공을 찾음
        if (ballObj.confidence > bestConfidence){ // 최고 신뢰도 갱신 조건
            bestConfidence = ballObj.confidence; // 갱신
            indexRealBall = i; // 진짜 공 후보 인덱스
        }
    }

    if (indexRealBall >= 0){ // 진짜 공 발견 시
        data->ballDetected = true; // 공 감지 플래그 true

        data->ball = ballObjs[indexRealBall]; // 감지된 공 저장
        data->ball.confidence = bestConfidence; // 신뢰도 갱신

        tree->setEntry<bool>("ball_location_known", true); // 행동트리에 공 위치 인식됨 플래그 전달
        updateBallOut(config, data, tree); // 공이 필드 밖인지 확인 후 갱신
    }
    else{ 
        // log->setTimeNow(); // 로그 타임스탬프 갱신
        data->ballDetected = false; // 감지 실패
        data->ball.boundingBox.xmin = 0; // 박스 초기화
        data->ball.boundingBox.xmax = 0;
        data->ball.boundingBox.ymin = 0;
        data->ball.boundingBox.ymax = 0;
    }

    // 로봇 → 공 방향 계산 (항상 실행됨)
    data->robotBallAngleToField = atan2(data->ball.posToField.y - data->robotPoseToField.y, data->ball.posToField.x - data->robotPoseToField.x); // 필드 기준 공 방향 계산
}


bool isBallOut(double locCompareDist, double lineCompareDist, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data){// 공이 필드 밖으로 나갔는지 판정하는 함수 (좌표 및 필드 라인 기준)

    auto ball = data->ball;  // 현재 공 데이터
    auto fd = config->fieldDimensions;  // 필드 치수

    // 공의 X, Y 좌표가 필드 경계 + 여유거리보다 크면 out 처리
    if (fabs(ball.posToField.x) > fd.length / 2 + locCompareDist)
        return true;
    if (fabs(ball.posToField.y) > fd.width / 2 + locCompareDist)
        return true;

    // 필드 라인을 이용한 정밀 판정
    auto fieldLines = data->getFieldLines();
    for (int i = 0; i < fieldLines.size(); i++) {
        auto line = fieldLines[i];
        if (
            (line.type == LineType::TouchLine || line.type == LineType::GoalLine)
            && line.confidence > 1.0  // 신뢰도 높은 라인만 사용
        ) {
            Point2D p = {ball.posToField.x, ball.posToField.y};  // 공의 2D 좌표
            // prtWarn(format("Ball: %.2f, %.2f PerpDist: %.2f", ball.posToField.x, ball.posToField.y, pointPerpDistToLine(p, line.posToField)));
            // 공이 라인에서 일정 거리 이상 떨어져 있으면 out 처리
            if (pointPerpDistToLine(p, line.posToField) > lineCompareDist) return true;
        }
    }
    return false;  // 모든 조건을 통과하지 않으면 in-field로 간주
}

void updateBallOut(const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data, const std::shared_ptr<BrainTree> &tree) {
    // 공이 필드 밖으로 나갔는지를 판단하고, 그 상태를 tree에 반영하는 함수
    bool lastBallOut = tree->getEntry<bool>("ball_out");  // 이전 프레임의 공 out 상태를 불러옴
    double range = lastBallOut ? 4.0 : 2.5;  // 이전에 out이었다면 감지 범위를 완화 (최대 4m)
    double threshold = config->ballOutThreshold;  // 기본 출계 판단 거리 임계값

    // 프리킥 중이면 출계 판정 완화 (1m 여유)
    threshold += (data->isFreekickKickingOff ? 1.0 : 0.0); 

    // 이전 프레임이 out이었다면 판정을 느슨하게 해서 흔들림 방지
    threshold *= (lastBallOut ? 1.0 : 1.5);  

    // 실제 출계 여부 계산: isBallOut() && 공이 너무 멀지 않음
    tree->setEntry<bool>("ball_out", isBallOut(threshold, 10.0, config, data) && data->ball.range < range);
}

void updateLinePosToField(FieldLine& line, const std::shared_ptr<BrainData> &data) { // 필드 라인의 좌표를 로봇 좌표계에서 필드 좌표계로 변환하는 함수
    double __;
    transCoord( // 첫 번째 끝점 좌표 변환
        line.posToRobot.x0, line.posToRobot.y0, 0, // 로봇 기준선의 시작점 (x0, y0)
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta, // 로봇의 필드 좌표 내 위치
        line.posToField.x0, line.posToField.y0, __ // 변환 결과 저장 (필드 기준 좌표)
    );
    transCoord( // 두 번째 끝점 좌표 변환
        line.posToRobot.x1, line.posToRobot.y1, 0, // 로봇 기준선의 끝점 (x1, y1)
        data->robotPoseToField.x, data->robotPoseToField.y, data->robotPoseToField.theta, // 로봇 포즈
        line.posToField.x1, line.posToField.y1, __ // 결과 저장
    );
}

vector<FieldLine> processFieldLines(vector<FieldLine>& fieldLines, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data, const std::shared_ptr<BrainTree> &tree) { // 감지된 라인 후처리
    vector<FieldLine> original = fieldLines; // 원본 복사
    vector<FieldLine> res; // 결과 벡터
    
    int sizeBefore = original.size(); // 병합 전 라인 수
    // merge lines that are actually the same line
    for (int i = 0; i < original.size(); i++) { // 모든 라인 쌍 비교
        for (int j = i + 1; j < original.size(); j++) {
            auto line1 = original[i].posToField;
            auto line2 = original[j].posToField;
            if (isSameLine(line1, line2, 0.1, 1.0)) { // 동일 라인인지 확인
                FieldLine mergedLine;
                mergedLine.posToField = mergeLines(line1, line2); // 병합된 필드 좌표
                mergedLine.posToRobot = mergeLines(original[i].posToRobot, original[j].posToRobot); // 로봇 좌표 병합
                mergedLine.posOnCam = mergeLines(original[i].posOnCam, original[j].posOnCam); // 카메라 좌표 병합
                mergedLine.timePoint = original[i].timePoint; // 시간은 첫 라인 기준

                // replace first line in original with merged line and remove second line
                original[i] = mergedLine;
                original.erase(original.begin() + j);
                j--;
            }
        }
    }
    int sizeAfter = original.size(); // 병합 후 라인 수

    // filter out lines that are too short and infer direction while ditch lines whose dir cannot be inferred
    double valve = 0.2; // 최소 길이 임계값
    for (int i = 0; i < original.size(); i++) {
        auto line = original[i];
        auto lineDir = atan2(line.posToField.y1 - line.posToField.y0, line.posToField.x1 - line.posToField.x0); // 라인 각도 계산

        if (fabs(toPInPI(lineDir - M_PI)) < 0.1 || fabs(lineDir) < 0.1) line.dir = LineDir::Vertical; // 수직선 판별
        else if (fabs(toPInPI(lineDir - M_PI/2)) < 0.1 || fabs(toPInPI(lineDir + M_PI/2)) < 0.1) line.dir = LineDir::Horizontal; // 수평선 판별
        else continue; // 애매하면 건너뜀

        // if line is direction can be verified, check if it is long enough
        if (lineLength(line.posToField) > valve) { // 충분히 긴 경우만 추가
            res.push_back(line);
        }
    }

    // identify each line 
    for (int i = 0; i < res.size(); i++) {
        identifyFieldLine(res[i], config, data, tree); // 라인 식별 수행
    }
    return res; // 최종 결과 반환
}

void identifyFieldLine(FieldLine& line, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data, const std::shared_ptr<BrainTree> &tree) { // 감지된 라인의 실제 필드상 의미(GoalLine, TouchLine 등) 식별
    auto mapLines = config->mapLines; // 미리 정의된 필드 라인 지도 불러오기
    FieldLine mapLine; // 비교용 라인 구조체
    double confidence; // 신뢰도
    line.type = LineType::NA; // 기본은 식별 불가 상태

    double bestConfidence = 0; // 최고 신뢰도 초기화
    double secondBestConfidence = 0; // 두 번째 후보 신뢰도
    int bestIndex = -1; // 가장 잘 맞는 라인 인덱스

    for (int i = 0; i < mapLines.size(); i++) { // 모든 기준 라인과 비교
        mapLine = mapLines[i]; // 비교 대상
        confidence = line.dir == mapLine.dir ?  // 방향 일치 시만 비교
            probPartOfLine(line.posToField, mapLine.posToField) // 선의 일부분일 확률 계산
            : 0.0; // 방향 불일치 시 신뢰도 0

        // Boost confidence with other features
        if (mapLine.type == LineType::GoalLine) {  // 골라인의 경우
            confidence += 0.3 * markCntOnFieldLine("TCross", line, 0.2, data); // T자 마킹 개수로 신뢰도 보정
            confidence += 0.5 * goalpostCntOnFieldLine(line, 0.2, data); // 골포스트 근접도 보정
            if (
                isBallOnFieldLine(line, data) // 공이 이 라인 위에 있고
                && (tree->getEntry<string>("gc_game_sub_state") == "GET_READY" || tree->getEntry<string>("gc_game_sub_state") == "SET") // 준비 상태이면서
                && (data->realGameSubState == "CORNER_KICK") // 코너킥 상황이라면
            ) confidence += 0.3; // 추가 신뢰도 부여
        }
        if (mapLine.type == LineType::MiddleLine) { // 중앙선의 경우
            confidence += 0.3 * markCntOnFieldLine("XCross", line, 0.2); // X자 교차점 근처 여부 보정
            if (
                isBallOnFieldLine(line, data)
                && (tree->getEntry<string>("gc_game_sub_state") == "GET_READY" || tree->getEntry<string>("gc_game_sub_state") == "SET")
                && (data->realGameSubState == "GOAL_KICK") // 골킥 상황이라면
            ) confidence += 0.3; // 신뢰도 보정
        }
        if (mapLine.type == LineType::TouchLine) { // 터치라인의 경우
            if (
                isBallOnFieldLine(line, data)
                && (tree->getEntry<string>("gc_game_sub_state") == "GET_READY" || tree->getEntry<string>("gc_game_sub_state") == "SET")
                && (data->realGameSubState == "GOAL_KICK" || data->realGameSubState == "CORNER_KICK" || data->realGameSubState == "THROW_IN")
            ) confidence += 0.3; // 발차기, 코너, 스로인 상황일 때 보정
        }
        
        // 防止将 goalarealine 误认为 goalline
        auto fd = config->fieldDimensions; // 필드 크기 정보 로드
        if (
            mapLine.type == LineType::GoalLine
            && fabs(line.posToField.y0) < fd.goalAreaWidth / 2 + 0.5
            && fabs(line.posToField.y1) < fd.goalAreaWidth / 2 + 0.5
        ) confidence -= 0.3; // 골 에어리어 라인을 골라인으로 잘못 인식하지 않도록 감점

        // 防止将 penalty area 误认为 touchline
        if (
            mapLine.type == LineType::TouchLine
            && min(fabs(line.posToField.x0), fabs(line.posToField.x1)) > fd.length / 2.0 -  fd.penaltyAreaLength - 0.5
            && line.posToField.x0 * line.posToField.x1 > 0
        ) confidence -= 0.3; // 패널티 구역 라인을 터치라인으로 오인 방지

        double length = norm(line.posToField.x0 - line.posToField.x1, line.posToField.y0 - line.posToField.y1); // 라인 길이 계산
        if (length < 0.5) confidence -= 0.5; // 너무 짧으면 감점
        else if (length < 1.0) confidence -= 0.1; // 짧으면 약간 감점
        
        if (confidence > bestConfidence) { // 최고 신뢰도 갱신
            secondBestConfidence = bestConfidence;
            bestConfidence = confidence;
            bestIndex = i;
        }
    }

    if (bestConfidence - secondBestConfidence < 0.5) bestConfidence -= 0.5; // 후보 간 차이가 작으면 감점

    if (bestIndex >= 0 && bestIndex < mapLines.size()) { // 가장 잘 맞는 라인이 존재하면
        line.type = mapLines[bestIndex].type; // 라인 타입 복사
        line.half = mapLines[bestIndex].half; // 필드 절반(O/S)
        line.side = mapLines[bestIndex].side; // 좌우 방향
        line.confidence = bestConfidence; // 신뢰도 저장
        return; // 종료
    }

    // else 
    line.type = LineType::NA; // 인식 실패 시 기본값 설정
    line.half = LineHalf::NA;
    line.side = LineSide::NA;
    line.confidence = 0.0;
    return; // 반환
}

int markCntOnFieldLine(const string markType, const FieldLine line, const std::shared_ptr<BrainData> &data, const double margin) { // 특정 마킹 종류가 라인 근처에 몇 개 있는지 계산
    int cnt = 0; // 카운터 초기화
    auto markings = data->getMarkings(); // 현재 감지된 마킹 리스트 가져오기
    for (int i = 0; i < markings.size(); i++) { // 모든 마킹 순회
        auto marking = markings[i]; // 현재 마킹 선택
        if (marking.label == markType) { // 마킹 종류 일치 여부 확인
            Point2D point = {marking.posToField.x, marking.posToField.y}; // 마킹 위치를 필드 좌표로 변환
            if (fabs(pointPerpDistToLine(point, line.posToField)) < margin) { // 라인과의 수직 거리 비교
                cnt += 1; // 범위 내 마킹이면 카운트 증가
            }
        }
    }
    return cnt; // 결과 반환
}

int goalpostCntOnFieldLine(const FieldLine line, const std::shared_ptr<BrainData> &data, const double margin) { // 골포스트가 라인 근처에 몇 개 있는지 계산
    int cnt = 0; // 카운터 초기화
    auto goalposts = data->getGoalposts(); // 감지된 골포스트 목록 가져오기
    for (int i = 0; i < goalposts.size(); i++) { // 모든 골포스트 순회
        auto post = goalposts[i]; // 현재 골포스트 선택
        Point2D point = {post.posToField.x, post.posToField.y}; // 위치를 2D 포인트로 변환
        if (pointMinDistToLine(point, line.posToField) < margin) { // 라인과의 최소 거리 비교
            cnt += 1; // 기준 이하이면 카운트 증가
        }
    }
    return cnt; // 결과 반환
}

bool isBallOnFieldLine(const FieldLine line, const std::shared_ptr<BrainData> &data, const double margin) { // 공이 특정 라인 위에 있는지 판단
    auto ballPos = data->ball.posToField; // 공의 필드 좌표 가져오기
    Point2D point = {ballPos.x, ballPos.y};  // 2D 포인트로 변환
    return fabs(pointPerpDistToLine(point, line.posToField)) < margin; // 수직 거리가 margin보다 작으면 true
}

} // namespace detection_utils
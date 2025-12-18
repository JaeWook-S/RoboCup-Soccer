#pragma once

#include <vector>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <vision_interface/msg/detections.hpp>
#include "types.h"
#include "brain_config.h"
#include "brain_data.h"

#include "brain_tree.h"

namespace detection_utils { // namespace detection_utils  

// 시간 관련 함수
rclcpp::Time timePointFromHeader(const std_msgs::msg::Header &header);

// 감지된 객체들을 게임 객체로 변환
std::vector<GameObject> detectionsToGameObjects(const vision_interface::msg::Detections &detections, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data);

/* ----------------------------- Ball 전처리 ----------------------------- */
// 감지된 공 객체들을 처리
void detectProcessBalls(const std::vector<GameObject> &ballObjs, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data, const std::shared_ptr<BrainTree> &tree);
// 공이 필드 밖으로 나갔는지 판단
bool isBallOut(double locCompareDist, double lineCompareDist);
void updateBallOut(const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data, const std::shared_ptr<BrainTree> &tree);


/* ----------------------------- Line 전처리 ----------------------------- */
// 필드 라인 필터링 함수
void updateLinePosToField(FieldLine& line, const std::shared_ptr<BrainData> &data);
vector<FieldLine> processFieldLines(vector<FieldLine>& fieldLines, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data, const std::shared_ptr<BrainTree> &tree);
void identifyFieldLine(FieldLine& line, const std::shared_ptr<BrainConfig> &config, const std::shared_ptr<BrainData> &data, const std::shared_ptr<BrainTree> &tree);
// 마킹 개수 계산
int markCntOnFieldLine(const string markType, const FieldLine line, const double margin, const std::shared_ptr<BrainData> &data);
// 골포스트 개수 계산
int goalpostCntOnFieldLine(const FieldLine line, const double margin, const std::shared_ptr<BrainData> &data);
// 공이 특정 라인 위에 있는지 판단
bool isBallOnFieldLine(const FieldLine line, const double margin, const std::shared_ptr<BrainData> &data);

}
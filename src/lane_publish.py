#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from velodyne_process.msg import CentroidWithLabelArray
import math
from datetime import datetime, timedelta

class LaneCenterToPath:
    def __init__(self):
        # /lane_center 토픽 구독
        rospy.Subscriber("/lane_center", Marker, self.lane_center_callback)
        # /centroid_info 토픽 구독 (장애물 감지)
        rospy.Subscriber("/centroid_info", CentroidWithLabelArray, self.centroid_callback)

        # /lane_path 퍼블리셔 설정
        self.path_pub = rospy.Publisher("/lane_path", Path, queue_size=10)

        # 경로 초기화
        self.path = Path()
        self.path.header.frame_id = "map"  # 전역 좌표계
        self.fixed_path = None  # 고정된 패스를 저장할 변수
        self.obstacles = []
        self.required_obstacle_count = 2  # 장애물이 2개 이상일 때 패스 고정
        self.extend_distance = 10.0  # 고정된 패스를 연장할 거리
        self.extended_once = False  # 한 번만 경로 연장 여부 플래그
        self.fixed_path_start_time = None  # 고정된 경로 시작 시간
        self.fixed_path_duration = timedelta(seconds=10)  # 고정된 경로 유지 시간 (10초)
        self.avoidance_offset = 0.5  # 장애물을 피하기 위한 경로 오프셋 (50cm)

        # 장애물 감지 시간 초기화
        self.obstacle_detected_time = None
        self.detection_duration = timedelta(seconds=1)  # 1초 동안 장애물 감지

    def lane_center_callback(self, marker_msg):
        if self.fixed_path is None:
            # Marker 메시지를 Path 형식으로 변환
            self.path.header.stamp = rospy.Time.now()
            self.path.poses = []

            for point in marker_msg.points:
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "map"
                pose.pose.position.x = point.x
                pose.pose.position.y = point.y
                pose.pose.position.z = point.z
                pose.pose.orientation.w = 1.0
                self.path.poses.append(pose)

            # 현재 lane_center로부터 받은 경로 퍼블리시
            self.path_pub.publish(self.path)

    def centroid_callback(self, msg):
        self.obstacles = msg.centroids
        if len(self.obstacles) >= self.required_obstacle_count:
            if self.obstacle_detected_time is None:
                self.obstacle_detected_time = datetime.now()
            elif datetime.now() - self.obstacle_detected_time >= self.detection_duration:
                if self.fixed_path is None:
                    self.fixed_path = self.create_avoidance_path(self.path)
                    self.fixed_path_start_time = datetime.now()
                    self.extended_once = True
                    rospy.loginfo("Fixed path with obstacle avoidance established.")
                self.obstacle_detected_time = None
        else:
            self.obstacle_detected_time = None

    def create_avoidance_path(self, path):
        if len(path.poses) < 2 or len(self.obstacles) < 2:
            return path

        avoidance_path = Path()
        avoidance_path.header = path.header

        # 장애물에 따른 y 값 조정
        first_obstacle, second_obstacle = self.obstacles[0], self.obstacles[1]
        
        for pose in path.poses:
            adjusted_pose = PoseStamped()
            adjusted_pose.header = pose.header
            adjusted_pose.pose.position.x = pose.pose.position.x
            adjusted_pose.pose.position.z = pose.pose.position.z
            adjusted_pose.pose.orientation = pose.pose.orientation
            
            # 첫 장애물이 왼쪽에 있는 경우 y를 -0.5로 조정하여 우측 회피
            if first_obstacle.centroid.y > 0:
                adjusted_pose.pose.position.y = pose.pose.position.y - self.avoidance_offset
            # 두 번째 장애물이 오른쪽에 있는 경우 y를 +0.5로 조정하여 좌측 회피
            elif second_obstacle.centroid.y < 0:
                adjusted_pose.pose.position.y = pose.pose.position.y + self.avoidance_offset
            else:
                adjusted_pose.pose.position.y = pose.pose.position.y  # 그대로 유지

            avoidance_path.poses.append(adjusted_pose)

        # 장애물 회피 이후 기존 경로 유지 및 10m 연장
        last_pose = path.poses[-1].pose.position
        second_last_pose = path.poses[-2].pose.position
        direction_x = last_pose.x - second_last_pose.x
        direction_y = last_pose.y - second_last_pose.y
        direction_norm = math.sqrt(direction_x**2 + direction_y**2)
        direction_x /= direction_norm
        direction_y /= direction_norm

        for i in range(1, int(self.extend_distance) + 1):
            extended_pose = PoseStamped()
            extended_pose.header.frame_id = "map"
            extended_pose.header.stamp = rospy.Time.now()
            extended_pose.pose.position.x = last_pose.x + direction_x * i
            extended_pose.pose.position.y = last_pose.y + direction_y * i
            extended_pose.pose.position.z = last_pose.z
            extended_pose.pose.orientation.w = 1.0
            avoidance_path.poses.append(extended_pose)

        rospy.loginfo(f"Path extended by {self.extend_distance}m with obstacle avoidance.")
        return avoidance_path

    def update_path(self):
        if self.fixed_path is not None and self.extended_once:
            self.path_pub.publish(self.fixed_path)
            rospy.loginfo("Publishing fixed path with obstacle avoidance.")

            if datetime.now() - self.fixed_path_start_time >= self.fixed_path_duration:
                rospy.loginfo("Fixed path duration exceeded. Switching back to lane_center path.")
                self.fixed_path = None
                self.extended_once = False

if __name__ == '__main__':
    rospy.init_node('lane_center_to_path')
    lane_center_to_path = LaneCenterToPath()
    rospy.Timer(rospy.Duration(0.1), lambda event: lane_center_to_path.update_path())
    rospy.spin()

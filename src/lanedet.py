#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
from math import sqrt, fabs, pow, atan2, degrees
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev


class LaneCandidatePublisher:
    def __init__(self):
        rospy.init_node('lane_candidate_publisher', anonymous=True)

        self.velodyne_points_topic = "/filtered_lane"
        rospy.Subscriber(self.velodyne_points_topic, PointCloud2, self.point_cloud_callback)
        self.lane_marker_pub = rospy.Publisher("/lane_marker", Marker, queue_size=1)
        self.lane_center_pub = rospy.Publisher("/lane_center", Marker, queue_size=1)

        self.max_angle_change = rospy.get_param("~max_angle_change", 50)  # 각도 변화 제한 (기본값 50도)

    def point_cloud_callback(self, msg):
        points = self.extract_points_and_intensities(msg)

        # 차선 감지
        left_points, right_points = self.extract_road_line(points)

        # 차선 부드럽게 만들기
        left_points = self.smooth_line(left_points)
        right_points = self.smooth_line(right_points)

        # 가운데 차선 생성
        center_points = self.extract_center_line(left_points, right_points)

        # 1미터 예측 차선 생성
        predicted_center_points = self.predict_center_line(center_points, 2)

        # 감지된 차선 시각화
        self.publish_lane_marker(left_points, right_points, msg.header)
        self.publish_lane_center(predicted_center_points, msg.header)

    def extract_points_and_intensities(self, cloud_msg):
        points = []
        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def extract_road_line(self, points):
        # 왼쪽 차선과 오른쪽 차선 후보를 분리
        left_points = []
        right_points = []

        for point in points:
            if point[1] < 0:  # 왼쪽 차선 후보
                left_points.append(point)
            else:  # 오른쪽 차선 후보
                right_points.append(point)

        # x축 기준으로 정렬하여 차선이 도로의 길이 방향을 따라 정렬되도록
        left_points = sorted(left_points, key=lambda p: p[0])
        right_points = sorted(right_points, key=lambda p: p[0])

        return left_points, right_points

    def smooth_line(self, points):
        # 차선을 부드럽게 만들기 위해 스플라인 보간 적용
        points = np.array(points)
        if len(points) > 2:
            try:
                # 스플라인 차수를 동적으로 조정하여 m > k 조건을 만족
                k = min(3, len(points) - 1)
                tck, u = splprep(points.T, s=10, k=k)
                smooth_points = splev(u, tck)
                points = np.vstack(smooth_points).T

                # 각도 변화 제한 적용 및 튀는 값 무시
                filtered_points = [points[0]]
                for i in range(1, len(points)):
                    prev_point = filtered_points[-1]
                    curr_point = points[i]
                    angle = degrees(atan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0]))
                    if fabs(angle) <= self.max_angle_change:
                        filtered_points.append(curr_point)
                points = np.array(filtered_points)
            except ValueError as e:
                rospy.logwarn(f"Splprep failed with error: {e}. Skipping smoothing for this line.")
        return points

    def extract_center_line(self, left_points, right_points):
        # 왼쪽 차선 또는 오른쪽 차선 중 하나만 존재할 경우 중앙선을 생성
        if len(left_points) > 1 and len(right_points) > 1:
            # 왼쪽 차선과 오른쪽 차선의 중간 지점을 계산하여 가운데 차선 생성
            center_points = []
            for lp, rp in zip(left_points, right_points):
                center_x = (lp[0] + rp[0]) / 2.0
                center_y = (lp[1] + rp[1]) / 2.0
                center_z = (lp[2] + rp[2]) / 2.0
                center_points.append([center_x, center_y, center_z])
        elif len(left_points) > 1:
            # 왼쪽 차선만 있는 경우, 일정 오프셋을 적용하여 중앙선 생성
            offset = -1.5  # 도로 폭의 절반 (예: 3.5미터)
            center_points = [[p[0], p[1] + offset, p[2]] for p in left_points]
        elif len(right_points) > 1:
            # 오른쪽 차선만 있는 경우, 일정 오프셋을 적용하여 중앙선 생성
            offset = 1.5  # 도로 폭의 절반 (예: 3.5미터)
            center_points = [[p[0], p[1] - offset, p[2]] for p in right_points]
        else:
            center_points = []

        # 가운데 차선을 부드럽게 만들기 위해 스플라인 보간 적용
        center_points = np.array(center_points)
        if len(center_points) > 2:
            try:
                k = min(3, len(center_points) - 1)
                tck, u = splprep(center_points.T, s=10, k=k)
                smooth_points = splev(u, tck)
                center_points = np.vstack(smooth_points).T

                # 각도 변화 제한 적용 및 튀는 값 무시
                filtered_points = [center_points[0]]
                for i in range(1, len(center_points)):
                    prev_point = filtered_points[-1]
                    curr_point = center_points[i]
                    angle = degrees(atan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0]))
                    if fabs(angle) <= self.max_angle_change:
                        filtered_points.append(curr_point)
                center_points = np.array(filtered_points)
            except ValueError as e:
                rospy.logwarn(f"Splprep failed with error: {e}. Skipping smoothing for center line.")

        return center_points

    def predict_center_line(self, center_points, distance):
        if len(center_points) < 2:
            return center_points

        # 각 구간의 기울기 계산
        slopes = []
        for i in range(1, len(center_points)):
            prev_point = center_points[i - 1]
            curr_point = center_points[i]
            slope_angle = atan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0])
            slopes.append(slope_angle)

        # 평균 기울기 계산
        average_slope = np.mean(slopes)

        # 마지막 점에서 평균 기울기를 사용해 예측 점 계산
        last_point = center_points[-1]
        predicted_points = [last_point]

        for i in range(1, 3):  # 예측할 포인트 수
            delta_x = distance * i
            delta_y = delta_x * np.tan(average_slope)
            predicted_point = [last_point[0] + delta_x, last_point[1] + delta_y, last_point[2]]
            predicted_points.append(predicted_point)

        predicted_points = np.vstack([center_points, predicted_points])
        
        return predicted_points

    def publish_lane_marker(self, left_points, right_points, header):
        # 왼쪽 차선 시각화
        if len(left_points) > 1:
            left_marker = Marker()
            left_marker.header = header
            left_marker.ns = "lane_candidates_left"
            left_marker.id = 0
            left_marker.type = Marker.LINE_STRIP
            left_marker.action = Marker.ADD
            left_marker.pose.orientation.w = 1.0
            left_marker.scale.x = 0.1
            left_marker.color.a = 1.0
            left_marker.color.b = 1.0  # 왼쪽 차선을 파란색으로 표시

            for point in left_points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                left_marker.points.append(p)

            try:
                self.lane_marker_pub.publish(left_marker)
            except rospy.exceptions.ROSException as e:
                rospy.logwarn(f"Failed to publish left lane marker: {e}")

        # 오른쪽 차선 시각화
        if len(right_points) > 1:
            right_marker = Marker()
            right_marker.header = header
            right_marker.ns = "lane_candidates_right"
            right_marker.id = 1
            right_marker.type = Marker.LINE_STRIP
            right_marker.action = Marker.ADD
            right_marker.pose.orientation.w = 1.0
            right_marker.scale.x = 0.1
            right_marker.color.a = 1.0
            right_marker.color.r = 1.0  # 오른쪽 차선을 빨간색으로 표시

            for point in right_points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                right_marker.points.append(p)

            try:
                self.lane_marker_pub.publish(right_marker)
            except rospy.exceptions.ROSException as e:
                rospy.logwarn(f"Failed to publish right lane marker: {e}")

    def publish_lane_center(self, predicted_center_points, header):
        
        predicted_marker = Marker()
        predicted_marker.header = header
        predicted_marker.ns = "predicted_lane_candidates_center"
        predicted_marker.id = 3
        predicted_marker.type = Marker.LINE_STRIP
        predicted_marker.action = Marker.ADD
        predicted_marker.pose.orientation.w = 1.0
        predicted_marker.scale.x = 0.1
        predicted_marker.color.a = 1.0
        predicted_marker.color.g = 0.5  # 예측된 가운데 차선을 연두색으로 표시

        for point in predicted_center_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            predicted_marker.points.append(p)

        self.lane_center_pub.publish(predicted_marker)

    def convert_points_to_point_cloud2(self, points, header):
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        cloud_msg = pc2.create_cloud(header, fields, points)
        return cloud_msg


if __name__ == "__main__":
    try:
        LaneCandidatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

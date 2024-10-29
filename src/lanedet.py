#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
from math import atan2, sqrt, tan, fabs, degrees
from scipy.interpolate import splprep, splev
from lidar_lane_detector.msg import CentroidWithLabel, CentroidWithLabelArray

class LaneCandidatePublisher:
    def __init__(self):
        rospy.init_node('lane_candidate_publisher', anonymous=True)

        rospy.Subscriber("/filtered_lane", PointCloud2, self.point_cloud_callback)
        rospy.Subscriber("/centroid_info", CentroidWithLabelArray, self.centroid_callback)
        self.lane_marker_pub = rospy.Publisher("/lane_marker", Marker, queue_size=1)
        self.lane_center_pub = rospy.Publisher("/lane_center", Marker, queue_size=1)

        # Parameters
        self.max_distance_change = rospy.get_param("~max_distance_change", 1.0)
        self.max_angle_change = rospy.get_param("~max_angle_change", 50)
        self.total_length = 10  # Lane length
        self.shift_amount = rospy.get_param("~shift_amount", 1.0)  # Obstacle avoidance distance
        self.obstacle_size = rospy.get_param("~obstacle_size", 1.0)  # Obstacle size

        # Obstacle detection and lane slope tracking variables
        self.obstacle_detected = False
        self.obstacle_centroids = None
        self.fixed_adjusted_points = None  # Stores adjusted lane points when avoiding obstacles
        self.last_slope = None  # Stores the slope right before obstacle detection
        self.saved_lane_points = None  # Stores the fixed lane points for consistent coordinates

    def point_cloud_callback(self, msg):
        points = self.extract_points_and_intensities(msg)

        # Detect and smooth lane lines
        left_points, right_points = self.extract_road_line(points)
        left_points = self.smooth_line(left_points)
        right_points = self.smooth_line(right_points)

        # Generate the center lane line
        center_points = self.extract_center_line(left_points, right_points)

        # Update slope if no obstacle is detected
        if not self.obstacle_detected and len(center_points) > 1:
            self.last_slope = self.calculate_average_slope(center_points)

        # If obstacle detected, create a fixed predicted lane using last saved slope
        if self.obstacle_detected and self.last_slope is not None:
            predicted_center_points = self.predict_center_line(center_points, self.total_length, self.last_slope)

            # Only apply obstacle avoidance if there are predicted points
            if len(predicted_center_points) > 0:
                adjusted_center_points = predicted_center_points.copy()

                # Apply lane shifting for each obstacle centroid
                start_offset = 1  # How far ahead of the obstacle to start shifting
                for centroid in self.obstacle_centroids:
                    distances = np.linalg.norm(predicted_center_points - centroid, axis=1)
                    closest_index = np.argmin(distances)
                    
                    # Start shifting a few points ahead of closest obstacle point
                    start_index = max(0, closest_index - start_offset)

                    for i in range(start_index, closest_index + 1):
                        closest_point = predicted_center_points[i]

                        # Calculate lane direction vector
                        if i > 0:
                            prev_point = predicted_center_points[i - 1]
                        else:
                            prev_point = predicted_center_points[i + 1]

                        direction_vector = closest_point - prev_point
                        direction_unit = direction_vector / np.linalg.norm(direction_vector)
                        lateral_vector = np.array([-direction_unit[1], direction_unit[0], 0])

                        # Set shift direction based on obstacle position and apply shift
                        shift_direction = np.sign(np.dot(centroid[:2] - closest_point[:2], lateral_vector[:2]))
                        shift_vector = -shift_direction * self.obstacle_size * lateral_vector

                        # Shift the selected point
                        adjusted_center_points[i] += shift_vector

                # Save adjusted points to maintain fixed coordinates
                self.fixed_adjusted_points = adjusted_center_points
                self.saved_lane_points = self.fixed_adjusted_points.copy()

        # Publish fixed or predicted lane center points
        if self.saved_lane_points is not None:
            self.publish_lane_center(self.saved_lane_points, msg.header)
        else:
            # Default behavior without obstacles
            predicted_center_points = self.predict_center_line(center_points, self.total_length)
            self.publish_lane_center(predicted_center_points, msg.header)

        # Publish the lane markers for left and right lanes
        self.publish_lane_marker(left_points, right_points, msg.header)

    def centroid_callback(self, msg):
        # Set obstacle positions if detected
        if len(msg.centroids) >= 1:
            self.obstacle_detected = True
            self.obstacle_centroids = [np.array([c.centroid.x, c.centroid.y, c.centroid.z]) for c in msg.centroids]
        else:
            # Clear obstacles when not detected
            self.obstacle_centroids = []
            self.obstacle_detected = False
            self.fixed_adjusted_points = None
            self.saved_lane_points = None

    # Other helper methods (extract_points_and_intensities, extract_road_line, etc.) remain unchanged...


    def extract_points_and_intensities(self, cloud_msg):
        points = []
        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def extract_road_line(self, points):
        left_points = []
        right_points = []

        for point in points:
            if point[1] > 0:
                right_points.append(point)
            else:
                left_points.append(point)

        # x축 기준으로 정렬하여 차선이 도로 방향을 따라 정렬
        left_points = sorted(left_points, key=lambda p: p[0])
        right_points = sorted(right_points, key=lambda p: p[0])

        return left_points, right_points

    def smooth_line(self, points):
        points = np.array(points)
        if len(points) > 2:
            try:
                k = min(3, len(points) - 1)
                tck, u = splprep(points.T, s=10, k=k)
                smooth_points = splev(u, tck)
                points = np.vstack(smooth_points).T

                # 각도 변화 제한 적용 및 튀는 값 무시
                filtered_points = [points[0]]
                for i in range(1, len(points)):
                    prev_point = filtered_points[-1]
                    curr_point = points[i]
                    distance = sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
                    angle = degrees(atan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0]))

                    if fabs(angle) <= self.max_angle_change and distance <= self.max_distance_change:
                        filtered_points.append(curr_point)

                points = np.array(filtered_points)
            except ValueError as e:
                rospy.logwarn(f"Splprep failed with error: {e}. Skipping smoothing for this line.")
        return points

    def extract_center_line(self, left_points, right_points):
        center_points = []

        if len(left_points) > 3 and len(right_points) > 3:
            for lp, rp in zip(left_points, right_points):
                center_x = (lp[0] + rp[0]) / 2.0
                center_y = (lp[1] + rp[1]) / 2.0
                center_z = (lp[2] + rp[2]) / 2.0
                center_points.append([center_x, center_y, center_z])

        center_points = np.array(center_points)

        if len(center_points) > 2:
            try:
                k = min(3, len(center_points) - 1)
                tck, u = splprep(center_points.T, s=10, k=k)
                smooth_points = splev(u, tck)
                center_points = np.vstack(smooth_points).T

                filtered_points = [center_points[0]]
                for i in range(1, len(center_points)):
                    prev_point = filtered_points[-1]
                    curr_point = center_points[i]
                    distance_change = sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
                    angle_change = degrees(atan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0]))

                    if fabs(angle_change) <= self.max_angle_change and distance_change <= self.max_distance_change:
                        filtered_points.append(curr_point)
                center_points = np.array(filtered_points)
            except ValueError as e:
                rospy.logwarn(f"Splprep failed with error: {e}. Skipping smoothing for center line.")

        return center_points

    def calculate_average_slope(self, points):
        """ points의 평균 기울기 계산 """
        if len(points) < 2:
            return 0

        slopes = []
        for i in range(1, len(points)):
            prev_point = points[i - 1]
            curr_point = points[i]
            slope_angle = atan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0])
            slopes.append(slope_angle)

        return np.mean(slopes)

    def predict_center_line(self, center_points, total_length, slope=None):
        """ 주어진 slope로 차선을 예측. slope가 None일 경우 center_points의 평균 기울기 사용 """
        if len(center_points) < 2:
            return center_points

        accumulated_length = 0
        for i in range(1, len(center_points)):
            prev_point = center_points[i - 1]
            curr_point = center_points[i]
            segment_length = sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
            accumulated_length += segment_length

        remaining_length = total_length - accumulated_length
        if remaining_length <= 0:
            return center_points

        # slope가 주어지지 않았을 경우 center_points의 평균 기울기 계산
        if slope is None:
            slope = self.calculate_average_slope(center_points)

        last_point = center_points[-1]
        predicted_points = []

        while remaining_length > 0:
            delta_x = min(remaining_length, 1.0)
            delta_y = delta_x * tan(slope)
            new_point = [last_point[0] + delta_x, last_point[1] + delta_y, last_point[2]]
            predicted_points.append(new_point)
            last_point = new_point
            segment_length = sqrt(delta_x**2 + delta_y**2)
            remaining_length -= segment_length

        predicted_points = np.array(predicted_points)
        return np.vstack([center_points, predicted_points])

    def publish_lane_marker(self, left_points, right_points, header):
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
            left_marker.color.b = 1.0

            for point in left_points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                left_marker.points.append(p)

            self.lane_marker_pub.publish(left_marker)

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
            right_marker.color.r = 1.0

            for point in right_points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                right_marker.points.append(p)

            self.lane_marker_pub.publish(right_marker)

    def publish_lane_center(self, predicted_center_points, header):
        if len(predicted_center_points) < 2:
            return

        predicted_marker = Marker()
        predicted_marker.header = header
        predicted_marker.ns = "predicted_lane_candidates_center"
        predicted_marker.id = 3
        predicted_marker.type = Marker.LINE_STRIP
        predicted_marker.action = Marker.ADD
        predicted_marker.pose.orientation.w = 1.0
        predicted_marker.scale.x = 0.1
        predicted_marker.color.a = 1.0
        predicted_marker.color.g = 0.5

        for point in predicted_center_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            predicted_marker.points.append(p)

        self.lane_center_pub.publish(predicted_marker)

if __name__ == "__main__":
    try:
        LaneCandidatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

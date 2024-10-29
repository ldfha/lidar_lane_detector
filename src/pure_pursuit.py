#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import os
import numpy as np
import math  # 추가된 부분
from math import sqrt, pow, atan2, sin, cos
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from erp_driver.msg import erpCmdMsg, erpStatusMsg
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker

class PurePursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        # 퍼블리셔: 차량의 제어 명령(erpCmdMsg)
        self.erp_cmd_pub = rospy.Publisher('/erp42_ctrl_cmd', erpCmdMsg, queue_size=1)

        # 서브스크라이버: 로컬 경로 및 차량 상태 구독
        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber("/erp42_status", erpStatusMsg, self.steering_angle_callback)
        rospy.Subscriber("/imu/global_heading", Float32, self.angle_callback)

        # 전환 신호 구독 및 LIDAR 경로 구독
        self.use_lidar_lane_center = False  # LIDAR 경로 사용 여부
        rospy.Subscriber('/switch_to_lidar', Bool, self.switch_callback)
        rospy.Subscriber("/lane_path",Path, self.lidar_path_callback)
        self.lidar_path = None
        self.is_lidar_path = False

        # erpCmdMsg 메시지 객체 생성
        self.ctrl_cmd_msg = erpCmdMsg()

        # 상태 변수
        self.is_path = False
        self.is_ego = False
        self.is_heading = False
        self.forward_point = Point()
        self.current_position = Point()
        self.is_look_forward_point = False
        self.vehicle_length = 1.63
        self.current_steering_angle = 0
        self.lfd = 3.0
        self.vehicle_yaw = 0.0
        self.current_speed = 0
        self.lat_err = []

        self.PID_steer = PID()
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.Dterm = 0.0


        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            if self.is_ego and self.is_heading:
                self.pure_pursuit_control()  # 제어 수행
            else:
                if not self.is_ego:
                    print("[2] Can't subscribe to '/erp42_status' topic...")
                if not self.is_heading:
                    print("[3] Can't subscribe to '/imu/global_heading' topic...")

            rate.sleep()

    def switch_callback(self, msg):
        self.use_lidar_lane_center = msg.data

        if self.use_lidar_lane_center:
            rospy.loginfo("Switched to LIDAR lane center control.")
            print("Switched to LIDAR lane center control.")
        else:
            rospy.loginfo("Switched to global path control.")
            print("Switched to global path control.")

    def lidar_path_callback(self, msg):
        # LIDAR 경로를 Marker 메시지에서 Path로 변환
        path_msg = Path()
        path_msg.header = msg.header

        for point in msg.points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = point.z
            path_msg.poses.append(pose)

        self.lidar_path = path_msg
        self.is_lidar_path = True

    def pure_pursuit_control(self):
        # LIDAR 경로로 스위칭된 경우 무조건 LIDAR 경로를 사용
        if self.use_lidar_lane_center and self.is_lidar_path:
            current_path = self.lidar_path
            print("Using LIDAR lane center path")
        elif self.is_path and not self.use_lidar_lane_center:  # LIDAR 경로로 스위칭되지 않은 경우에만 로컬 경로 사용
            current_path = self.path
            print("Using local path")
        else:
            print("No path available")
            return

        self.is_look_forward_point = False

        # Path 메시지에서 차선의 포인트를 가져와 처리
        for pose in current_path.poses:
            path_point = pose.pose.position
            dis = sqrt(pow(path_point.x, 2) + pow(path_point.y, 2))
            if dis >= self.lfd:
                self.forward_point = path_point
                self.is_look_forward_point = True
                break

        if self.is_look_forward_point:
            theta = atan2(self.forward_point.y, self.forward_point.x)
            steering_angle = atan2((2 * self.vehicle_length * sin(theta)), self.lfd)

            steering_angle_deg = np.degrees(steering_angle)
            steering_angle_deg = int(max(min(steering_angle_deg, 30), -30))

            delta_err = steering_angle_deg - self.current_steering_angle
            steer_cmd = self.PID_steer.control(delta_err)
            steer_cmd = int(max(min(steer_cmd, 30), -30))

            # LIDAR 경로를 사용할 때는 조향각을 반대로 적용
            if self.use_lidar_lane_center:
                steering_angle_cmd = -int((0.8*steering_angle_deg / 30.0) * 1800)
            else:
                steering_angle_deg = steer_cmd  # LIDAR 경로를 사용하지 않을 때 PID 제어 조향 사용.
                steering_angle_cmd = int((steering_angle_deg/ 30.0) * 1800)
            
            # # look_forward_point 각도에 따라 속도 가변 (세부 값 조정 필요)
            # k = abs(2*sin(theta)/self.lfd)
            
            # if k < 0.3:
            #     speed = 120
            # elif k < 0.5:
            #     speed = 90
            # elif k < 0.6:
            #     speed = 70
            # else:
            #     steering_angle_cmd = int((steering_angle_deg/ 30.0) * 2000)
            #     speed = 40
                
            self.lat_err.append(abs(self.forward_point.y))
            avg_lat_err = sum(self.lat_err) / len(self.lat_err)

            speed = 80

            # 프린트 정보
            print("-------------------------------------")
            print(f"Using {'LIDAR' if self.use_lidar_lane_center else 'local'} path")
            print(f"Forward Point: x={self.forward_point.x}, y={self.forward_point.y}")
            print(f"theta (rad) = {theta}, theta (deg) = {np.degrees(theta)}")
            #print(f"Curvature = {k}")
            print(f"Steering Angle Command = {steering_angle_cmd}")
            print(f"Steering Angle (deg) = {steering_angle_deg}")
            print(f"Speed (kph) = {speed}")
            print(f"Average Lateral offset = {avg_lat_err}")
            print("-------------------------------------")
            

            self.ctrl_cmd_msg.steer = steering_angle_cmd
            self.ctrl_cmd_msg.speed = speed
            self.ctrl_cmd_msg.gear = 0
            self.ctrl_cmd_msg.e_stop = False
            self.ctrl_cmd_msg.brake = 0

        else:
            print("No forward point found.")
            self.ctrl_cmd_msg.steer = 0
            self.ctrl_cmd_msg.speed = 0
            self.ctrl_cmd_msg.brake = 1

        self.erp_cmd_pub.publish(self.ctrl_cmd_msg)

    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

    def angle_callback(self, msg):
        raw_heading_deg = msg.data
        vehicle_heading_deg = raw_heading_deg % 360
        self.vehicle_yaw = math.radians(vehicle_heading_deg)
        self.is_heading = True

    def steering_angle_callback(self, msg):
        # ERP42 상태 메시지에서 속도 및 조향각을 업데이트
        self.current_speed = msg.speed
        self.current_steering_angle = int((msg.steer / 2000.0) * 30)
        self.is_ego = True

class PID:
    def __init__(self):
        self.kp = 1.3
        self.ki = 0.15
        self.kd = 0.0
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.Dterm = 0.0
        self.prev_error = 0.0
        self.dt = 0.03

    def control(self, error):
        self.Pterm = self.kp * error
        if error < 15:
            self.Iterm += error * self.dt
        self.Dterm = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        output = self.Pterm + self.ki * self.Iterm + self.Dterm
        print(f"PTerm:{self.Pterm}, ITerm:{self.ki*self.Iterm}, DTerm{self.Dterm}")
        return output

if __name__ == '__main__':
    try:
        pure_pursuit = PurePursuit()
    except rospy.ROSInterruptException:
        pass
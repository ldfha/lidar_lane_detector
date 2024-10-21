#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, os
import numpy as np
import math
from math import sqrt, pow, atan2, sin, cos
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker  # Import Marker message type
from erp_driver.msg import erpCmdMsg, erpStatusMsg  # erpCmdMsg and erpStatusMsg message types
from std_msgs.msg import Float32  # Float32 message type import

class PurePursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        # Publisher: Vehicle control command (erpCmdMsg)
        self.erp_cmd_pub = rospy.Publisher('/erp42_ctrl_cmd', erpCmdMsg, queue_size=1)

        # Subscribers: Subscribe to lane center marker and vehicle status
        rospy.Subscriber("/lane_center", Marker, self.lane_center_callback)
        rospy.Subscriber("/erp42_status", erpStatusMsg, self.steering_angle_callback)  # Subscribe to current status
        rospy.Subscriber("/magnetometer_heading_angle2", Float32, self.angle_callback)  # Subscribe to Magnetometer heading angle

        # erpCmdMsg message object
        self.ctrl_cmd_msg = erpCmdMsg()

        # State variables
        self.is_lane_center = False
        self.is_ego = False
        self.is_heading = False  # Heading value reception status
        self.forward_point = Point()
        self.current_position = Point()  # Initialize current_position
        self.is_look_forward_point = False
        self.vehicle_length = 1.63
        self.lfd = 1.5  # Lookahead distance
        self.vehicle_yaw = 0.0  # Initial yaw value (in radians)

        rate = rospy.Rate(15)  # 15Hz
        while not rospy.is_shutdown():
            if self.is_lane_center and self.is_ego and self.is_heading:
                self.pure_pursuit_control()  # Perform control
            else:
                os.system('clear')
                if not self.is_lane_center:
                    print("[1] Can't subscribe to '/lane_center' topic...")
                if not self.is_ego:
                    print("[2] Can't subscribe to '/erp42_status' topic...")
                if not self.is_heading:
                    print("[3] Can't subscribe to '/magnetometer_heading_angle2' topic...")

            rate.sleep()

    def pure_pursuit_control(self):
        self.is_look_forward_point = False

        # Iterate over the points in the lane_center marker
        for num, point in enumerate(self.lane_center.points):
            # Calculate distance between the point and the vehicle (assuming vehicle is at (0,0))
            dis = sqrt(pow(point.x, 2) + pow(point.y, 2))
            if dis >= self.lfd:
                self.forward_point = point
                self.is_look_forward_point = True
                break

        if self.is_look_forward_point:
            # Pure Pursuit algorithm: Calculate steering angle
            theta = atan2(self.forward_point.y, self.forward_point.x)

            # Calculate steering value
            steering_angle = atan2((2 * self.vehicle_length * sin(theta)), self.lfd)

            # Convert angle to degrees
            steering_angle_deg = np.degrees(steering_angle)

            # Limit the steering angle between -30 to 30 degrees
            steering_angle_deg = max(min(steering_angle_deg, 30), -30)

            # Convert to ERP42 steering command range (-2000 to 2000)
            steering_angle_cmd = int((steering_angle_deg / 30.0) * 2000)

            # Set the control command fields
            self.ctrl_cmd_msg.steer = -steering_angle_cmd  # Set steering command
            self.ctrl_cmd_msg.speed = 40  # Set desired speed
            self.ctrl_cmd_msg.gear = 0  # Gear setting (0: neutral)
            self.ctrl_cmd_msg.e_stop = False  # Emergency stop setting (False: release)
            self.ctrl_cmd_msg.brake = 0  # Brake setting

            # Debug print
            os.system('clear')
            print("-------------------------------------")
            print(f"Forward Point: x={self.forward_point.x}, y={self.forward_point.y}")
            print(f"theta (rad) = {theta}, theta (deg) = {np.degrees(theta)}")
            print(f"Steering Angle (rad) = {steering_angle}, Steering Angle (deg) = {steering_angle_deg}")
            print(f"Steering Angle Command = {steering_angle_cmd}")
            print(f"Speed (kph) = {self.ctrl_cmd_msg.speed}")
            print("-------------------------------------")

        else:
            print("No forward point found.")
            self.ctrl_cmd_msg.steer = 0  # Set steering angle to 0
            self.ctrl_cmd_msg.speed = 0  # Set speed to 0
            self.ctrl_cmd_msg.brake = 1  # Apply brake

        # Publish control command
        self.erp_cmd_pub.publish(self.ctrl_cmd_msg)

    def lane_center_callback(self, msg):
        # Update the lane center marker points
        self.lane_center = msg
        self.is_lane_center = True

    def angle_callback(self, msg):
        # Receive heading value from Magnetometer (assuming in degrees)
        raw_heading_deg = msg.data  # Heading value from sensor (in degrees)

        # Apply correction if necessary (using as is)
        vehicle_heading_deg = raw_heading_deg

        # Normalize angle range (0° ~ 360°)
        vehicle_heading_deg = vehicle_heading_deg % 360

        # Convert to radians
        self.vehicle_yaw = math.radians(vehicle_heading_deg)
        self.is_heading = True

    def steering_angle_callback(self, msg):
        # Subscribe to current ERP42 status
        self.current_position.x = msg.speed  # Use current speed information
        self.is_ego = True


if __name__ == '__main__':
    try:
        pure_pursuit = PurePursuit()
    except rospy.ROSInterruptException:
        pass

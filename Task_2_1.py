#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from tf.transformations import euler_from_quaternion
import math

class StaticController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('static_controller', anonymous=True)
        
        # Controller parameters
        self.lookahead_distance = 1.0  # Look-ahead distance for pure pursuit
        self.max_linear_speed = 1.0    # Maximum linear speed
        self.max_angular_speed = 1.0   # Maximum angular speed
        self.goal_threshold = 0.5      # Distance threshold to consider waypoint reached
        
        # Current state variables
        self.current_pose = None       # Current vehicle pose
        self.current_x = 0.0           # Current x position
        self.current_y = 0.0           # Current y position
        self.current_yaw = 0.0         # Current yaw angle
        self.waypoints = []            # List of waypoints to follow
        self.current_waypoint_index = 0  # Index of the current target waypoint
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path_sub = rospy.Subscriber('/path', Path, self.path_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Static controller initialized")

    def path_callback(self, msg):
        """Process the path message and extract waypoints"""
        self.waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.waypoints.append((x, y))
        
        rospy.loginfo(f"Received {len(self.waypoints)} waypoints")
        self.current_waypoint_index = 0

    def odom_callback(self, msg):
        """Process the odometry message to get current vehicle state"""
        self.current_pose = msg.pose.pose
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_yaw = euler_from_quaternion(orientation_list)

    def get_target_point(self):
        """Find the target point along the path at lookahead distance"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return None
        
        # Start from current waypoint
        target_point = None
        
        # Check if we've reached the current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]
        distance_to_current = self.distance(self.current_x, self.current_y, 
                                           current_waypoint[0], current_waypoint[1])
        
        if distance_to_current < self.goal_threshold:
            # Move to next waypoint
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                return None  # End of path
            current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Use the current waypoint as the target
        target_point = current_waypoint
        
        # Check if we need to look ahead further
        if distance_to_current < self.lookahead_distance and self.current_waypoint_index < len(self.waypoints) - 1:
            next_waypoint = self.waypoints[self.current_waypoint_index + 1]
            target_point = self.interpolate_point(current_waypoint, next_waypoint, 
                                                 self.lookahead_distance - distance_to_current)
        
        return target_point

    def interpolate_point(self, p1, p2, distance):
        """Interpolate a point along the line from p1 to p2 at the given distance from p1"""
        x1, y1 = p1
        x2, y2 = p2
        
        # Calculate the direction vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate the distance between p1 and p2
        segment_length = math.sqrt(dx**2 + dy**2)
        
        if segment_length < 0.001:
            return p1  # Points are too close together
        
        # Normalize the direction vector
        dx /= segment_length
        dy /= segment_length
        
        # Calculate the interpolated point
        x = x1 + dx * min(distance, segment_length)
        y = y1 + dy * min(distance, segment_length)
        
        return (x, y)

    def distance(self, x1, y1, x2, y2):
        """Calculate the Euclidean distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def control_loop(self, event):
        """Main control loop that calculates and publishes control commands"""
        if not self.current_pose or not self.waypoints:
            # Not ready yet
            return
        
        # Get the target point
        target_point = self.get_target_point()
        if not target_point:
            # End of path or no valid target
            self.stop_vehicle()
            return
        
        # Calculate the steering angle using pure pursuit
        target_x, target_y = target_point
        
        # Calculate the angle to the target point in the vehicle's frame
        target_angle = math.atan2(target_y - self.current_y, target_x - self.current_x)
        
        # Calculate the steering angle
        steering_angle = self.normalize_angle(target_angle - self.current_yaw)
        
        # Calculate the distance to the target
        distance_to_target = self.distance(self.current_x, self.current_y, target_x, target_y)
        
        # Calculate linear and angular velocities
        linear_velocity = min(self.max_linear_speed, distance_to_target)
        angular_velocity = min(self.max_angular_speed, 
                              max(-self.max_angular_speed, steering_angle))
        
        # Create and publish the command
        twist_cmd = Twist()
        twist_cmd.linear.x = linear_velocity
        twist_cmd.angular.z = angular_velocity
        self.cmd_vel_pub.publish(twist_cmd)

    def normalize_angle(self, angle):
        """Normalize an angle to be between -pi and pi"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stop_vehicle(self):
        """Stop the vehicle"""
        twist_cmd = Twist()
        twist_cmd.linear.x = 0.0
        twist_cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_cmd)

if __name__ == '__main__':
    try:
        controller = StaticController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

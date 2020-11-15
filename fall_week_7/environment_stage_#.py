#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def getGoalDistace(self):
        # function called when reset or goal achieved
        # meaning goal_dist only calculated for the beginning of episode
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        # Odometry : using data from sensors to calculate the robot's change in position over time
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation

        # not sure what orientation.w is
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]

        # yaw means rotation about the z-axis : rotating right or left
        # euler_from_quarternion : change from [x,y,z,w] to [roll, pitch, yaw]
        _, _, yaw = euler_from_quaternion(orientation_list)

        # which direction the robot should rotate to in order to reach the goal
        # somewhat like calculation of arctan with some conditional values
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        # is it going in the right direction?
        heading = goal_angle - yaw

        # too much off to the right?
        if heading > pi:
            # correction seems way~~~~ to big
            # is there going to be some adjustments?
            # OR
            # it could be that the robot only can do heading : within range of pi and -pi
            # so correcting the numbers just incase
            heading -= 2 * pi

        # too much off to the left
        elif heading < -pi:
            heading += 2 * pi

        # yeah but lets keep the floating point simple
        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading

        # proximity to obstacle for the bot to collide
        min_range = 0.13
        # done means collision
        done = False

        # scanned by robot
        for i in range(len(scan.ranges)):
            # nothing within range
            if scan.ranges[i] == float('Inf'):
                # nothing within range 3.5
                scan_range.append(3.5)

            # so close that dist = NAN
            # this would mean not just collision but also fusion, combining of the bot and the obstacle
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)

            # otherwise just use the scan results
            else:
                scan_range.append(scan.ranges[i])

        # out of the scanned distances, which one shows how close the bot is to an obstacle?
        obstacle_min_range = round(min(scan_range), 2)

        # and in which direction?
        # for direction, just track which laser gave the min result by argmin
        obstacle_angle = np.argmin(scan_range)

        # yeah so condition for collision
        # is min(scan_range) > 0: necessary? debugging efforts for scan_range of negative or 0 which shouldn't occur?
        if min_range > min(scan_range) > 0:
            done = True

        # so current distance would mean how much further to go for the GOAL
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)

        # if close enough, let it be a GOAL
        if current_distance < 0.2:
            self.get_goalbox = True

        # concatenation with scan_range and heading, ... , obstanle_angle to return as STATE
        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

    def setReward(self, state, done, action):
        yaw_reward = []

        # state = [scan_range ...., heading, current_distance, obstacle_min_range, obstacle_angle]
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        # five times for five actions
        # ACTION
        # linear velocity is always 0.15m/s
        # angular velocity = { 0 : -1.5rad/s, 1 : -0.75rad/s, 2 : 0rad/s, 3 : 0.75rad/s, 4 : 1.5rad/s}
        for i in range(5):
            # angle
            # 0 : heading + pi/4
            # 1 : heading + 3pi/8
            # 2 : heading + pi/2
            # 3 : heading + 5pi/8
            # 3 : heading + 3pi/4
            # note heading is within [-pi, pi]
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2

            # fabs = floation absolute value
            # modf = tuple of fraction and integer part

            # hmm basically seems to be correcting the equation for angle,
            # so remaining would be something like
            # tr = 1 - 4 * (heading + (pi / 8 * i) corrected to the appropriate units)
            # so it would result as a table that shouls
            # action (rotation) and how the heading is in the right direction
            # totally on track : tr = 1,
            # off track : tr < 1 or even tr < 0
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        # the closer to GOAL, to smaller this value
        # exponentially
        distance_rate = 2 ** (current_distance / self.goal_distance)

        # if too close to obstacle, negative reward
        if obstacle_min_range < 0.5:
            ob_reward = -5
        else:
            ob_reward = 0

        # so reward for each action would be returned as,
        # on track? how far? safe distance from obstacles?
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward

        # or just extreme values in the case of collision
        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())
            ### no need to do anything after collision?
            ### get state always resets done = False, but is that enough to prevent getting stuck?
            ### such implementation could be in dqn.py

        # and goal
        # note that immediately after goal,
        # it respawns, goal dist calculated, and goal = false which means goal box should be found again
        ### is this enough for reset?
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward


    def step(self, action):
        max_angular_vel = 1.5
        # action_size defined in dqn.py
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        # fixed linear vel as commented above
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                ### if my laptop is too slow, would increasing the timeout prevent errors?
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        # with condition, goal is resetted
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)

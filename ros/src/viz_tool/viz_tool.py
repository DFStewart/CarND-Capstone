#!/usr/bin/env python

import cv2
import pygame
import rospy
import math

import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TwistStamped

from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd
from styx_msgs.msg import Lane, TrafficLightArray
from light_msgs.msg import UpcomingLight

BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
GREY    = (128, 128, 128)
GREEN   = (0, 255, 0)
RED     = (255, 0, 0)
YELLOW  = (255, 255, 0)

SCALING = 20. # Scale the size of the position to fill the window

class VizTool(object):
    bridge               = CvBridge()

    def __init__(self):
        self.base_waypoints       = None
        self.final_waypoints      = None
        self.steering_cmd         = None
        self.brake_cmd            = None
        self.throttle_cmd         = None

        self.screen               = None
        self.win_dim              = None
        self.screen_dim           = None

        self.upcoming_light_state = None
        self.twist_cmd            = None
        self.track_image          = None

        self.dashboard_img        = None
        self.cv_image_smaller     = None
        self.image                = None

        rospy.init_node('viz_tool')
        self.win_dim     = (1000, 1000)
        self.screen_dim = (self.win_dim[0], self.win_dim[1])
        rospy.Subscriber('/current_pose', PoseStamped, self.cb_current_pose)
        self.base_waypoints_sub  = rospy.Subscriber('/base_waypoints', Lane, self.cb_base_waypoints)
        rospy.Subscriber('/final_waypoints', Lane, self.cb_final_waypoints)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.cb_twist_cmd)
        rospy.Subscriber('/current_velocity', TwistStamped, self.cb_curr_vel)
        rospy.Subscriber('/image_color', Image, self.cb_image)
        rospy.Subscriber('/vehicle/steering_cmd',SteeringCmd, self.cb_steering_cmd)
        rospy.Subscriber('/vehicle/throttle_cmd',ThrottleCmd, self.cb_throttle_cmd)
        rospy.Subscriber('/vehicle/brake_cmd',BrakeCmd, self.cb_brake_cmd)
        rospy.Subscriber('/vehicle/dbw_enabled',Bool, self.cb_dwb_enabled)
        rospy.Subscriber('/upcoming_light',UpcomingLight,self.cb_upcoming_lt)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.cb_truth_trafficlight) 
        
        self.initialize()
        self.main_loop()

    def initialize(self):
        self.screen = pygame.display.set_mode(self.win_dim, pygame.DOUBLEBUF)
        self.screen.fill(BLACK)

    def update_screen(self):
        if self.dashboard_img is not None:
            if self.cv_image_smaller is not None:
                self.dashboard_img[0:self.cv_image_smaller.shape[0],0:self.cv_image_smaller.shape[1]] = self.cv_image_smaller
            self.dashboard_img = pygame.image.fromstring(cv2.resize(self.dashboard_img, self.win_dim).tobytes(), self.win_dim, 'RGB')
            self.screen.blit(self.dashboard_img, (0, 0))
            pygame.display.flip()

    def draw_track(self):
        basewpts_x = []
        basewpts_y = []
        for wp in self.base_waypoints:
            tempx,tempy = self.scale_points(wp.pose.pose.position.x,wp.pose.pose.position.y)
            basewpts_x.append(tempx)
            basewpts_y.append(tempy)

        vertices = [np.column_stack((basewpts_x, basewpts_y)).astype(np.int32)]
        self.track_image = np.empty((self.screen_dim[0], self.screen_dim[1], 3), dtype=np.uint8)
        cv2.polylines(self.track_image, vertices, True, WHITE, 1)

    def draw_current_position(self):
        if self.current_pose is not None:
            x,y = self.scale_points(self.current_pose.pose.position.x,self.current_pose.pose.position.y)
            cv2.circle(self.dashboard_img, (x, y), 10, GREEN, -1)
    
    def draw_image(self):
        if self.image is not None:
            self.cv_image = self.bridge.imgmsg_to_cv2(self.image, "rgb8")
            self.cv_image_smaller = cv2.resize(self.cv_image,(0,0),fx=0.5,fy=0.5)

    def draw_final_waypoints(self):
        if self.final_waypoints is not None and self.dashboard_img is not None:
            xs = []
            ys = []
            for wp in self.final_waypoints:
                tempx,tempy = self.scale_points(wp.pose.pose.position.x,wp.pose.pose.position.y)
                xs.append(tempx)
                ys.append(tempy)
            vertices = [np.column_stack((xs, ys)).astype(np.int32)]
            cv2.polylines(self.dashboard_img, vertices, False, RED, 8)

    def write_text(self, text, offset_left=50, offset_top=15, fontsize=1, thickness=1, color=WHITE):
        cv2.putText(self.dashboard_img, text, (offset_left, offset_top), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontsize,color, thickness)
        return offset_top

    def text_control_output(self):
        BASE_TOP_OFFSET = 850

        if self.throttle_cmd is not None and self.brake_cmd is not None:
            throttle = self.throttle_cmd.pedal_cmd
            brake    = self.brake_cmd.pedal_cmd
            steer    = -1. * 180. / math.pi * self.steering_cmd.steering_wheel_angle_cmd
            self.write_text("Thr(%):{0:.2f}".format(throttle),offset_left=25, offset_top=BASE_TOP_OFFSET)
            self.write_text("Brk(Nm):{0:.2f}".format(brake),offset_left=25, offset_top=BASE_TOP_OFFSET+30)
            self.write_text("Str(deg):{0:.2f}".format(steer),offset_left=25, offset_top=BASE_TOP_OFFSET+60)

        if self.twist_cmd is not None:
            cmd_linear_velocity   = self.twist_cmd.linear.x
            cmd_angular_velocity  = self.twist_cmd.angular.z
            self.write_text("Cmd Lin Vel:{0:.2f}".format(cmd_linear_velocity),  offset_left=25, offset_top=BASE_TOP_OFFSET+90)
            self.write_text("Cmd Ang Vel:{0:.2f}".format(cmd_angular_velocity), offset_left=25, offset_top=BASE_TOP_OFFSET+120)

    def text_nav_output(self):
        curr_pose_x = 0.0
        curr_pose_y = 0.0
        curr_vel    = 0.0

        BASE_TOP_OFFSET = 850

        if self.current_pose is not None:
            curr_pose_x = self.current_pose.pose.position.x
            curr_pose_y = self.current_pose.pose.position.y
        
        if self.curr_vel is not None:
            curr_lin_vel = self.curr_vel.linear.x
            curr_ang_vel = self.curr_vel.angular.z
        
        if self.dwb_enabled is not None:
            dwb_enabled = self.dwb_enabled.data

        self.write_text("CurrPose X:{0:.1f}".format(curr_pose_x), offset_left=260, offset_top=BASE_TOP_OFFSET)
        self.write_text("CurrPose Y:{0:.1f}".format(curr_pose_y), offset_left=260, offset_top=BASE_TOP_OFFSET+30)
        self.write_text("CurrLinVel:{0:.1f}".format(curr_lin_vel),offset_left=260, offset_top=BASE_TOP_OFFSET+60)
        self.write_text("CurrAngVel:{0:.1f}".format(curr_ang_vel),offset_left=260, offset_top=BASE_TOP_OFFSET+90)

        if dwb_enabled:
            self.write_text("DBW ENABLED", offset_left=260, offset_top=BASE_TOP_OFFSET+120, color = GREEN)
        else:
            self.write_text("DBW DISABLED",offset_left=260, offset_top=BASE_TOP_OFFSET+120, color = RED)
    
    def text_light_output(self):
        BASE_TOP_OFFSET = 850

        if self.veh_trafficlight is not None:
            stop_x = self.veh_trafficlight.lights[0].pose.pose.position.x
            stop_y = self.veh_trafficlight.lights[0].pose.pose.position.y
            self.write_text("STOPLINE X:{0:.1f}".format(stop_x), offset_left=500, offset_top=BASE_TOP_OFFSET, color = WHITE)
            self.write_text("STOPLINE Y:{0:.1f}".format(stop_y), offset_left=500, offset_top=BASE_TOP_OFFSET+30, color = WHITE)
            
            if self.current_pose is not None:
                curr_pose_x = self.current_pose.pose.position.x
                curr_pose_y = self.current_pose.pose.position.y
                dist2light = self.distance(curr_pose_x,curr_pose_y,stop_x,stop_y)
                self.write_text("DIST 2 STOPLINE:{0:.1f}".format(dist2light), offset_left=500, offset_top=BASE_TOP_OFFSET+60, color = WHITE)

        if self.upcoming_light_state is not None:
            state = self.upcoming_light_state
            if state == 0:
                self.write_text("LIGHT STATE 0: RED", offset_left=500, offset_top=BASE_TOP_OFFSET+90, color = RED)
            elif state == 1:
                self.write_text("LIGHT STATE 1: YELLOW", offset_left=500, offset_top=BASE_TOP_OFFSET+90, color = YELLOW)
            elif state == 2:
                self.write_text("LIGHT STATE 2: GREEN", offset_left=500, offset_top=BASE_TOP_OFFSET+90, color = GREEN)
            elif state == 4:
                self.write_text("LIGHT STATE 4: UNKNOWN", offset_left=500, offset_top=BASE_TOP_OFFSET+90, color = GREY)
            else:
                self.write_text("DETECTOR NOT RUNNING", offset_left=500, offset_top=BASE_TOP_OFFSET+90, color = GREY)
        else:
                self.write_text("DETECTOR NOT RUNNING", offset_left=500, offset_top=BASE_TOP_OFFSET+90, color = WHITE)

    def scale_points(self,x_point,y_point):
        SCALING_FACTOR = 0.3
        x = int(x_point*SCALING + self.screen_dim[0]*SCALING_FACTOR)
        y = int(self.screen_dim[1] - (y_point*SCALING  + self.screen_dim[1]*SCALING_FACTOR))
        return x,y
    
    def distance(self,x1,y1,x2,y2):
        dist = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
        return dist

    def main_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.base_waypoints is not None:
                pygame.event.pump()
                self.dashboard_img = np.copy(self.track_image)
                self.draw_current_position()
                self.text_control_output()
                self.text_nav_output()
                self.text_light_output()
                self.draw_image()
                self.draw_final_waypoints()
                self.update_screen()
            rate.sleep()
        self.close()

    def cb_current_pose(self, msg):
        self.current_pose = msg

    def cb_base_waypoints(self, msg):
        self.base_waypoints = msg.waypoints
        self.draw_track()
        self.base_waypoints_sub.unregister()

    def cb_final_waypoints(self, lane):
        self.final_waypoints = lane.waypoints

    def cb_steering_cmd(self, msg):
        self.steering_cmd = msg

    def cb_throttle_cmd(self, msg):
        self.throttle_cmd = msg

    def cb_brake_cmd(self, msg):
        self.brake_cmd = msg

    def cb_dwb_enabled(self, msg):
        self.dwb_enabled = msg

    def cb_upcoming_lt(self,msg):
        self.upcoming_light_state = msg.state
    
    def cb_truth_trafficlight(self,msg):
        self.veh_trafficlight = msg

    def cb_image(self,msg):
        self.image = msg
    
    def cb_twist_cmd(self,msg):
        self.twist_cmd = msg.twist

    def cb_curr_vel(self,msg):
        self.curr_vel = msg.twist

    @staticmethod
    def close():
        pygame.quit()

if __name__ == '__main__':
    VizTool()

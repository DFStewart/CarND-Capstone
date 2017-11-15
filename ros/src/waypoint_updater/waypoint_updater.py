#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray
import tf
import math
from light_msgs.msg import UpcomingLight
from std_msgs.msg import Int32
from geometry_msgs.msg import TwistStamped

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

# TUNABLE PARAMETERS
LOOKAHEAD_WPS = 30       # Number of waypoints published
SPEED_MPH     = 10.      # Forward speed in miles per hour
STOP_DIST_ERR = 20.      # Distance to start applying brakes ahead of STOP_LINE
LL_WPT_SEARCH = 3        # Number of waypoints behind to search over to detect new waypoint ahead from previous nearest waypoint
UL_WPT_SEARCH = 20       # Number of waypoints ahead to search over to detect new waypoint ahead from previous nearest waypoint

# CONSTANTS
MPH2MPS       = 0.44704  # Conversion miles per hour to meters per second

# DEBUG FLAGS
DEBUG         = False    # True = Print Statements appear in Terminal with Debug info
DEBUG_TOPICS  = True     # Enable debug output topics
SIMULATE_TL   = False    # True = Simulate traffic light positions with /vehicle/traffic_lights, False = use tl_detector /upcoming_light topic
DEBUG_TLDET   = False    # True = Print TL_DETECTOR topic debug information
DEBUG_TLSIM   = False    # True = Print traffic light information from simulator data debug information
DEBUG_TGTSPD  = True     # True = Print debug information for target velocity calculations
DEBUG_SEARCH  = False    # True = Pring debug information on searching the base_waypoints for our current position

class WaypointUpdater(object):
    def __init__(self):
    
        #Initialize node
        rospy.init_node('waypoint_updater')

        # Setup Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.CurrVel_cb,queue_size=1)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        
        # If tl_detector node is active get traffic light data from /upcoming_light topic, otherwise use the topic from the sim /vehicle/traffic_lights
        if SIMULATE_TL:
            rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.sim_traffic_cb, queue_size=1) 
        else:
            #rospy.Subscriber('/upcoming_light',UpcomingLight,self.upcoming_lt_cb,queue_size=1)
            rospy.Subscriber('/traffic_waypoint',Int32,self.traffic_cb)
        
        # Setup Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        # Topics to publish only if debugging
        if DEBUG_TOPICS:
            self.debug_currentpos_pub= rospy.Publisher('debug_current_pose', PoseStamped, queue_size=1)

        # Initialize Member Variables
        self.pos_x            = 0.
        self.pos_y            = 0.
        self.pos_z            = 0.
        self.light_ahead_idx  = -1.
        self.waypoints        = None
        self.wpt_ahead_idx    = None       
        self.wpt_ahead        = None
        self.final_wpts       = None
        self.current_orient   = None
        self.light_ahead      = None
        self.search_range     = None
        self.target_speed_mps = SPEED_MPH*MPH2MPS
        self.base_wpt_spd_mps = SPEED_MPH*MPH2MPS
        self.current_velocity = SPEED_MPH*MPH2MPS
        self.dist2light_m     = 9999.
        self.prev_wpt_ahead_idx = 0.
        rospy.spin()
        
    def loop(self): 
        if self.waypoints != None:
            
            # Find waypoint directly ahead of us and assign it to self.wpt_ahead
            self.find_next_waypoint()
            
            # Initialize the final waypoints ahead of us
            self.final_wpts = Lane()
            
            if False:
                print('WPT Ahead        ',"x: ",self.wpt_ahead.pose.pose.position.x,"y: ",self.wpt_ahead.pose.pose.position.y,"idx: ",self.wpt_ahead_idx)
                print('WPT List Length: ', len(self.waypoints)) 
                            
            # Form final waypoint list starting from the waypoint directly ahead
            final_wpt_idx = self.wpt_ahead_idx+LOOKAHEAD_WPS
            if final_wpt_idx < len(self.waypoints): # protect against wrapping around the waypoints array
                self.final_wpts.waypoints = self.waypoints[self.wpt_ahead_idx:final_wpt_idx]
            else:
                # Handle the case when the base waypoint indicies wrap back to zero
                self.final_wpts.waypoints = self.waypoints[self.wpt_ahead_idx:len(self.waypoints)]
                idx_prev_0 = len(self.waypoints) - self.wpt_ahead_idx
                idx_past_0 = LOOKAHEAD_WPS - idx_prev_0
                for idx in range(idx_past_0):
                    self.final_wpts.waypoints.append(self.waypoints[idx])

            # Find the index of the stopline in the final waypoints list           
            self.num_wpts_to_stopline = self.light_ahead_idx - self.wpt_ahead_idx
            if self.num_wpts_to_stopline < 0:
                self.num_wpts_to_stopline = 30.
            if self.num_wpts_to_stopline == 0:
                self.num_wpts_to_stopline = 1.
            if self.num_wpts_to_stopline > 30:
                self.num_wpts_to_stopline = 30.
            idx = 0
            fwpt_idx = 0            
            wpt_ahead    = self.final_wpts.waypoints[0]
            if self.light_ahead_idx >= 0:
                wpt_stopline = self.waypoints[self.light_ahead_idx]
                dis_wptahead2stopline = math.sqrt((wpt_ahead.pose.pose.position.x - wpt_stopline.pose.pose.position.x)**2 + (wpt_ahead.pose.pose.position.y-wpt_stopline.pose.pose.position.y)**2)
            else:
                dis_wptahead2stopline = 999.

            self.target_speed_mps = self.current_velocity
            vel_plan = []
            dist_plan= []
            for idx,wpt in enumerate(self.final_wpts.waypoints):           
                # Two Checks before we set deceleration profile:
                # 1) Check if we are NOT past the white stop line in the middle of the intersection => dont stop in middle
                # 2) Check if we close enough to the stop line to start braking
                # If either are false then just continue at normal speed
                #in_intersection = dis_wptahead2stopline < 3. # arbitrary tuned number
                start_braking   = math.fabs(dis_wptahead2stopline) < STOP_DIST_ERR 
                if start_braking: #and not in_intersection:
                    # For each waypoint compute in order:
                        # Distance to the stoplight from waypoint
                        # Distance from waypoint to stopline in front of stoplight
                        # Desired deceleration
                        # Final velocity command for waypoint
                    # Find the distance from the final wpt to the stoplight
                    dis_wpt2stopline = math.sqrt((wpt.pose.pose.position.x - wpt_stopline.pose.pose.position.x)**2 + (wpt.pose.pose.position.y-wpt_stopline.pose.pose.position.y)**2)
                    # Find the desired final stop position of the car
                    self.target_speed_mps = self.target_speed_mps-2.0*1./self.num_wpts_to_stopline*dis_wpt2stopline
                    if self.target_speed_mps < 3.0: # if the velocity is small, just command vel to 0
                        self.target_speed_mps = 0.0
                    if idx >= self.num_wpts_to_stopline:# if the final wpts are ahead of the stopline set the commanded vel to 0
                         self.target_speed_mps = 0.0
                    if idx < self.num_wpts_to_stopline and dis_wpt2stopline < 2.: # if the stopline is 2m ahead just stop
                         self.target_speed_mps = 0.0
                    braking = self.current_velocity > 1e-2                    
                else:
                    # Car goes at normal speed
                    self.target_speed_mps = self.base_wpt_spd_mps #SPEED_MPH*MPH2MPS
                    braking = False
                    dis_wpt2stopline = 999. 
                
                # Set speeds in waypoint list
                wpt.twist.twist.linear.x = self.target_speed_mps
                vel_plan.append(self.target_speed_mps) # for debugging 
                dist_plan.append(dis_wpt2stopline)                  

            #Publish final waypoints
            self.final_wpts.header.stamp    = rospy.Time.now()
            self.final_waypoints_pub.publish(self.final_wpts)
            
            if DEBUG_TGTSPD and braking:
                #print('Distance Plan: {0:.1f}--{1:.1f}--{2:.1f}--{3:.1f}--{4:.1f}--{5:.1f}--{6:.1f}--{7:.1f}--{8:.1f}--{9:.1f}--{10:.1f}--{11:.1f}--{12:.1f}--{13:.1f}--{14:.1f}--{15:.1f}--{16:.1f}--{17:.1f}--{18:.1f}--{19:.1f}--{20:.1f}--{21:.1f}--{22:.1f}--{23:.1f}--{24:.1f}--{25:.1f}--{26:.1f}--{27:.1f}--{28:.1f}--{29:.1f}'.format(\
                #dist_plan[0], dist_plan[1],dist_plan[2],dist_plan[3],dist_plan[4],dist_plan[5],dist_plan[6],dist_plan[7],dist_plan[8],dist_plan[9],dist_plan[10],\
                #dist_plan[11],dist_plan[12],dist_plan[13],dist_plan[14],dist_plan[15],dist_plan[16],dist_plan[17],dist_plan[18],dist_plan[19],dist_plan[20],\
                #dist_plan[21],dist_plan[22],dist_plan[23],dist_plan[24],dist_plan[25],dist_plan[26],dist_plan[27],dist_plan[28],dist_plan[29]))  
                print('Velocity Plan:  {0:.1f}--{1:.1f}--{2:.1f}--{3:.1f}--{4:.1f}--{5:.1f}--{6:.1f}--{7:.1f}--{8:.1f}--{9:.1f}--{10:.1f}--{11:.1f}--{12:.1f}--{13:.1f}--{14:.1f}--{15:.1f}--{16:.1f}--{17:.1f}--{18:.1f}--{19:.1f}--{20:.1f}--{21:.1f}--{22:.1f}--{23:.1f}--{24:.1f}--{25:.1f}--{26:.1f}--{27:.1f}--{28:.1f}--{29:.1f}'.format(\
                vel_plan[0],vel_plan[1],vel_plan[2],vel_plan[3],vel_plan[4],vel_plan[5],vel_plan[6],vel_plan[7],vel_plan[8],vel_plan[9],vel_plan[10],\
                vel_plan[11],vel_plan[12],vel_plan[13],vel_plan[14],vel_plan[15],vel_plan[16],vel_plan[17],vel_plan[18],vel_plan[19],vel_plan[20],\
                vel_plan[21],vel_plan[22],vel_plan[23],vel_plan[24],vel_plan[25],vel_plan[26],vel_plan[27],vel_plan[28],vel_plan[29]))         

            #Topics to publish for debugging
            if DEBUG_TOPICS:
                self.debug_currpos                  = PoseStamped()
                self.debug_currpos.header.stamp     = rospy.Time.now()
                self.debug_currpos.pose.position.x  = self.pos_x
                self.debug_currpos.pose.position.y  = self.pos_y
                self.debug_currpos.pose.position.z  = self.pos_z
                self.debug_currpos.pose.orientation = self.current_orient
                self.debug_currentpos_pub.publish(self.debug_currpos)
        pass

    def pose_cb(self, msg):
        self.pos_x          = msg.pose.position.x
        self.pos_y          = msg.pose.position.y
        self.pos_z          = msg.pose.position.z
        self.current_orient = msg.pose.orientation
        self.loop();
        if False:
            print("WAYPOINT UPDATER :: Curr Pos  ","x: ",self.pos_x, "y: ", self.pos_y)
        
        pass
        
    def upcoming_lt_cb(self,msg):
        # Set default values for light ahead
        self.light_ahead       = None
        self.light_ahead_idx   = None
        self.light_ahead_state = 999
        self.dist2light_m      = 999.
        # Check if light is red or yellow, fill message
        if (msg.state == 0) or (msg.state == 1):
            self.light_ahead       = msg.pose
            self.light_ahead_idx   = msg.waypoint
            self.light_ahead_state = msg.state
            self.dist2light_m      = math.sqrt((self.pos_x - self.light_ahead.pose.position.x)**2 + (self.pos_y-self.light_ahead.pose.position.y)**2)
        if DEBUG_TLDET:
            print("From TL_DETECTOR :: Light Ahead IDX  ",self.light_ahead_idx)
            print("From TL_DETECTOR :: Light Ahead DIST ",self.dist2light_m)
            print("From TL_DETECTOR :: Light Ahead STATE",self.light_ahead_state)
        pass
    
    def traffic_cb(self,msg):
        self.light_ahead_idx = msg.data
        pass

    def sim_traffic_cb(self, msg):
        #Find closest light ahead of us
        closestlen = 9999999
        if self.current_orient != None:
            for idx,light in enumerate(msg.lights):
                dist = self.distance_wpt2curr(light)
                brg  = self.bearing_wpt2curr(light)
                #Find closest light, in front of us, that is either red or yellow
                if dist < closestlen and brg > 0.0 and (light.state == 0 or light.state == 1):
                    self.light_ahead     = light
                    self.light_ahead_idx = idx
                    closestlen = dist
            self.dist2light_m    = closestlen
            
            if DEBUG_TLSIM and self.light_ahead != None:
                print("SIM TL :: Light Ahead IDX          ", self.light_ahead_idx)
                print("SIM TL :: Light Ahead DIST         ", self.dist2light_m)
                
        #Find nearest waypoint to closest light
        if self.waypoints != None and self.light_ahead != None:
            closestlen = 9999999
            base_range = range(len(self.waypoints))
            if self.search_range != None:
                base_range = self.search_range 
            for idx in base_range:
                wpt  = self.waypoints[idx] 
                dist = self.distance_2wpts(self.light_ahead,wpt)
                #Find waypoint that is closest to the nearest red light we just found
                if dist < closestlen:
                    self.wpt_light_ahead     = wpt
                    self.wpt_light_ahead_idx = idx
                    closestlen = dist
        
            if DEBUG_TLSIM:
                print("SIM TL :: Nearest WPT to Light IDX ", self.wpt_light_ahead_idx)
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity
        pass
        
    # Find the nearest waypoint ahead based on current position and bearing to next wpt
    def find_next_waypoint(self):
        closestlen = 9999999
        # Define a range to search for our current positioncurrent_velocity
        self.search_range = range(len(self.waypoints))
                
        # Narrow the search range so we don't need to search the whole \base_waypoints set
        if self.wpt_ahead_idx != None:
            # Handle the case where we wrap around the 0 index of the waypoints
            if self.wpt_ahead_idx+UL_WPT_SEARCH <= len(self.waypoints):
                        
                lower_lim = self.wpt_ahead_idx
                upper_lim = self.wpt_ahead_idx+UL_WPT_SEARCH
                
                if DEBUG_SEARCH:
                        print("--------------------------------------")
                        print("wpt_ahead_idx               ", self.wpt_ahead_idx) 
                        print("wpt_ahead_idx+UL_WPT_SEARCH ", upper_lim)
                    
                self.search_range = range(lower_lim,upper_lim)
            else:
                idx_prev_0        = len(self.waypoints) - self.wpt_ahead_idx
                idx_past_0        = UL_WPT_SEARCH - idx_prev_0
                if idx_prev_0 == 0:
                    range_behind0     = len(self.waypoints)
                else:
                    range_behind0     = range(self.wpt_ahead_idx,len(self.waypoints))
                range_ahead0      = range(0,idx_past_0)
                self.search_range = range_behind0+range_ahead0
                if DEBUG_SEARCH:
                    print("--------------------------------------")
                    print("wpt_ahead_idx ", self.wpt_ahead_idx) 
                    print("idx_prev_0    ", idx_prev_0) 
                    print("idx_past_0    ", idx_past_0)
                    print("range_behind0 ", range_behind0)
                    print("range_ahead0  ", range_ahead0)
                             
        # Loop over \base_waypoints and find the nearest wpt in front of us   
        for idx in self.search_range:
            wpt  = self.waypoints[idx] 
            dist = self.distance_wpt2curr(wpt)
            brg= self.bearing_wpt2curr(wpt)
            #if dist < closestlen and (math.pi/2-brg) < math.pi/4: #check if waypoint is directly ahead within some window
            if dist < closestlen and brg > 0.0:
                self.wpt_ahead     = wpt
                self.wpt_ahead_idx = idx
                closestlen         = dist
                stored_brg         = brg
        #print('stored_brg, self.wpt_ahead_idx, closestlen: ', stored_brg, self.wpt_ahead_idx, closestlen)
        if self.prev_wpt_ahead_idx > self.wpt_ahead_idx:
            rospy.loginfo('Computed waypoint ahead index %i is less than previous waypoint ahead index %i',self.wpt_ahead_idx,self.prev_wpt_ahead_idx)
        
        if self.wpt_ahead_idx > self.prev_wpt_ahead_idx + 10:
            rospy.loginfo('Computed waypoint ahead index %i is at least 10 more than previous waypoint ahead index %i',self.wpt_ahead_idx,self.prev_wpt_ahead_idx)
        
        self.prev_wpt_ahead_idx = self.wpt_ahead_idx
        pass
    
    # When a message is recieved from /base_waypoints topic store it
    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints
        #only need to set /base_waypoints once, unsubscribe to improve performance 
        self.base_waypoints_sub.unregister()
        self.base_wpt_spd_mps = msg.waypoints[1].twist.twist.linear.x
        pass    
    
    # Find the distance from a waypoint to the car's current position
    def distance_wpt2curr(self, wpt):
        dist = math.sqrt((self.pos_x-wpt.pose.pose.position.x)**2 + (self.pos_y-wpt.pose.pose.position.y)**2  + (self.pos_z-wpt.pose.pose.position.z)**2)
        return dist
        
    # Find the distance between 2 specific waypoints   
    def distance_2wpts(self, wpt1, wpt2):
        dist = math.sqrt((wpt2.pose.pose.position.x-wpt1.pose.pose.position.x)**2 + (wpt2.pose.pose.position.y-wpt1.pose.pose.position.y)**2  + (wpt2.pose.pose.position.z-wpt1.pose.pose.position.z)**2)
        return dist
    
    # Find the bearing from current position to waypoint, need to rotate to car body frame
    def bearing_wpt2curr(self, wpt):
        global_car_x = self.pos_x
        global_car_y = self.pos_y
        
        roll,pitch,yaw = tf.transformations.euler_from_quaternion([self.current_orient.x,self.current_orient.y,self.current_orient.z,self.current_orient.w])
        yaw = -1.*yaw 

        self.shiftx = wpt.pose.pose.position.x - global_car_x
        self.shifty = wpt.pose.pose.position.y - global_car_y

        self.del_x =  (self.shiftx)*math.cos(yaw) - (self.shifty)*math.sin(yaw)
        self.del_y =  (self.shiftx)*math.sin(yaw) + (self.shifty)*math.cos(yaw)  
  
        bearing = math.atan2(self.del_x,self.del_y)
        return bearing
        
    # Find the distance between 2 waypoints in a series of waypoints   
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
        
    # When a message is recieved from /current_velocity topic store it
    def CurrVel_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

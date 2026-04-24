#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
import tf
import numpy
import math

"""
 wrenches:
      -
        force:
          x: -0.134995398774
          y: -0.252811705608
          z: -0.0861598399337
        torque:
          x: -0.00194729925705
          y: 0.028723398244
          z: -0.081229664152
    total_wrench:
      force:
        x: -0.134995398774
        y: -0.252811705608
        z: -0.0861598399337
      torque:
        x: -0.00194729925705
        y: 0.028723398244
        z: -0.081229664152
    contact_positions:
      -
        x: -0.0214808318267
        y: 0.00291348151391
        z: -0.000138379966267
    contact_normals:
      -
        x: 0.0
        y: 0.0
        z: 1.0
    depths: [0.000138379966266991]
  -
    info: "Debug:  i:(2/4)     my geom:monoped::lowerleg::lowerleg_contactsensor_link_collision_1\
  \   other geom:ground_plane::link::collision         time:50.405000000\n"
    collision1_name: "monoped::lowerleg::lowerleg_contactsensor_link_collision_1"
    collision2_name: "ground_plane::link::collision"

"""
"""
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
gazebo_msgs/ContactState[] states
  string info
  string collision1_name
  string collision2_name
  geometry_msgs/Wrench[] wrenches
    geometry_msgs/Vector3 force
      float64 x
      float64 y
      float64 z
    geometry_msgs/Vector3 torque
      float64 x
      float64 y
      float64 z
  geometry_msgs/Wrench total_wrench
    geometry_msgs/Vector3 force
      float64 x
      float64 y
      float64 z
    geometry_msgs/Vector3 torque
      float64 x
      float64 y
      float64 z
  geometry_msgs/Vector3[] contact_positions
    float64 x
    float64 y
    float64 z
  geometry_msgs/Vector3[] contact_normals
    float64 x
    float64 y
    float64 z
  float64[] depths
"""

class MonopedState(object):

    def __init__(self, max_height, min_height, abs_max_roll, abs_max_pitch, joint_increment_value = 0.05, done_reward = -1000.0, alive_reward=10.0, desired_force=7.08, desired_yaw=0.0, weight_r1=1.0, weight_r2=1.0, weight_r3=1.0, weight_r4=1.0, weight_r5=1.0, discrete_division=10):
        rospy.logdebug("Starting MonopedState Class object...")
        self.desired_world_point = Vector3(0.0, 0.0, 0.0)
        self._min_height = min_height
        self._max_height = max_height
        self._abs_max_roll = abs_max_roll
        self._abs_max_pitch = abs_max_pitch
        self._joint_increment_value = joint_increment_value
        self._done_reward = done_reward
        self._alive_reward = alive_reward
        self._desired_force = desired_force
        self._desired_yaw = desired_yaw
        self._liftoff_compression = 0.0

        self._weight_r1 = weight_r1
        self._weight_r2 = weight_r2
        self._weight_r3 = weight_r3
        self._weight_r4 = weight_r4
        self._weight_r5 = weight_r5
        self._weight_r6 = rospy.get_param("/weight_r6")

        self._list_of_observations = [
                "distance_from_desired_point",
                "base_roll",
                "base_pitch",
                "base_yaw",
                "contact_force",
                "joint_states_haa",
                "joint_states_hfe",
                "joint_states_kfe",
                "joint_vel_haa",
                "joint_vel_hfe",
                "joint_vel_kfe"]

        self._discrete_division = discrete_division
        # We init the observation ranges and We create the bins now for all the observations
        # self.init_bins()

        self.base_position = Point()
        self.base_orientation = Quaternion()
        self.base_linear_acceleration = Vector3()
        self.contact_force = Vector3()
        self.joints_state = JointState()
        self._was_airborne = False        # tracks if liftoff occurred
        self._jump_count = 0              # counts ground contacts after liftoff
        self._liftoff_height = 0.0        # height at moment of liftoff

        # Odom we only use it for the height detection and planar position ,
        #  because in real robots this data is not trivial.
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # We use the IMU for orientation and linearacceleration detection
        rospy.Subscriber("/monoped/imu/data", Imu, self.imu_callback)
        # We use it to get the contact force, to know if its in the air or stumping too hard.
        rospy.Subscriber("/lowerleg_contactsensor_state", ContactsState, self.contact_callback)
        # We use it to get the joints positions and calculate the reward associated to it
        rospy.Subscriber("/monoped/joint_states", JointState, self.joints_state_callback)

    def check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
        data_pose = None
        while data_pose is None and not rospy.is_shutdown():
            try:
                data_pose = rospy.wait_for_message("/odom", Odometry, timeout=0.1)
                self.base_position = data_pose.pose.pose.position
                rospy.logdebug("Current odom READY")
            except:
                rospy.logdebug("Current odom pose not ready yet, retrying for getting robot base_position")

        imu_data = None
        while imu_data is None and not rospy.is_shutdown():
            try:
                imu_data = rospy.wait_for_message("/monoped/imu/data", Imu, timeout=0.1)
                self.base_orientation = imu_data.orientation
                self.base_linear_acceleration = imu_data.linear_acceleration
                rospy.logdebug("Current imu_data READY")
            except:
                rospy.logdebug("Current imu_data not ready yet, retrying for getting robot base_orientation, and base_linear_acceleration")

        contacts_data = None
        while contacts_data is None and not rospy.is_shutdown():
            try:
                contacts_data = rospy.wait_for_message("/lowerleg_contactsensor_state", ContactsState, timeout=0.1)
                for state in contacts_data.states:
                    self.contact_force = state.total_wrench.force
                rospy.logdebug("Current contacts_data READY")
            except:
                rospy.logdebug("Current contacts_data not ready yet, retrying")

        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/monoped/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))

        rospy.logdebug("ALL SYSTEMS READY")

    def set_desired_world_point(self, x, y, z):
        """
        Point where you want the Monoped to be
        :return:
        """
        self.desired_world_point.x = x
        self.desired_world_point.y = y
        self.desired_world_point.z = z


    def get_base_height(self):
        return self.base_position.z

    def get_base_rpy(self):
        euler_rpy = Vector3()
        euler = tf.transformations.euler_from_quaternion(
            [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w])

        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]
        return euler_rpy

    def get_distance_from_point(self, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((self.base_position.x, self.base_position.y, self.base_position.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_contact_force_magnitude(self):
        """
        You will see that because the X axis is the one pointing downwards, it will be the one with
        higher value when touching the floor
        For a Robot of total mas of 0.55Kg, a gravity of 9.81 m/sec**2, Weight = 0.55*9.81=5.39 N
        Falling from around 5centimetres ( negligible height ), we register peaks around
        Fx = 7.08 N
        :return:
        """
        contact_force = self.contact_force
        contact_force_np = numpy.array((contact_force.x, contact_force.y, contact_force.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

    def get_joint_states(self):
        return self.joints_state

    def odom_callback(self,msg):
        self.base_position = msg.pose.pose.position

    def imu_callback(self,msg):
        self.base_orientation = msg.orientation
        self.base_linear_acceleration = msg.linear_acceleration

    def contact_callback(self,msg):
        """
        /lowerleg_contactsensor_state/states[0]/contact_positions ==> PointContact in World
        /lowerleg_contactsensor_state/states[0]/contact_normals ==> NormalContact in World

        ==> One is an array of all the forces, the other total,
         and are relative to the contact link referred to in the sensor.
        /lowerleg_contactsensor_state/states[0]/wrenches[]
        /lowerleg_contactsensor_state/states[0]/total_wrench
        :param msg:
        :return:
        """
        for state in msg.states:
            self.contact_force = state.total_wrench.force

    def joints_state_callback(self,msg):
        self.joints_state = msg

    def monoped_height_ok(self):

        height_ok = self._min_height <= self.get_base_height() < self._max_height
        return height_ok

    def monoped_orientation_ok(self):

        orientation_rpy = self.get_base_rpy()
        roll_ok = self._abs_max_roll > abs(orientation_rpy.x)
        pitch_ok = self._abs_max_pitch > abs(orientation_rpy.y)
        orientation_ok = roll_ok and pitch_ok
        return orientation_ok

    def calculate_reward_joint_position(self, weight=1.0):
        """
        Only penalize HAA (lateral sway joint) for deviating from 0.
        KFE and HFE are NOT penalized here — they need to bend for balance and jumping.
        """
        haa_pos = abs(self.joints_state.position[0])
        reward = weight * haa_pos
        rospy.logdebug("calculate_reward_joint_position>>reward=" + str(reward))
        return reward

    def calculate_reward_knee_bend(self, weight=1.0):
        hfe = self.joints_state.position[1]
        kfe = self.joints_state.position[2]
        # HFE should be positive, KFE should be negative
        # Reward their coordinated opposite-direction bend
        # Use the product — positive only when both have correct signs
        coordination = hfe * (-kfe)   # positive when hfe>0 and kfe<0
        reward = weight * max(0.0, coordination)
        return reward

    def calculate_reward_joint_effort(self, weight=1.0):
        """
        We calculate reward base on the joints effort readings. The more near 0 the better.
        :return:
        """
        acumulated_joint_effort = 0.0
        for joint_effort in self.joints_state.effort:
            # Abs to remove sign influence, it doesnt matter the direction of the effort.
            acumulated_joint_effort += abs(joint_effort)
            rospy.logdebug("calculate_reward_joint_effort>>joint_effort=" + str(joint_effort))
            rospy.logdebug("calculate_reward_joint_effort>>acumulated_joint_effort=" + str(acumulated_joint_effort))
        reward = weight * acumulated_joint_effort
        rospy.logdebug("calculate_reward_joint_effort>>reward=" + str(reward))
        return reward

    def calculate_reward_contact_force(self, weight=1.0):
        """
        We calculate reward base on the contact force.
        The nearest to the desired contact force the better.
        We use exponential to magnify big departures from the desired force.
        Default ( 7.08 N ) desired force was taken from reading of the robot touching
        the ground from a negligible height of 5cm.
        :return:
        """
        force_magnitude = self.get_contact_force_magnitude()
        force_displacement = force_magnitude - self._desired_force

        rospy.logdebug("calculate_reward_contact_force>>force_magnitude=" + str(force_magnitude))
        rospy.logdebug("calculate_reward_contact_force>>force_displacement=" + str(force_displacement))
        # Abs to remove sign
        reward = weight * abs(force_displacement)
        rospy.logdebug("calculate_reward_contact_force>>reward=" + str(reward))
        return reward

    def calculate_reward_orientation(self, weight=1.0):
        """
        We calculate the reward based on the orientation.
        The more its closser to 0 the better because it means its upright
        desired_yaw is the yaw that we want it to be.
        to praise it to have a certain orientation, here is where to set it.
        :return:
        """
        curren_orientation = self.get_base_rpy()
        yaw_displacement = curren_orientation.z - self._desired_yaw
        rospy.logdebug("calculate_reward_orientation>>[R,P,Y]=" + str(curren_orientation))
        acumulated_orientation_displacement = abs(curren_orientation.x) + abs(curren_orientation.y) + abs(yaw_displacement)
        reward = weight * acumulated_orientation_displacement
        rospy.logdebug("calculate_reward_orientation>>reward=" + str(reward))
        return reward

    def calculate_reward_distance_from_des_point(self, weight=1.0):
        dx = self.base_position.x - self.desired_world_point.x
        dy = self.base_position.y - self.desired_world_point.y
        distance = numpy.sqrt(dx**2 + dy**2)
        reward = weight * distance
        return reward

    def calculate_total_reward(self):
        
        r1 = self.calculate_reward_joint_position(self._weight_r1)
        r2 = self.calculate_reward_joint_effort(self._weight_r2)
        r4 = self.calculate_reward_orientation(self._weight_r4)
        r_height = self.calculate_reward_height(self._weight_r6)
        r_knee = self.calculate_reward_knee_bend(weight=3.0)
        r_jump = self.calculate_reward_jump()

        total_reward = self._alive_reward + r_height + r_jump - r1 - r2 - r4

        rospy.logdebug("###############")
        rospy.logdebug("r_height=" + str(r_height))
        rospy.logdebug("r_knee=" + str(r_knee))
        rospy.logdebug("r_jump=" + str(r_jump))
        rospy.logdebug("total_reward=" + str(total_reward))
        rospy.logdebug("###############")

        return total_reward
    
    def reset_jump_state(self):
        self._was_airborne = False
        self._jump_count = 0
        self._liftoff_height = 0.0
        self._liftoff_compression = 0.0
    
    def calculate_reward_forward_progress(self):
        x = self.base_position.x
        y = self.base_position.y
        horizontal_displacement = numpy.sqrt(x**2 + y**2)
        reward = 0.8 * horizontal_displacement    # was 1.5
        return reward
    
    def calculate_reward_jump(self):
        contact = self.get_contact_force_magnitude()
        height = self.get_base_height()
        standing_baseline = 0.45
        is_airborne = contact < 2.0
        reward = 0.0

        if is_airborne:
            if not self._was_airborne:
                self._liftoff_height = height
                self._was_airborne = True
            reward = 15.0 * max(0.0, height - standing_baseline)
        else:
            if self._was_airborne:
                if self._liftoff_height > standing_baseline + 0.05:
                    reward = 20.0
                self._was_airborne = False
                self._liftoff_height = 0.0

        return reward
    
    def calculate_reward_height(self, weight=1.0):
        """
        Reward being close to target_height (slightly crouched).
        Gaussian-shaped: peaks at target, zero if error >= 0.2m.
        This prevents straight-leg being optimal and allows
        compression phase of jump without heavy penalty.
        """
        height = self.get_base_height()
        reward = weight * height
        rospy.logdebug("calculate_reward_height>>reward=" + str(reward))
        return reward


    def get_observations(self):
        """
        Returns the state of the robot needed for OpenAI QLearn Algorithm
        The state will be defined by an array of the:
        1) distance from desired point in meters
        2) The pitch orientation in radians
        3) the Roll orientation in radians
        4) the Yaw orientation in radians
        5) Force in contact sensor in Newtons
        6-7-8) State of the 3 joints in radians

        observation = [distance_from_desired_point,
                 base_roll,
                 base_pitch,
                 base_yaw,
                 contact_force,
                 joint_states_haa,
                 joint_states_hfe,
                 joint_states_kfe]

        :return: observation
        """

        distance_from_desired_point = self.get_distance_from_point(self.desired_world_point)

        base_orientation = self.get_base_rpy()
        base_roll = base_orientation.x
        base_pitch = base_orientation.y
        base_yaw = base_orientation.z

        contact_force = self.get_contact_force_magnitude()

        joint_states = self.get_joint_states()
        joint_states_haa = joint_states.position[0]
        joint_states_hfe = joint_states.position[1]
        joint_states_kfe = joint_states.position[2]

        observation = []
        # import ipdb; ipdb.set_trace()
        for obs_name in self._list_of_observations:
            if obs_name == "distance_from_desired_point":
                observation.append(distance_from_desired_point)
            elif obs_name == "base_roll":
                observation.append(base_roll)
            elif obs_name == "base_pitch":
                observation.append(base_pitch)
            elif obs_name == "base_yaw":
                observation.append(base_yaw)
            elif obs_name == "contact_force":
                observation.append(contact_force)
            elif obs_name == "joint_states_haa":
                observation.append(joint_states_haa)
            elif obs_name == "joint_states_hfe":
                observation.append(joint_states_hfe)
            elif obs_name == "joint_states_kfe":
                observation.append(joint_states_kfe)
            elif obs_name == "joint_vel_haa":
                observation.append(self.joints_state.velocity[0])
            elif obs_name == "joint_vel_hfe":
                observation.append(self.joints_state.velocity[1])
            elif obs_name == "joint_vel_kfe":
                observation.append(self.joints_state.velocity[2])
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))

        return observation

    def get_action_to_position(self, action):
        joint_states = self.get_joint_states()
        joint_states_position = joint_states.position

        action_position = [0.0, 0.0, 0.0]
        action_position[0] = joint_states_position[0] + action[0]
        action_position[1] = joint_states_position[1] + action[1]
        action_position[2] = joint_states_position[2] + action[2]

        # Clamp to prevent extreme joint configurations
        action_position[0] = numpy.clip(action_position[0], -0.5, 0.5)   # HAA
        action_position[1] = numpy.clip(action_position[1],  0.0, 1.0)   # HFE: was 1.0
        action_position[2] = numpy.clip(action_position[2], -1.0, 0.0)   # KFE: was -1.0 only

        return action_position

    def process_data(self):
        """
        We return the total reward based on the state in which we are in and if its done or not
        ( it fell basically )
        :return: reward, done
        """
        monoped_height_ok = self.monoped_height_ok()
        monoped_orientation_ok = self.monoped_orientation_ok()
        done = not(monoped_height_ok and monoped_orientation_ok)

        if done:
            rospy.logdebug("It fell, so the reward has to be very low")
            total_reward = self._done_reward
            self.reset_jump_state()    # ADD THIS
        else:
            rospy.logdebug("Calculate normal reward because it didn't fall.")
            total_reward = self.calculate_total_reward()

        return total_reward, done

    def testing_loop(self):

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.calculate_total_reward()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('monoped_state_node', anonymous=True)
    monoped_state = MonopedState(max_height=3.0,
                                 min_height=0.6,
                                 abs_max_roll=0.7,
                                 abs_max_pitch=0.7)
    monoped_state.testing_loop()
#!/usr/bin/env python
# Imports
import rospy
import sys
# from agile_grasp2.msg import GraspListMsg
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseArray, Pose, Twist
from visualization_msgs.msg import MarkerArray, Marker
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header, Float64
import numpy as np
import tf
from tf import TransformListener
import copy
from time import sleep
import roslaunch
import math
import pdb
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rosbag
import time, timeit
import serial

import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState, RobotTrajectory, CollisionObject
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, \
    _Robotiq2FGripper_robot_input as inputMsg
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from util import dist_to_guess, vector3ToNumpy, find_center, dist_two_points, smallestSignedAngleBetween, \
    calculate_approach, generate_push_pose, find_nearest_corner, floatToMsg, command_gripper, get_robot_state, \
    lift_up_plan, move_back_plan, add_front_wall, add_right_wall, add_left_wall, add_back_wall, add_roof

from pyquaternion import Quaternion

import pdb
from enum import Enum


# -*- coding: utf-8 -*-

# -- Standard
import open3d
import numpy as np
import sys, os, cv2
from copy import copy
PYTHON_FILE_PATH = os.path.join(os.path.dirname(__file__))+"/"

# -- ROS
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler, euler_matrix

# -- My lib
sys.path.append(PYTHON_FILE_PATH + "../src_python")
# from lib_baxter import MyBaxter
from lib_geo_trans_ros import form_T, quaternion_to_R, toRosPose, pose2T, transXYZ
from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromOpen3dToRos

# -- Message types
# from std_msgs.msg import Int32  # used for indexing the ith robot pose
from geometry_msgs.msg import Pose, Point, Quaternion
from scan3d_by_baxter.msg import T4x4


# ------------------------------------------------------------

# -- Functions: basic
# Change int to str and filled prefix with 0s
def int2str(x, width): return ("{:0"+str(width)+"d}").format(x)
    
def getCloudSize(open3d_cloud):
    return np.asarray(open3d_cloud.points).shape[0]

# -- Functions: Baxter/Camera related
class JointPosPublisher(object):
    def __init__(self, topic_endeffector_pos):
        self.pub = rospy.Publisher(topic_endeffector_pos, T4x4, queue_size=10)

    def publishPose(self, pose):
        T = pose
        # Trans to 1x16 array
        pose_1x16 = []
        for i in range(4):
            for j in range(4):
                pose_1x16 += [T[i, j]]
        self.pub.publish(pose_1x16)
        return

def readKinectCameraPose():
    T_base_to_arm = my_Baxter.getFramePose('/left_lower_forearm')
    T_arm_to_depth  # This is read from file
    T = T_base_to_arm.dot(T_arm_to_depth)
    return T

def set_target_joint_angles():
    target_joint_angles=[
        [-1.4545972347259521, 0.40343695878982544, 1.460349678993225, 2.304422616958618, -1.2390730381011963, 1.0768544673919678, -3.0480198860168457]
        ,[-0.849825382232666, -0.35895150899887085, 1.144733190536499, 1.5957235097885132, -0.32597091794013977, 1.2973642349243164, -3.0480198860168457]
        # ,[-0.849825382232666, -0.35895150899887085, 1.144733190536499, 1.5957235097885132, -0.32597091794013977, 1.2973642349243164, -3.0480198860168457]
        ,[-0.5173349976539612, -0.5058301687240601, 0.9146360158920288, 1.584985613822937, -0.08973787724971771, 1.1888351440429688, -3.0480198860168457]
        ,[0.06404370069503784, -0.49854376912117004, 0.7117670774459839, 1.1489516496658325, -0.2105388641357422, 1.5604419708251953, -3.0468692779541016]
        ,[0.6830049753189087, -0.358568012714386, 0.10277671366930008, 0.6925923228263855, -0.060208745300769806, 1.9170924425125122, -3.0480198860168457]
        ,[0.8114758133888245, -0.520786464214325, 0.41072335839271545, 0.8248981833457947, -0.3497476279735565, 1.8860293626785278, -3.0480198860168457]
        ,[1.0270001888275146, -0.7516505718231201, 0.6688156127929688, 1.6904468536376953, -0.8095583319664001, 1.4799079895019531, -3.0476362705230713]
        ,[0.8563447594642639, -0.8893253803253174, 1.0243157148361206, 1.940102219581604, -0.7616214752197266, 1.4526797533035278, -3.047252893447876]
        ,[1.2160632610321045, -0.4141748249530792, 1.0515438318252563, 2.356194496154785, -1.3150050640106201, 1.7598594427108765, -3.0468692779541016]
        # ,[0.7405292391777039, -0.962956428527832, 1.0948787927627563, 2.177485704421997, 2.173267364501953, -1.5439516305923462, -3.0480198860168457]
        # ,[1.260932207107544, -0.9418641924858093, 1.0753204822540283, 2.452451705932617, 2.1418206691741943, -1.5136555433273315, -3.023859739303589]
   ]
    return target_joint_angles


# -- Functions: Write results to file

# Write ndarray to file
def write_ndarray_to_file(fout, mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            fout.write(str(mat[i][j])+" ")
        fout.write("\n")

# Write rgb-d camera's current pose to file
g_poses_storage=[]
def savePoseToFile(pose, ith_goalpose, clear=False):
    filename = file_folder+file_name_pose + ".txt"
    if clear==True:
        fout = open(filename,"w")
        fout.close()
        return

    # Write to file
    fout = open(filename,"a")
    fout.write("\n" + int2str(ith_goalpose,2)+"th pose: \n")
    write_ndarray_to_file(fout, pose)
    
    # Return 
    fout.close()
    g_poses_storage.append(pose)
    return

# --
# Grasp Class
class GraspExecutor:
    # Initialisation
    def __init__(self):
        # Initialisation
        rospy.init_node('push_grasp', anonymous=True)

        self.tf_listener_ = TransformListener()
        self.launcher = roslaunch.scriptapi.ROSLaunch()
        self.launcher.start()
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        # Publisher for grasp poses
        self.pose_publisher = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)

        # Hard-coded joint values
        # self.view_home_joints = [0.24985386431217194, -0.702608887349264, -2.0076406637774866, -1.7586587111102503, 1.5221580266952515, 0.25777095556259155]
        self.view_home_joints = [0.07646834850311279, -0.7014802137957972, -2.008395496998922, -1.1388691107379358,
                                 1.5221940279006958, 0.06542113423347473]

        self.view_home_pose = PoseStamped()
        self.view_home_pose.header.frame_id = "base_link"
        self.view_home_pose.pose.position.x = -0.284710
        self.view_home_pose.pose.position.y = 0.099278
        self.view_home_pose.pose.position.z = 0.442958
        self.view_home_pose.pose.orientation.x = 0.243318
        self.view_home_pose.pose.orientation.y = 0.657002
        self.view_home_pose.pose.orientation.z = -0.669914
        self.view_home_pose.pose.orientation.w = 0.245683

        self.move_home_joints = [0.04602504149079323, -2.2392290274249476, -1.0055387655841272, -1.4874489943133753,
                                 1.6028196811676025, 0.030045202001929283]

        # Set default robot states
        self.move_home_robot_state = get_robot_state(self.move_home_joints)
        self.view_home_robot_state = get_robot_state(self.view_home_joints)

        # RGB Image
        self.rgb_sub = rospy.Subscriber('/realsense/rgb', Image, self.rgb_callback)
        self.cv_image = []
        self.image_number = 0

        # Depth Image
        self.rgb_sub = rospy.Subscriber('/realsense/depth', Image, self.depth_image_callback)
        self.depth_image = []

    def rgb_callback(self, image):
        self.rgb_image = image
        self.cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        self.image_number += 1

    def depth_image_callback(self, image):
        self.depth_image = image

    def move_to_position(self, grasp_pose, plan=None, first_move=False):
        if first_move:
            run_flag = "d"
        else:
            run_flag = "d"

        if not plan:
            if not first_move:
                (plan, fraction) = self.move_group.compute_cartesian_path([grasp_pose.pose], 0.01, 0)
                if fraction != 1:
                    rospy.logwarn("lol rip: %f", fraction)
                    run_flag = "n"
            elif first_move:
                plan = self.move_group.plan()


        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

        if run_flag == "y":
            self.move_group.execute(plan, wait=True)
            successful = True
        else:
            successful = False
            rospy.loginfo("Path cancelled")

        self.move_group.stop()
        self.move_group.clear_pose_targets()

        return successful

    def move_to_joint_position(self, joint_array, plan=None):
        self.move_group.set_joint_value_target(joint_array)
        if not plan:
            plan = self.move_group.plan()

        run_flag = "y"

        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

        if run_flag == "y":
            self.move_group.execute(plan, wait=True)

        self.move_group.stop()
        self.move_group.clear_pose_targets()

        return

    
# --


# -- Main
if __name__ == "__main__":

    # ---------------------------------------------------------------------

    rospy.init_node('Node 1')

    # -- Set Params

    file_folder = rospy.get_param("file_folder")
    file_name_pose = rospy.get_param("file_name_pose")
    file_name_index_width = rospy.get_param("file_name_index_width")

    config_folder = rospy.get_param("file_folder_config")
    file_name_T_arm_to_depth = rospy.get_param("file_name_T_arm_to_depth")
    T_arm_to_depth = np.loadtxt(config_folder+file_name_T_arm_to_depth)
    # T_baxter_to_chess = np.loadtxt(config_folder+"T_baxter_to_chess.txt")

    topic_endeffector_pos = rospy.get_param("topic_n1_to_n2")
    num_goalposes = rospy.get_param("num_goalposes")

    # -- Set Baxter
    # my_Baxter = MyBaxter(['left', 'right'][0])
    # my_Baxter.enableBaxter()

    # -- Set UR5
    grasper = GraspExecutor()
    grasper.main()

    # -- Set publisher: After Baxter moves to the next goalpose position,
    #   sends the pose to node2 to tell it to take the picture.
    pub = JointPosPublisher(topic_endeffector_pos)

    # ---------------------------------------------------------------------

    # Start node when pressing enter
    rospy.loginfo("\n\nWaiting for pressing 'enter' to start ...")
    raw_input("")

    # -- Speicify the goalpose_joint_angles that the Baxter needs to move to
    list_target_joint_angles = set_target_joint_angles()
    
    # -- Move Baxter to initial position
    DEBUG__I_DONT_HAVE_BAXTER=False
    if not DEBUG__I_DONT_HAVE_BAXTER:
        rospy.sleep(1)
        init_joint_angles = list_target_joint_angles[0]
        rospy.loginfo("\n\nNode 1: Initialization. Move Baxter to init pose: "+str(init_joint_angles))
        # my_Baxter.moveToJointAngles(init_joint_angles, time_cost=3.0)

        rospy.loginfo("Node 1: Baxter reached the initial pose!\n\n")

    # -- Move Baxter to all goal positions
    ith_goalpose = 0
    savePoseToFile(None, None, clear=True)

    while ith_goalpose < num_goalposes and not rospy.is_shutdown():
        ith_goalpose += 1
        joint_angles = list_target_joint_angles[ith_goalpose-1]

        # Move robot to the next pose for taking picture
        if not DEBUG__I_DONT_HAVE_BAXTER:
            rospy.loginfo("\n\n------------------------------------------------------")
            rospy.loginfo("Node 1: {}th pos".format(ith_goalpose))
            rospy.loginfo("Node 1: Baxter is moving to pos: "+str(joint_angles))

            my_Baxter.moveToJointAngles(joint_angles, 4.0)
            
            # if ith_goalpose<=8:
            # elif ith_goalpose==9:
                # my_Baxter.moveToJointAngles(joint_angles, 8.0)
            # else:
                # my_Baxter.moveToJointAngles(joint_angles, 3.0)

            rospy.loginfo("Node 1: Baxter reached the pose!")

        rospy.loginfo("Node 1: Wait until stable for 1 more second")
        rospy.sleep(1.0)
        rospy.loginfo("Node 1: publish "+str(ith_goalpose) +
                      "th camera pose to node2")

        # Publish camera pose to node2
        pose = readKinectCameraPose()
        pub.publishPose(pose)
        
        # End
        savePoseToFile(pose, ith_goalpose)
        rospy.loginfo("--------------------------------")
        rospy.sleep(1)
        # if ith_goalpose==num_goalposes: ith_goalpose = 0

    # -- Node stops
    rospy.loginfo("!!!!! Node 1 stops.")





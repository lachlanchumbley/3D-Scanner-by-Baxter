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

from util import dist_to_guess, vector3ToNumpy, find_center, dist_two_points, smallestSignedAngleBetween, \
    calculate_approach, generate_push_pose, find_nearest_corner, floatToMsg, command_gripper, get_robot_state, \
    lift_up_plan, move_back_plan, add_front_wall, add_right_wall, add_left_wall, add_back_wall, add_roof

from pyquaternion import Quaternion

import pdb
from enum import Enum

import ros_numpy

from open3d_ros_helper import open3d_ros_helper as orh

# -*- coding: utf-8 -*-

# -- Standard
import open3d as o3d
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
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
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
        # self.pub = rospy.Publisher(topic_endeffector_pos, T4x4, queue_size=10)

    def publishPose(self, pose):
        T = pose
        # Trans to 1x16 array
        pose_1x16 = []
        for i in range(4):
            for j in range(4):
                pose_1x16 += [T[i, j]]
        self.pub.publish(pose_1x16)
        return

# def readKinectCameraPose():
#     T_base_to_arm = my_Baxter.getFramePose('/left_lower_forearm')
#     T_arm_to_depth  # This is read from file
#     T = T_base_to_arm.dot(T_arm_to_depth)
#     return T

def readRealsenseCameraPose():
    # Listen to tf
    tf_listener.waitForTransform("/camera_link", "/base_link", rospy.Time(), rospy.Duration(4))
    (trans, rot) = tf_listener.lookupTransform('/base_link', '/camera_link', rospy.Time(0))
    T = tf_listener.fromTranslationRotation(trans, rot)
    return T

def set_target_joint_angles():
#     black_target_joint_angles=[
#         [0.07061217725276947, -0.29163867632021123, -1.6646230856524866, -1.9139745871173304, 1.600701093673706, -0.0441821257220667],
#         [-0.8805802504168909, -0.8451650778399866, -2.0639851729022425, -1.5810025374041956, 2.2264199256896973, -0.023767773305074513],
#         [-0.5446856657611292, -1.6238592306720179, -1.3436005751239222, -1.936120335255758, 1.9973962306976318, -0.023576084767476857],
#         [-0.27595216432680303, -1.8121359984027308, -1.0498798529254358, -2.1475256125079554, 1.8328351974487305, -0.02369577089418584],
#         [0.08777359127998352, -1.923241917287008, -0.732833210621969, -2.3586061636554163, 1.6853671073913574, -0.023492161427633107],
#         [0.47369709610939026, -1.8756073156939905, -0.9190285841571253, -2.325972382222311, 1.5827951431274414, -0.03040486971010381],
#         [0.6731565594673157, -1.6868918577777308, -1.1725323835956019, -2.2223241964923304, 1.4596389532089233, 0.17617036402225494],
#         [1.0090272426605225, -1.249017063771383, -1.590496842061178, -2.113292996083395, 1.2705281972885132, 0.1747685819864273],
#         [1.1530221700668335, -1.0215514341937464, -1.7954867521869105, -2.0860732237445276, 1.2926769256591797, 0.17481651902198792],
#         [0.1879156529903412, -0.7676246801959437, -1.5441902319537562, -1.9610946814166468, 1.6043753623962402, 0.17708078026771545]
#    ]

    orange_target_joint_angles=[
        [0.003808362176641822, 0.07165288925170898, -2.6987369696246546, -1.4492157141314905, 1.7440826892852783, -0.07086211839784795],
        [-1.1421869436847132, -1.3745406309710901, -1.9023945967303675, -2.5925162474261683, 2.7412893772125244, 0.3463781774044037],
        [-0.8620308081256312, -1.9371183554278772, -1.4351885954486292, -2.755387608204977, 2.3295176029205322, 0.014399795792996883],
        [-0.30975419679750615, -1.8845709005938929, -1.2908647696124476, -2.525616470967428, 2.0564534664154053, 0.34377872943878174],
        [0.09318708628416061, -2.1642192045794886, -1.0047724882708948, -2.7548721472369593, 1.6673308610916138, 0.014519637450575829],
        [0.6234591007232666, -1.9098685423480433, -1.2959659735309046, -2.5702598730670374, 1.270193099975586, -0.19125396410097295],
        [1.0967791080474854, -1.759998623524801, -1.5978997389422815, -2.712764088307516, 1.0855352878570557, -0.1526878515826624],
        [1.5562212467193604, -1.1320880095111292, -2.0887721220599573, -2.567897144948141, 0.8982843160629272, -0.19519550005067998],
        [0.09521092474460602, -1.1995633284198206, -1.4911907354937952, -2.2984231154071253, 1.6042555570602417, 0.03313600644469261]
   ]

#    Result from /tool0 to /camera_link:
#     0.386381  -0.757978   0.525528   0.156615
#     -0.038549   0.556008   0.830282 -0.0567307
#     -0.921533  -0.341064   0.185612   0.128036
#             0          0          0          1

#     Translation (x,y,z) :   0.156615 -0.0567307   0.128036
#     Rotation q(x,y,z,w): -0.401485 0.495988 0.246588 0.729384
#     Now you can publish tf in: [ Translation, Rotation] /tool0 /camera_link

#   <group if="$(eval arg('calibration') == 'orange')">
#     <node pkg="tf" 
#         type="static_transform_publisher" 
#         name="realsense_link" 
#         args="-0.033595 -0.064158  0.066946  
#                0.18722 0.00188329 -0.0080284 0.982283
#               /tool0
#               /camera_link
#               100"
#   />
#   </group>


    return orange_target_joint_angles

def rospc_to_o3dpc(rospc, remove_nans=False):
    """ covert ros point cloud to open3d point cloud
    Args: 
        rospc (sensor.msg.PointCloud2): ros point cloud message
        remove_nans (bool): if true, ignore the NaN points
    Returns: 
        o3dpc (open3d.geometry.PointCloud): open3d point cloud
    """
    field_names = [field.name for field in rospc.fields]
    is_rgb = 'rgba' in field_names
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(rospc).ravel()
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    if is_rgb:
        cloud_npy = np.zeros(cloud_array.shape + (4,), dtype=np.float)
    else: 
        cloud_npy = np.zeros(cloud_array.shape + (3,), dtype=np.float)
    
    cloud_npy[...,0] = cloud_array['x']
    cloud_npy[...,1] = cloud_array['y']
    cloud_npy[...,2] = cloud_array['z']
    o3dpc = o3d.geometry.PointCloud()

    if len(np.shape(cloud_npy)) == 3:
        cloud_npy = np.reshape(cloud_npy[:, :, :3], [-1, 3], 'F')
    o3dpc.points = o3d.utility.Vector3dVector(cloud_npy[:, :3])

    if is_rgb:
        rgb_npy = cloud_array['rgba']
        rgb_npy.dtype = np.uint32
        r = np.asarray((rgb_npy >> 16) & 255, dtype=np.uint8)
        g = np.asarray((rgb_npy >> 8) & 255, dtype=np.uint8)
        b = np.asarray(rgb_npy & 255, dtype=np.uint8)
        rgb_npy = np.asarray([r, g, b])
        rgb_npy = rgb_npy.astype(np.float)/255
        rgb_npy = np.swapaxes(rgb_npy, 0, 1)
        o3dpc.colors = o3d.utility.Vector3dVector(rgb_npy)
    return o3dpc

def realsense_callback(pc2_msg):
    o3dpc = rospc_to_o3dpc(pc2_msg)
    global current_point_cloud
    current_point_cloud = o3dpc
    return

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
    # fout.write("\n" + int2str(ith_goalpose,2)+"th pose: \n")
    write_ndarray_to_file(fout, pose)
    fout.write("\n")
    
    # Return 
    fout.close()
    g_poses_storage.append(pose)
    return

# Write point cloud to file
def savePointCloudToFile(cloud, ith_goalpose):
    filename = "/home/acrv/new_ws/src/3D-Scanner-by-Baxter/data/data/" + "src_" + str(ith_goalpose) + ".pcd"

    # Write to file
    o3d.io.write_point_cloud(filename, cloud)
    return

# --
# Grasp Class
class GraspExecutor:
    # Initialisation
    def __init__(self):
        # Initialisation
        # rospy.init_node('push_grasp', anonymous=True)

        # self.tf_listener_ = TransformListener()
        self.launcher = roslaunch.scriptapi.ROSLaunch()
        self.launcher.start()
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # Hard-coded joint values
        self.view_home_joints = [0.07646834850311279, -0.7014802137957972, -2.008395496998922, -1.1388691107379358, 1.5221940279006958, 0.06542113423347473]
        self.move_home_joints = [0.04602504149079323, -2.2392290274249476, -1.0055387655841272, -1.4874489943133753, 1.6028196811676025, 0.030045202001929283]

        # View home pose
        self.view_home_pose = PoseStamped()
        self.view_home_pose.header.frame_id = "base_link"
        self.view_home_pose.pose.position.x = -0.284710
        self.view_home_pose.pose.position.y = 0.099278
        self.view_home_pose.pose.position.z = 0.442958
        self.view_home_pose.pose.orientation.x = 0.243318
        self.view_home_pose.pose.orientation.y = 0.657002
        self.view_home_pose.pose.orientation.z = -0.669914
        self.view_home_pose.pose.orientation.w = 0.245683

        # Set default robot states
        self.move_home_robot_state = get_robot_state(self.move_home_joints)
        self.view_home_robot_state = get_robot_state(self.view_home_joints)

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

    # Initialise tf listener
    tf_listener = TransformListener()

    # -- Set Params

    file_folder = rospy.get_param("file_folder")
    file_name_pose = rospy.get_param("file_name_pose")
    file_name_index_width = rospy.get_param("file_name_index_width")
    topic_name_rgbd_cloud = rospy.get_param("topic_name_rgbd_cloud")

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
    ur5 = GraspExecutor()

    # -- Set publisher: After Baxter moves to the next goalpose position,
    #   sends the pose to node2 to tell it to take the picture.
    pub = JointPosPublisher(topic_endeffector_pos)

    # Sub to Realsense
    
    rospy.Subscriber(topic_name_rgbd_cloud, PointCloud2, realsense_callback)

    # Finished publisher
    # n1_finish_pub = rospy.Publisher("n1_finished", int, queue_size=1)
    # n1_finish_pub.publish(0)

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
        ur5.move_to_joint_position(init_joint_angles)
        rospy.sleep(3)

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

            # my_Baxter.moveToJointAngles(joint_angles, 4.0)
            ur5.move_to_joint_position(joint_angles)
            rospy.sleep(4)

            rospy.loginfo("Node 1: Baxter reached the pose!")

        rospy.loginfo("Node 1: Wait until stable for 1 more second")
        rospy.sleep(5.0)
        rospy.loginfo("Node 1: publish "+str(ith_goalpose) +
                      "th camera pose to node2")

        # Publish camera pose to node2
        pose = readRealsenseCameraPose()
        pub.publishPose(pose)

        # Publish camera pose to node2
        savePointCloudToFile(current_point_cloud, ith_goalpose)
        
        # End
        savePoseToFile(pose, ith_goalpose)
        rospy.loginfo("--------------------------------")
        rospy.sleep(1)
        # if ith_goalpose==num_goalposes: ith_goalpose = 0

    rospy.loginfo("Node 1 Finished")
    # n1_finish_pub.publish(1)

    # -- Node stops
    rospy.loginfo("!!!!! Node 1 stops.")





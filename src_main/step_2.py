#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Include common
import numpy as np
import open3d as o3d
import sys, os, copy
from collections import deque
PYTHON_FILE_PATH = os.path.join(os.path.dirname(__file__))+"/"

# Include ROS
import rospy
from sensor_msgs.msg import PointCloud2

# Include my lib
sys.path.append(PYTHON_FILE_PATH + "../src_python")
from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromOpen3dToRos, convertCloudFromRosToOpen3d
from lib_cloud_registration import CloudRegister, resizeCloudXYZ, mergeClouds, createXYZAxis, getCloudSize, filtCloudByRange

from lib_geo_trans import rotx, roty, rotz

VIEW_RES_BY_OPEN3D=True # This is difficult to set orientation. And has some bug.
VIEW_RES_BY_RVIZ=~VIEW_RES_BY_OPEN3D
OBJECT_RANGE = 0.12 #The object is inside a region of x=(-r,r) && y=(-r,r)
n2_done_flag = False

# ---------------------------- Two viewers (choose one) ----------------------------
class Open3DViewer(object):
    def __init__(self):
        self.vis_cloud = o3d.geometry.PointCloud()
        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window()
        self.viewer.add_geometry(self.vis_cloud)
        print("Initialised")

    def updateCloud(self, new_cloud):
        self.vis_cloud.points = copy.deepcopy(new_cloud.points)
        self.vis_cloud.colors = copy.deepcopy(new_cloud.colors)
        self.viewer.add_geometry(self.vis_cloud)
        self.viewer.update_geometry()

    def updateView(self):
        self.viewer.poll_events()
        self.viewer.update_renderer()
        print("NOTICE: OPEN3D VIEWER HAS BEEN UPDATED!!!")
    
    def destroy_window(self):
        self.viewer.destroy_window()

class RvizViewer(object):
    def __init__(self):
        topic_n3_to_rviz=rospy.get_param("topic_n3_to_rviz")
        self.pub = rospy.Publisher(topic_n3_to_rviz, PointCloud2, queue_size=10)
        self.updateView = lambda: None
        self.destroy_window = lambda: None
        self.cloud_XYZaxis = createXYZAxis(coord_axis_length=0.1, num_points_in_axis=50)

    def updateCloud(self, new_cloud):
        try:
            new_cloud = mergeClouds(new_cloud, self.cloud_XYZaxis)
            # new_cloud = resizeCloudXYZ(new_cloud, 5.0) # Resize cloud, so rviz has better view
            self.pub.publish(convertCloudFromOpen3dToRos(new_cloud))
        except:
            print "Node 3 fails to update cloud, due to the input is empty!\n"

def chooseViewer(selection):
    if selection: 
        # This is 1) more difficult to set viewer angle. 
        # 2) cannot set window size. 
        # 3) Much Much Slower to display. Big latency. I don't know why.
        # Thus, Better not use this.
        return Open3DViewer() # open3d
    else:
        return RvizViewer() # rviz

# ---------------------------- Read Transforms from file ----------------------------
def get_transforms_list():
    transform_list = np.array([], dtype=np.float).reshape(0,4)
    matrix = np.array([], dtype=np.float).reshape(0,4)
    with open('/home/acrv/new_ws/src/3D-Scanner-by-Baxter/data/data/camera_pose.txt', 'r') as f:
        for line in f:
            if line.strip() != "":
                row = np.array([float(num) for num in line.split(' ', 3)])
                matrix = np.vstack([matrix, row])
            else:
                # print(matrix)
                transform_list = np.vstack([transform_list, matrix])
                # print(transform_list)
                matrix = np.array([], dtype=np.float).reshape(0,4)
    transform_list = transform_list.reshape(9,4,4)
    return transform_list

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    rospy.init_node("node3")
    num_goalposes = rospy.get_param("num_goalposes")

    # -- Set output filename
    file_folder = rospy.get_param("file_folder") 
    file_name_cloud_final = rospy.get_param("file_name_cloud_final")
    file_name_cloud_src = "src_"

    # -- Subscribe to cloud + Visualize it
    # viewer = chooseViewer(1) # set viewer

    # -- Parameters
    radius_registration=rospy.get_param("~radius_registration") # 0.002
    radius_merge=rospy.get_param("~radius_merge")  # 0.001

    # -- Loop
    rate = rospy.Rate(10) #100
    cnt = 0
    cloud_register = CloudRegister(
        voxel_size_regi=0.01, global_regi_ratio=4.0, 
        voxel_size_output=0.001,
        USE_GLOBAL_REGI=False, USE_ICP=False, USE_COLORED_ICP=True)

    print("\n\n --- Node 3 Started --- \n\n")

    transform_list = get_transforms_list()

    pcds = []

    while not rospy.is_shutdown():

        while cnt < num_goalposes:
            cnt += 1
            rospy.loginfo("=========================================")
            rospy.loginfo("Node 3: Loading the {}th segmented cloud.".format(cnt))

            filename = "src_" + str(cnt) + ".pcd"
            new_cloud = o3d.io.read_point_cloud(file_folder + filename)

            if getCloudSize(new_cloud)==0:
                print(" The received cloud is empty. Not processing it.")
                continue
            
            # Transform
            t_matrix = transform_list[cnt-1]
            new_cloud.transform(t_matrix)
            
            # Filter
            # cl,ind = new_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0) # Can't be too aggressive otherwise it breaks ICP
            # new_cloud = new_cloud.select_down_sample(ind)
            # cl,ind = new_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
            # new_cloud = new_cloud.select_down_sample(ind)

            pcds.append(new_cloud)
            
            # Registration
            res_cloud = cloud_register.addCloud(new_cloud)
            print("Size of the registered cloud: ", getCloudSize(res_cloud))
            
            # Update and save to file
            # viewer.updateCloud(res_cloud)
            # o3d.visualization.draw_geometries(pcds)
            # o3d.visualization.draw_geometries([res_cloud])
            # rospy.sleep(1.0)
        
        rospy.loginfo("=========== Cloud Registration Completed ===========")
        rospy.loginfo("====================================================")
        rospy.sleep(1.0)
        
        # Filter by range to remove things around our target
        # res_cloud = filtCloudByRange(res_cloud, xmin=-OBJECT_RANGE, xmax=OBJECT_RANGE, ymin=-OBJECT_RANGE, ymax=OBJECT_RANGE )

        cl,ind = res_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        res_cloud = res_cloud.select_down_sample(ind)
        # cl,ind = res_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
        # res_cloud = res_cloud.select_down_sample(ind)

        # viewer.updateCloud(res_cloud)
        o3d.visualization.draw_geometries(pcds)

        rospy.loginfo("=========== Cloud Cleaned ===========")
        rospy.loginfo("====================================================")
        rospy.sleep(1.0)

        o3d.visualization.draw_geometries([res_cloud])
        
        # Save resultant point cloud
        if getCloudSize(new_cloud)==0:
            print("The received cloud is empty. Not processing it.")
        else:
            o3d.io.write_point_cloud(file_folder+file_name_cloud_final, res_cloud)

        
        # Update viewer
        # viewer.updateView()

        # Sleep
        rate.sleep()
        input("")

    # -- Node stops
    # viewer.destroy_window()
    rospy.loginfo("!!!!! Node 3 stops.")

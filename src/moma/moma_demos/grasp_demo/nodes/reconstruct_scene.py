#!/usr/bin/env python3

import sys
import numpy as np

from actionlib import SimpleActionServer, SimpleActionClient
from sensor_msgs.msg import Image, PointCloud2
import rospy
import std_srvs.srv
from next_best_view.msg import *

from grasp_demo.msg import ScanSceneAction, ScanSceneResult
from moma_utils.ros.moveit import MoveItClient
from moma_utils.spatial import Transform, Rotation
import vpp_msgs.srv
import vgn.srv
import tf


class ReconstructSceneNode(object):
    """Reconstruct scene moving the camera along a fixed trajectory."""

    def __init__(self, semantic):
        self.moveit = MoveItClient("panda_arm")

        if semantic:
            self.init_gsm_services()
        else:
            self.init_tsdf_services()

        self.captured_image = None

        self.scene_cloud_pub = rospy.Publisher("scene_cloud", PointCloud2, queue_size=1)
        self.map_cloud_pub = rospy.Publisher("map_cloud", PointCloud2, queue_size=1)

        self.image_sub = rospy.Subscriber("/wrist_camera/color/image_raw", Image, self.image_callback)        

        self.action_server = SimpleActionServer(
            "scan_action",
            ScanSceneAction,
            execute_cb=self.reconstruct_scene,
            auto_start=False,
        )
        self.action_server.start()
        rospy.loginfo("Scan action server ready")

    def image_callback(self, data):
        self.captured_image = data

    def init_gsm_services(self):
        self.reset_map = rospy.ServiceProxy(
            "/gsm_node/reset_map",
            std_srvs.srv.Empty,
        )
        self.toggle_integration = rospy.ServiceProxy(
            "/gsm_node/toggle_integration",
            std_srvs.srv.SetBool,
        )
        self.get_scene_cloud = rospy.ServiceProxy(
            "/gsm_node/get_scene_pointcloud",
            vpp_msgs.srv.GetScenePointcloud,
        )
        self.get_map_cloud = rospy.ServiceProxy(
            "/gsm_node/get_map",
            vpp_msgs.srv.GetMap,
        )

    def init_tsdf_services(self):
        self.reset_map = rospy.ServiceProxy(
            "/reset_map",
            std_srvs.srv.Empty,
        )
        self.toggle_integration = rospy.ServiceProxy(
            "/toggle_integration",
            std_srvs.srv.SetBool,
        )
        self.get_scene_cloud = rospy.ServiceProxy(
            "/get_scene_cloud",
            vgn.srv.GetSceneCloud,
        )
        self.get_map_cloud = rospy.ServiceProxy(
            "/get_map_cloud",
            vgn.srv.GetMapCloud,
        )

    def reconstruct_scene(self, goal):
        i = rospy.get_param("moma_demo/workspace")
        scan_joints = rospy.get_param("moma_demo/workspaces")[i]["scan_joints"]

        self.reset_map()
        self.moveit.goto(scan_joints[0], velocity_scaling=0.2)
        

        rospy.loginfo("Nex best view planning")

        # Setup action client for the 'get_next_best_view' action server
        client = SimpleActionClient('get_next_best_view', GetNextBestViewAction)
        client.wait_for_server()

        # Create and send the goal to the action server
        self.toggle_integration(std_srvs.srv.SetBoolRequest(data=True))
        rospy.sleep(1.0)
        
        action_goal = GetNextBestViewGoal()
        action_goal.image = self.captured_image  # Use the captured image as the goal
        client.send_goal(action_goal)
        client.wait_for_result()

        # Get the result from the action server
        result = client.get_result()

        rospy.loginfo("Mapping scene with next best view")
        trans_rot = result.output.data

        self.toggle_integration(std_srvs.srv.SetBoolRequest(data=False))
        
        result = ScanSceneResult()

        msg = self.get_scene_cloud()
        self.scene_cloud_pub.publish(msg.scene_cloud)

        msg = self.get_map_cloud()
        self.map_cloud_pub.publish(msg.map_cloud)
        result.voxel_size = msg.voxel_size
        result.map_cloud = msg.map_cloud

        # TODO vpp map conversion
        # map_cloud = self.get_map_srv().map_cloud
        # data = ros_numpy.numpify(map_cloud)
        # x, y, z = data["x"], data["y"], data["z"]
        # points = np.column_stack((x, y, z)) - self.T_base_task.translation
        # d = (data["distance"] + 0.03) / 0.06  # scale to [0, 1]
        # tsdf_grid = np.zeros((40, 40, 40), dtype=np.float32)
        # for idx, point in enumerate(points):
        #     if np.all(point > 0.0) and np.all(point < 0.3):
        #         i, j, k = np.floor(point / voxel_size).astype(int)
        #         tsdf_grid[i, j, k] = d[idx]
        # points, distances = grid_to_map_cloud(voxel_size, tsdf_grid)
        # self.vis.map_cloud(self.task_frame_id, points, distances)

        self.action_server.set_succeeded(result)
        rospy.loginfo("Scan scene action succeeded")


def main():
    rospy.init_node("scan_action_node")
    semantic = sys.argv[1] in ["True", "true"]
    ReconstructSceneNode(semantic)
    rospy.spin()


if __name__ == "__main__":
    main()

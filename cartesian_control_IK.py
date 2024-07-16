# Cartesian Control & IK

import rclpy
import math
import numpy as np
import tf2_ros
import transforms3d
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

def quaternion_from_euler(ai, aj, ak):
	ai /= 2.0
	aj /= 2.0
	ak /= 2.0
	ci = math.cos(ai)
	si = math.sin(ai)
	cj = math.cos(aj)
	sj = math.sin(aj)
	ck = math.cos(ak)
	sk = math.sin(ak)
	cc = ci*ck
	cs = ci*sk
	sc = si*ck
	ss = si*sk

	q = np.empty((4, ))
	q[0] = cj*sc - sj*cs
	q[1] = cj*ss + sj*cc
	q[2] = cj*cs - sj*sc
	q[3] = cj*cc + sj*ss

	return q


class FramePublisher(Node):
    
    def __init__(self):
        super().__init__('tf2_frame_publisher')
        self.tf_publisher = self.create_publisher(TransformStamped, 'topic', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        timer_period = 0.1
        self.timerBO = self.create_timer(timer_period, self.base_object_broadcast_transform)
        self.timerBR = self.create_timer(timer_period, self.base_robot_broadcast_transform)
        self.timerRC = self.create_timer(timer_period, self.robot_camera_broadcast_transform)
        self.i = 0
    
    # Transformation from 'base_frame' to 'object_frame'
    def base_object_broadcast_transform(self):

        # Transformation: Parent_frame to child_frame
        BO_transform_stamped = TransformStamped()
        BO_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        BO_transform_stamped.header.frame_id = 'base_frame'
        BO_transform_stamped.child_frame_id = 'object_frame'

	# Homogeneous Transformation
        T_xy = transforms3d.affines.compose((1.5, 0.8, 0.0), (transforms3d.euler.euler2mat(0.0, 0.0, 0.0, axes='rxyz')), (1,1,1))
        R_rp = transforms3d.affines.compose((0.0, 0.0, 0.0), (transforms3d.euler.euler2mat(0.64, 0.64, 0.0, axes='rxyz')), (1,1,1))
        H_BO = np.dot(R_rp, T_xy)
        T, R, Z, S = transforms3d.affines.decompose(H_BO)
        ax, ay, az = transforms3d.euler.mat2euler(R, 'rxyz')

        # Translation
        BO_transform_stamped.transform.translation.x = T[0]
        BO_transform_stamped.transform.translation.y = T[1]
        BO_transform_stamped.transform.translation.z = T[2]

        # Rotation using Roll-Pitch-Yaw
        q = quaternion_from_euler(ax, ay, az)
        BO_transform_stamped.transform.rotation.x = q[0]
        BO_transform_stamped.transform.rotation.y = q[1]
        BO_transform_stamped.transform.rotation.z = q[2]
        BO_transform_stamped.transform.rotation.w = q[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(BO_transform_stamped)

    # Transformation from 'base_frame' to 'robot_frame'
    def base_robot_broadcast_transform(self):
    
        # Transformation: Parent_frame to child_frame
        BR_transform_stamped = TransformStamped()
        BR_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        BR_transform_stamped.header.frame_id = 'base_frame'
        BR_transform_stamped.child_frame_id = 'robot_frame'

	# Homogeneous Transformation        
        T_z = transforms3d.affines.compose((0.0, 0.0, -2.0), (transforms3d.euler.euler2mat(0.0, 0.0, 0.0, axes='rxyz')), (1,1,1))
        R_p = transforms3d.affines.compose((0.0, 0.0, 0.0), (transforms3d.euler.euler2mat(0.0, 1.5, 0.0, axes='rxyz')), (1,1,1))
        H_BR = np.dot(R_p, T_z)
        T, R, Z, S = transforms3d.affines.decompose(H_BR)
        ax, ay, az = transforms3d.euler.mat2euler(R, 'rxyz')

        # Translation
        BR_transform_stamped.transform.translation.x = T[0]
        BR_transform_stamped.transform.translation.y = T[1]
        BR_transform_stamped.transform.translation.z = T[2]

        # Rotation using Roll-Pitch-Yaw
        q = quaternion_from_euler(ax, ay, az)
        BR_transform_stamped.transform.rotation.x = q[0]
        BR_transform_stamped.transform.rotation.y = q[1]
        BR_transform_stamped.transform.rotation.z = q[2]
        BR_transform_stamped.transform.rotation.w = q[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(BR_transform_stamped)
     
    # Transformation from 'robot_frame' to 'camera_frame'
    def robot_camera_broadcast_transform(self):
    
        # Transformation: Parent_frame to child_frame
        RC_transform_stamped = TransformStamped()
        RC_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        RC_transform_stamped.header.frame_id = 'robot_frame'
        RC_transform_stamped.child_frame_id = 'camera_frame'

	# Homogeneous Transformation
	# H_BO Transformation
        T_xy = transforms3d.affines.compose((1.5, 0.8, 0.0), (transforms3d.euler.euler2mat(0.0, 0.0, 0.0, axes='rxyz')), (1,1,1))
        R_rp = transforms3d.affines.compose((0.0, 0.0, 0.0), (transforms3d.euler.euler2mat(0.64, 0.64, 0.0, axes='rxyz')), (1,1,1))
        H_BO = np.dot(R_rp, T_xy)
        # H_BR Transformation
        T_z = transforms3d.affines.compose((0.0, 0.0, -2.0), (transforms3d.euler.euler2mat(0.0, 0.0, 0.0, axes='rxyz')), (1,1,1))
        R_p = transforms3d.affines.compose((0.0, 0.0, 0.0), (transforms3d.euler.euler2mat(0.0, 1.5, 0.0, axes='rxyz')), (1,1,1))
        H_BR = np.dot(R_p, T_z)
        # H_RC Transformation           
        ObjectPoint_in_BF = np.dot(H_BO, [0.0, 0.0, 0.0, 1.0]) 
        ObjectPoint_in_RF = np.dot(np.linalg.inv(H_BR), ObjectPoint_in_BF)
        ObjectPoint_in_RF = np.array([ObjectPoint_in_RF[0], ObjectPoint_in_RF[1], ObjectPoint_in_RF[2]])
        Camera_in_RF = np.array([0.3, 0.0, 0.3])
        vector_CO = ObjectPoint_in_RF - Camera_in_RF
        unit_vector_CO = vector_CO / np.linalg.norm(vector_CO)
        dot_product = np.dot([1,0,0], unit_vector_CO)
        if -1 < dot_product < 1:
          angle = np.arccos(dot_product)
          rot_axis = np.cross([1,0,0], unit_vector_CO)
          R = transforms3d.axangles.axangle2mat(rot_axis, angle)
        else:
          R = transforms3d.axangles.axangle2mat([1,0,0], 0)
        ax, ay, az = transforms3d.euler.mat2euler(R)
        T= np.array([0.3, 0.0, 0.3])

        # Translation
        RC_transform_stamped.transform.translation.x = T[0]
        RC_transform_stamped.transform.translation.y = T[1]
        RC_transform_stamped.transform.translation.z = T[2]

        # Rotation using Roll-Pitch-Yaw
        q = quaternion_from_euler(ax, ay, az)
        RC_transform_stamped.transform.rotation.x = q[0]
        RC_transform_stamped.transform.rotation.y = q[1]
        RC_transform_stamped.transform.rotation.z = q[2]
        RC_transform_stamped.transform.rotation.w = q[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(RC_transform_stamped)
    
      
def main():
    rclpy.init()
    node = FramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

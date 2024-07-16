#!/usr/bin/env python3

# Cartesian Control & IK

import math
import numpy
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from custom_msg.msg import CartesianCommand
from urdf_parser_py.urdf import URDF
import random
import transforms3d
import transforms3d._gohlketransforms as tf
from threading import Thread, Lock

'''This is a class which will perform both cartesian control and inverse
   kinematics'''
class CCIK(Node):
    def __init__(self):
        super().__init__('ccik')
    #Load robot from parameter server
        # self.robot = URDF.from_parameter_server()
        self.declare_parameter(
            'rd_file', rclpy.Parameter.Type.STRING)
        robot_desription = self.get_parameter('rd_file').value
        with open(robot_desription, 'r') as file:
            robot_desription_text = file.read()
        # print(robot_desription_text)
        self.robot = URDF.from_xml_string(robot_desription_text)

    #Subscribe to current joint state of the robot
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.get_joint_state, 10)

    #This will load information about the joints of the robot
        self.num_joints = 0
        self.joint_names = []
        self.q_current = []
        self.joint_axes = []
        self.get_joint_info()

    #This is a mutex
        self.mutex = Lock()

    #Subscribers and publishers for for cartesian control
        self.cartesian_command_sub = self.create_subscription(
            CartesianCommand, '/cartesian_command', self.get_cartesian_command, 10)
        self.velocity_pub = self.create_publisher(JointState, '/joint_velocities', 10)
        self.joint_velocity_msg = JointState()

        #Subscribers and publishers for numerical IK
        self.ik_command_sub = self.create_subscription(
            Transform, '/ik_command', self.get_ik_command, 10)
        self.joint_command_pub = self.create_publisher(JointState, '/joint_command', 10)
        self.joint_command_msg = JointState()

    '''This is a function which will collect information about the robot which
       has been loaded from the parameter server. It will populate the variables
       self.num_joints (the number of joints), self.joint_names and
       self.joint_axes (the axes around which the joints rotate)'''
    def get_joint_info(self):
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link
        #self.get_logger().info(f'Joint names {self.joint_names}')
        #self.get_logger().info(f'Joint axes {self.joint_axes}')

    '''This is the callback which will be executed when the cartesian control
       recieves a new command. The command will contain information about the
       secondary objective and the target q0. At the end of this callback, 
       you should publish to the /joint_velocities topic.'''
    def get_cartesian_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        # Extract data from the Cartesian command x_tagret message
        joint_transforms, b_T_ee = self.forward_kinematics(self.q_current)
        x_trans_t = command.x_target.translation.x
        y_trans_t = command.x_target.translation.y
        z_trans_t = command.x_target.translation.z
        x_rot_t = command.x_target.rotation.x
        y_rot_t = command.x_target.rotation.y
        z_rot_t = command.x_target.rotation.z
        w_rot_t = command.x_target.rotation.w
        R_target = transforms3d.quaternions.quat2mat([w_rot_t,x_rot_t,y_rot_t,z_rot_t])
        T_target = numpy.identity(4)
        T_target[0:3,0:3] = R_target
        T_target[0,3] = x_trans_t
        T_target[1,3] = y_trans_t
        T_target[2,3] = z_trans_t
        T_ee_target = numpy.linalg.inv(b_T_ee)@T_target
        angle_t, axis_t = self.rotation_from_matrix(T_ee_target)
        x_ee_trans_t = T_ee_target[0:3,3]
        x_ee_rot_t = angle_t*axis_t[0]
        y_ee_rot_t = angle_t*axis_t[1]
        z_ee_rot_t = angle_t*axis_t[2]
        x_target = numpy.array([x_ee_trans_t[0],x_ee_trans_t[1],x_ee_trans_t[2],x_ee_rot_t,y_ee_rot_t,z_ee_rot_t]).reshape(-1,1)
        
        # Extract data from the Cartesian command secondary objective message
        secondary_objective = command.secondary_objective
        q0_target = command.q0_target

        # Define gains
        p = 1.0                 # Proportional gain
        secondary_p = 3.0       # Secondary objective gain

        # Calculate the desired delta x
        delta_x = p*x_target

        # Calculate joint velocities using the pseudoinverse of the Jacobian
        J = self.get_jacobian(b_T_ee, joint_transforms)
        J_pseudoinv = numpy.linalg.pinv(J, rcond= 1.0e-2)
        delta_q = numpy.dot(J_pseudoinv, delta_x)

        # Secondary objective
        if secondary_objective == True:
            J_pseudoinv_actual = numpy.linalg.pinv(J)
            q0 = self.q_current[0]
            r0 = q0_target
            q_secondary = numpy.zeros((self.num_joints,1))
            q_secondary[0] = secondary_p*(r0-q0)
            delta_q_secondary = numpy.dot((numpy.identity(self.num_joints)-J_pseudoinv_actual@J),q_secondary)
            delta_q += delta_q_secondary

        # Publish the joint velocities
        self.joint_velocity_msg.name = self.joint_names
        self.joint_velocity_msg.velocity = numpy.array(delta_q).flatten().tolist()
        self.velocity_pub.publish(self.joint_velocity_msg)
        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This is a function which will assemble the jacobian of the robot using the
       current joint transforms and the transform from the base to the end
       effector (b_T_ee). Both the cartesian control callback and the
       inverse kinematics callback will make use of this function.
       Usage: J = self.get_jacobian(b_T_ee, joint_transforms)'''
    def get_jacobian(self, b_T_ee, joint_transforms):
        J = numpy.zeros((6,self.num_joints))
        #--------------------------------------------------------------------------
        for i in range(self.num_joints):
            joint_transform_j_ee = numpy.dot(numpy.linalg.inv(joint_transforms[i]), b_T_ee)
            joint_transform_ee_j = numpy.linalg.inv(joint_transform_j_ee)
            trans_j_ee = joint_transform_j_ee[0:3,3]
            rot_ee_j = joint_transform_ee_j[0:3,0:3]
            skew_trans_j_ee = numpy.array([[0, -trans_j_ee[2], trans_j_ee[1]],
                                          [trans_j_ee[2], 0, -trans_j_ee[0]],
                                          [-trans_j_ee[1], trans_j_ee[0], 0]])
            V_j_trans = -numpy.dot(rot_ee_j, skew_trans_j_ee)
            V_j = numpy.zeros((6,6))
            V_j[0:3,0:3] = rot_ee_j
            V_j[0:3,3:6] = V_j_trans
            V_j[3:6,0:3] = 0
            V_j[3:6,3:6] = rot_ee_j
            J[:,i] = V_j[:, 5] 
        #--------------------------------------------------------------------------
        return J

    '''This is the callback which will be executed when the inverse kinematics
       recieve a new command. The command will contain information about desired
       end effector pose relative to the root of your robot. At the end of this
       callback, you should publish to the /joint_command topic. This should not
       search for a solution indefinitely - there should be a time limit. When
       searching for two matrices which are the same, we expect numerical
       precision of 10e-3.'''
    def get_ik_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        # Extract the desired end-effector pose from the IK command message
        x_trans_t = command.translation.x
        y_trans_t = command.translation.y
        z_trans_t = command.translation.z
        x_rot_t = command.rotation.x
        y_rot_t = command.rotation.y
        z_rot_t = command.rotation.z
        x_target = numpy.array([x_trans_t,y_trans_t,z_trans_t,x_rot_t,y_rot_t,z_rot_t]).reshape(-1,1)

        # Initialize variables
        max_attempts = 3
        timeout = 10       # Timeout in seconds
        solution = False
        threshold = 1e-3
       
        for k in range(max_attempts):
                 
            # Calculate the random current configuration
            q_vec_current = numpy.zeros([self.num_joints,1])
            for i in range(self.num_joints):
                q_random = random.uniform(-math.pi, math.pi)
                q_vec_current[i] = q_random
            q_current = numpy.array(q_vec_current).reshape(-1,1)

            start_time = time.time()
            while time.time()-start_time < timeout:

                # Calculate current end-effector random pose
                joint_transforms, b_T_ee = self.forward_kinematics(q_current)
                angle, axis = self.rotation_from_matrix(b_T_ee)
                x_trans_c = b_T_ee[0,3]
                y_trans_c = b_T_ee[1,3]
                z_trans_c = b_T_ee[2,3]
                x_rot_c = angle*axis[0]
                y_rot_c = angle*axis[1]
                z_rot_c = angle*axis[2]
                x_current = numpy.array([x_trans_c,y_trans_c,z_trans_c,x_rot_c,y_rot_c,z_rot_c]).reshape(-1,1)
            
                # Calculate the error in the end-effector pose
                ee_error = x_target-x_current
                error = numpy.linalg.norm(ee_error)

                # Calculate the Jacobian
                J = self.get_jacobian(b_T_ee, joint_transforms)
                J_pseudoinv = numpy.linalg.pinv(J)

                # Calculate joint positions using the IK control law
                delta_q = numpy.dot(J_pseudoinv, ee_error)

                # Update joint positions
                q_current += delta_q
                
                # Check if the desired pose is achieved within tolerance
                if error<threshold:
                    solution = True
                    break

            if solution == True:
                break
        
        if solution == True:
            # Publish the joint velocities
            self.joint_command_msg.name = self.joint_names
            self.joint_command_msg.position = q_current.flatten().tolist()
            self.joint_command_pub.publish(self.joint_command_msg)
        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This function will return the angle-axis representation of the rotation
       contained in the input matrix. Use like this: 
       angle, axis = rotation_from_matrix(R)'''
    def rotation_from_matrix(self, matrix):
        R = numpy.array(matrix, dtype=numpy.float64, copy=False)
        R33 = R[:3, :3]
        # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, W = numpy.linalg.eig(R33.T)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = numpy.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, Q = numpy.linalg.eig(R)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        # rotation angle depending on axis
        cosa = (numpy.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return angle, axis

    '''This is the function which will perform forward kinematics for your 
       cartesian control and inverse kinematics functions. It takes as input
       joint values for the robot and will return an array of 4x4 transforms
       from the base to each link of the robot, as well as the transform from
       the base to the end effector.
       Usage: joint_transforms, b_T_ee = self.forward_kinematics(joint_values)'''
    def forward_kinematics(self, joint_values):
        joint_transforms = []

        link = self.robot.get_root()
        T = tf.identity_matrix()

        while True:
            if link not in self.robot.child_map:
                break

            (joint_name, next_link) = self.robot.child_map[link][0]
            joint = self.robot.joint_map[joint_name]

            T_l = numpy.dot(tf.translation_matrix(joint.origin.xyz), tf.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2], 'rxyz'))
            T = numpy.dot(T, T_l)

            if joint.type != "fixed":
                joint_transforms.append(T)
                q_index = self.joint_names.index(joint_name)
                T_j = tf.rotation_matrix(joint_values[q_index], numpy.asarray(joint.axis))
                T = numpy.dot(T, T_j)

            link = next_link
        return joint_transforms, T #where T = b_T_ee

    '''This is the callback which will recieve and store the current robot
       joint states.'''
    def get_joint_state(self, msg):
        self.mutex.acquire()
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])
        self.mutex.release()


def main(args = None):
    rclpy.init()
    ccik = CCIK()
    rclpy.spin(ccik)
    ccik.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


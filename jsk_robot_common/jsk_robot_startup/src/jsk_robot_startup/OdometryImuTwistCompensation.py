#! /usr/bin/env python

import rospy
import numpy
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3, TwistWithCovarianceStamped, TwistWithCovariance, PoseWithCovarianceStamped, Pose
import tf
import time
import threading
import copy
from odometry_utils import make_homogeneous_matrix, update_twist_covariance, update_pose, update_pose_covariance, broadcast_transform, fuse_pose_distribution, transform_twist, transform_twist_covariance

class OdometryImuTwistCompensation(object):
    def __init__(self):
        rospy.init_node("OdometryImuTwistCompensation", anonymous=True)
        self.odom_frame = rospy.get_param("~odom_frame", "compensated_odom")
        self.base_link_frame = rospy.get_param("~base_link_frame", "BODY")
        self.publish_tf = rospy.get_param("~publish_tf", True)
        if self.publish_tf:
            self.broadcast = tf.TransformBroadcaster()
            self.invert_tf = rospy.get_param("~invert_tf", True)
        self.rate = float(rospy.get_param("~rate", 100))
        self.r = rospy.Rate(self.rate)        
        self.lock = threading.Lock()
        self.imu_odom = None
        self.odom = None
        self.pub_odom = None
        self.imu_twist_sub = rospy.Subscriber("~imu_twist", TwistWithCovarianceStamped, self.imu_twist_callback)
        self.odom_sub = rospy.Subscriber("~input_odom", Odometry, self.odom_callback)
        self.pub = rospy.Publisher("~output", Odometry, queue_size = 1)
        self.init_signal_sub = rospy.Subscriber("~init_signal", Empty, self.init_signal_callback)

    def init_signal_callback(self):
        with self.lock:
            self.imu_odom = None
            self.odom = None
            self.pub_odom = None

    def init_imu_odom(self, trans, rot, cov, stamp):
        self.imu_odom = Odometry()
        self.imu_odom.header.stamp = stamp
        self.imu_odom.header.frame_id = self.base_link_frame
        self.imu_odom.pose.pose = Pose(Point(*trans), Quaternion(*rot))
        self.imu_odom.pose.covariance = cov
        self.imu_odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
        self.imu_odom.twist.covariance = numpy.diag([0.001**2]*6).reshape(-1,).tolist()

    def imu_twist_callback(self, msg):
        with self.lock:
            if self.odom == None:
                return
            if self.imu_odom == None:
                self.init_imu_odom([getattr(self.odom.pose.pose.position, attr) for attr in ["x", "y", "z"]],
                                   [getattr(self.odom.pose.pose.orientation, attr) for attr in ["x", "y", "z", "w"]],
                                   self.odom.pose.covariance, self.odom.header.stamp)
            global_twist_with_cov = TwistWithCovariance(transform_twist(self.imu_odom.pose.pose, msg.twist.twist, to_global = True),
                                                        transform_twist_covariance(self.imu_odom.pose.pose, msg.twist.covariance, to_global = True))
            dt = (msg.header.stamp - self.imu_odom.header.stamp).to_sec()
            # update imu_odom
            self.imu_odom.pose.pose = update_pose(self.imu_odom.pose.pose, global_twist_with_cov.twist, dt)
            self.imu_odom.pose.covariance = update_pose_covariance(self.imu_odom.pose.covariance, global_twist_with_cov.covariance, dt)
            self.imu_odom.twist = msg.twist
            self.imu_odom.header.stamp = msg.header.stamp
            
    def odom_callback(self, msg):
        with self.lock:
            if self.odom == None:
                self.odom = msg
            if self.imu_odom == None:
                self.pub.publish(msg) # pass through msg
                if self.publish_tf:
                    broadcast_transform(self.broadcast, msg, self.invert_tf)
            else:
                if self.pub_odom == None:
                    self.pub_odom = msg
                    self.pub_odom.header.frame_id = self.odom_frame
                    self.pub_odom.child_frame_id = self.base_link_frame
                else:
                    dt = (msg.header.stamp - self.odom.header.stamp).to_sec()
                    # update odom
                    global_twist_with_cov = TwistWithCovariance(transform_twist(msg.pose.pose, msg.twist.twist, to_global = True),
                                                                transform_twist_covariance(msg.pose.pose, msg.twist.covariance, to_global = True))
                    self.odom.pose.pose = update_pose(self.odom.pose.pose, global_twist_with_cov.twist, dt)
                    self.odom.pose.covariance = update_pose_covariance(self.odom.pose.covariance, global_twist_with_cov.covariance, dt)
                    self.odom.twist.twist = transform_twist(self.odom.pose.pose, global_twist_with_cov.twist, to_global = False)
                    self.odom.twist.covariance = transform_twist_covariance(self.odom.pose.pose, global_twist_with_cov.covariance, to_global = False)
                    self.odom.header.stamp = msg.header.stamp
                    
                    # fuse imu integrated pose and raw odom
                    new_pose, new_cov = fuse_pose_distribution(self.odom.pose, self.imu_odom.pose)
                    self.odom.pose.pose = Pose(Point(*new_pose[0:3]), Quaternion(*tf.transformations.quaternion_from_euler(*new_pose[3:6])))
                    self.odom.pose.covariance = new_cov
                    self.odom.twist.twist = transform_twist(self.odom.pose.pose, global_twist_with_cov.twist, to_global = False)
                    self.odom.twist.covariance = transform_twist_covariance(self.odom.pose.pose, global_twist_with_cov.covariance, to_global = False)
            
                    # initialize imu_odom by current estimated odometry
                    self.init_imu_odom([getattr(self.odom.pose.pose.position, attr) for attr in ["x", "y", "z"]],
                                       [getattr(self.odom.pose.pose.orientation, attr) for attr in ["x", "y", "z", "w"]],
                                       self.odom.pose.covariance, self.odom.header.stamp)
                    
                    # publish odometry
                    self.pub.publish(self.odom)
                    if self.publish_tf:
                        broadcast_transform(self.broadcast, self.odom, self.invert_tf)

    # def calculate_transformation(prev_odom, current_odom):
    #     prev_homogeneous_matrix = make_homogeneous_matrix([getattr(prev_odom.pose.pose.position, attr) for attr in ["x", "y", "z"]],
    #                                                       [getattr(prev_odom.pose.pose.orientation, attr) for attr in ["x", "y", "z", "w"]])
    #     current_homogeneous_matrix = make_homogeneous_matrix([getattr(current_odom.pose.pose.position, attr) for attr in ["x", "y", "z"]],
    #                                                          [getattr(current_odom.pose.pose.orientation, attr) for attr in ["x", "y", "z", "w"]])
    #     # Hnew = Hold * T -> T = Hold^-1 * Hnew
    #     return numpy.dot(numpy.linalg.inv(prev_homogeneous_matrix), numpy.linalg.inv(current_homogeneous_matrix))

    # def apply_transform_to_odom(prev_odom, transform_homogeneous_matrix):
    #     prev_homogeneous_matrix = make_homogeneous_matrix([getattr(prev_odom.pose.pose.position, attr) for attr in ["x", "y", "z"]],
    #                                                       [getattr(prev_odom.pose.pose.orientation, attr) for attr in ["x", "y", "z", "w"]])
    #     new_homogeneous_matrix = numpy.dot(prev_homogeneous_matrix, transform_homogeneous_matrix)
    #     new_odom = copy.deepcopy(prev_odom)
    #     new_odom.pose.pose.position = Point(*list(new_homogeneous_matrix[:3, 3]))
    #     new_odom.pose.pose.orientation = Quaternion(*list(tf.transformations.quaternion_from_matrix(new_homogeneous_matrix)))
    #     # todo: calculate twist
    #     return new_odom
    
    def execute(self):
        while not rospy.is_shutdown():
            self.r.sleep()

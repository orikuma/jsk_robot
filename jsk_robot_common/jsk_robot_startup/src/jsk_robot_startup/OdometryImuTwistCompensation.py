#! /usr/bin/env python

import rospy
import numpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3, TwistWithCovarianceStamped, TwistWithCovariance, PoseWithCovarianceStamped, Pose
import tf
import time
import threading
import copy
from odometry_utils import make_homogeneous_matrix, update_twist_covariance, update_pose, update_pose_covariance, broadcast_transform, transform_local_twist_to_global, transform_local_twist_covariance_to_global, fuse_pose_distribution

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
        self.imu_pose = None
        self.odom = None
        self.imu_twist_sub = rospy.Subscriber("~imu_twist", TwistWithCovarianceStamped, self.imu_twist_callback)
        self.odom_sub = rospy.Subscriber("~input_odom", Odometry, self.odom_callback)
        self.pub = rospy.Publisher("~output", Odometry, queue_size = 1)

    def init_imu_pose(self, trans, rot, cov, stamp):
        self.imu_pose = PoseWithCovarianceStamped()
        self.imu_pose.header.stamp = stamp
        self.imu_pose.header.frame_id = self.base_link_frame
        self.imu_pose.pose.pose = Pose(Point(*trans), Quaternion(*rot))
        self.imu_pose.pose.covariance = cov

    def init_odometry(self, init_odom):
        self.odom = init_odom
        
    def imu_twist_callback(self, msg):
        with self.lock:
            if self.odom == None or self.imu_pose == None:
                return
            global_twist_with_cov = TwistWithCovariance(transform_local_twist_to_global(self.odom.pose.pose, msg.twist.twist),
                                                        transform_local_twist_covariance_to_global(self.odom.pose.pose, msg.twist.covariance))
            dt = (msg.header.stamp - self.imu_pose.header.stamp).to_sec()
            # update imu_pose
            self.imu_pose.pose.pose = update_pose(self.imu_pose.pose.pose, global_twist_with_cov.twist, dt)
            self.imu_pose.pose.covariance = update_pose_covariance(self.imu_pose.pose.covariance, global_twist_with_cov.covariance, dt)
            self.imu_pose.header.stamp = msg.header.stamp
            # update odom temporary
            self.odom.pose.pose = update_pose(self.odom.pose.pose, global_twist_with_cov.twist, dt)
            self.odom.pose.covariance = update_pose_covariance(self.odom.pose.covariance, global_twist_with_cov.covariance, dt)
            
    def odom_callback(self, msg):
        with self.lock:
            if self.odom == None:
                self.init_odometry(msg)
            if self.imu_pose == None:
                self.init_imu_pose([getattr(self.odom.pose.pose.position, attr) for attr in ["x", "y", "z"]],
                                   [getattr(self.odom.pose.pose.orientation, attr) for attr in ["x", "y", "z", "w"]],
                                   self.odom.pose.covariance, self.odom.header.stamp)
            dt = (msg.header.stamp - self.odom.header.stamp)

            # fuse imu integrated pose and raw odom
            new_pose, new_cov = fuse_pose_distribution(msg.pose, self.imu_pose.pose)
            self.odom.pose.pose = Pose(Point(*new_pose[0:3]), Quaternion(*tf.transformations.quaternion_from_euler(*new_pose[3:6])))
            self.odom.pose.covariance = new_cov
            self.odom.header.stamp = msg.header.stamp
            
            # initialize imu_pose by current estimated odometry
            self.init_imu_pose([getattr(self.odom.pose.pose.position, attr) for attr in ["x", "y", "z"]],
                               [getattr(self.odom.pose.pose.orientation, attr) for attr in ["x", "y", "z", "w"]],
                               self.odom.pose.covariance, self.odom.header.stamp)

            # publish odometry
            self.pub.publish(self.odom)
            if self.publish_tf:
                broadcast_transform(self.broadcast, self.odom, self.invert_tf)
            
    def execute(self):
        while not rospy.is_shutdown():
            self.r.sleep()

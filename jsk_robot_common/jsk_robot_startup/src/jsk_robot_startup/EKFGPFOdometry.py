#! /usr/bin/env python

import rospy
import numpy
import scipy.stats
import random
import threading
import itertools
import tf
import time
from operator import itemgetter 

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, Twist, Pose, Point, Quaternion, Vector3, TransformStamped
from ParticleOdometry import ParticleOdometry
from odometry_utils import norm_pdf_multivariate, transform_quaternion_to_euler, transform_local_twist_to_global, transform_local_twist_covariance_to_global, update_pose, update_pose_covariance, broadcast_transform

class EKFGPFOdometry(ParticleOdometry):
    def __init__(self):
        ParticleOdometry.__init__(self)

    def initialize_odometry(self, trans, rot):
        with self.lock:
            self.particles = None
            self.weights = []
            self.odom = Odometry()
            self.odom.pose.pose = Pose(Point(*trans), Quaternion(*rot))
            self.odom.pose.covariance = numpy.diag([x ** 2 for x in self.init_sigma])
            self.odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
            self.odom.twist.covariance = numpy.diag([0.001**2]*6).reshape(-1,).tolist()
            self.odom.header.stamp = rospy.Time.now()
            self.odom.header.frame_id = self.odom_frame
            self.odom.child_frame_id = self.base_link_frame
            self.source_odom = None
            self.measure_odom = None
            self.measurement_updated = False
            self.imu = None
            self.imu_rotation = None
            self.prev_rpy = None
            
    ## top odometry calculations 
    def calc_odometry(self):
        self.ekf_update(self.odom, self.source_odom)
        if self.measurement_updated:
            # sampling and weighting
            self.particles, self.weights = self.sampling_and_weighting(self.odom.pose, self.min_weight)
            # resampling
            self.particles = self.resampling(self.particles, self.weights)
            # estimate new pdf 
            self.approximate_odometry(self.particles, self.weights)
            # wait next measurement
            self.measurement_updated = False

    # EKF
    def ekf_update(self, current_odom, source_odom):
        dt = (source_odom.header.stamp - current_odom.header.stamp).to_sec()
        new_pose_with_covariance = self.update_pose_with_covariance(current_odom.pose, source_odom.twist, dt)
        # update odom (only pose)
        self.odom.pose = new_pose_with_covariance
                    
    ## particle filter functions
    # input: EKF result (current_pose_with_covariance), min_weight(float) output: list of particles and weights
    # sampling and weighting at the same time to reduce computational cost in loop calculation
    def sampling_and_weighting(self, current_pose_with_covariance, min_weight):
        pose_mean = self.convert_pose_to_list(current_pose_with_covariance.pose) 
        pose_cov_matrix = zip(*[iter(current_pose_with_covariance.covariance)]*6)
        weights = []
        particles = []
        if not self.measure_odom:
            rospy.logwarn("[%s] measurement does not exist.", rospy.get_name())
            particles = [self.convert_list_to_pose(x) for x in numpy.random.multivariate_normal(pose_mean, pose_cov_matrix, int(self.particle_num)).tolist()] # random sampling from ekf result distribution
            weights = [1.0 / self.particle_num] * int(self.particle_num) # use uniform weights when measure_odom has not been subscribed yet
        else:
            measure_to_source_dt = (self.source_odom.header.stamp - self.measure_odom.header.stamp).to_sec() # adjust timestamp of pose in measure_odom to source_odom
            current_measure_pose_with_covariance = self.update_pose_with_covariance(self.measure_odom.pose, self.measure_odom.twist, measure_to_source_dt) # assuming dt is small and measure_odom.twist is do not change in dt
            measurement_pose_array = numpy.array(self.convert_pose_to_list(current_measure_pose_with_covariance.pose))
            try:
                measurement_cov_matrix_inv = numpy.linalg.inv(numpy.matrix(current_measure_pose_with_covariance.covariance).reshape(6, 6)) # calculate inverse matrix first to reduce computation cost
                for i in range(int(self.particle_num)):
                    prt = self.convert_list_to_pose(numpy.random.multivariate_normal(pose_mean, pose_cov_matrix))
                    w = max(min_weight, self.calculate_weighting_likelihood(prt, measurement_pose_array, measurement_cov_matrix_inv))
                    particles.append(prt)
                    weights.append(w)
            except numpy.linalg.LinAlgError:
                rospy.logwarn("[%s] covariance matrix is not singular.", rospy.get_name())
                particles = [self.convert_list_to_pose(x) for x in numpy.random.multivariate_normal(pose_mean, pose_cov_matrix, int(self.particle_num)).tolist()] # random sampling from ekf result distribution
                weights = [min_weight] * len(particles)
            if all([x == min_weight for x in weights]):
                rospy.logwarn("[%s] likelihood is too small and all weights are limited by min_weight.", rospy.get_name())
            normalization_coeffs = sum(weights) # normalization and each weight is assumed to be larger than 0
            weights = [w / normalization_coeffs for w in weights]            
        return particles, weights

    def approximate_odometry(self, particles, weights):
        # use only important particels
        combined_prt_weight = zip(self.particles, self.weights)
        selected_prt_weight = zip(*sorted(combined_prt_weight, key = itemgetter(1), reverse = True)[:int(self.valid_particle_num)]) # [(p0, w0), (p1, w1), ..., (pN, wN)] -> [(sorted_p0, sorted_w0), (sorted_p1, sorted_w1), ..., (sorted_pN', sorted_wN')] -> [(sorted_p0, ..., sorted_pN'), (sorted_w0, ..., sorted_wN')]
        # estimate gaussian distribution for Odometry msg 
        mean, cov = self.guess_normal_distribution(selected_prt_weight[0], selected_prt_weight[1])
        # overwrite pose pdf
        self.odom.pose.pose = self.convert_list_to_pose(mean)
        self.odom.pose.covariance = list(itertools.chain(*cov))

    def publish_odometry(self):
        # refrect source_odom informations
        self.odom.header.stamp = self.source_odom.header.stamp
        self.odom.twist = self.source_odom.twist
        self.pub.publish(self.odom)
        if self.publish_tf:
            broadcast_transform(self.broadcast, self.odom, self.invert_tf)
        # update prev_rpy to prevent jump of angles
        self.prev_rpy = transform_quaternion_to_euler([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w], self.prev_rpy)
        
    # main functions
    def update(self):
        if not self.odom or not self.source_odom:
            rospy.logwarn("[%s]: odometry is not initialized", rospy.get_name())
            return
        else:
            self.calc_odometry()
            self.publish_odometry()
            if self.publish_histogram:
                histgram_msg = self.make_histogram_array(self.particles, self.source_odom.header.stamp)
                self.pub_hist.publish(histgram_msg)

    def execute(self):
        while not rospy.is_shutdown():
            with self.lock:
                self.update() # call update() when control input is subscribed
            self.r.sleep()

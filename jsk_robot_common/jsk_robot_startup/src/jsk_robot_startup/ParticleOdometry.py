#! /usr/bin/env python

import rospy
import numpy
import scipy.stats
import math
import random
import threading
import itertools
import tf
import time
import copy

from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, Twist, Pose, Point, Quaternion, Vector3

# scipy.stats.multivariate_normal only can be used after SciPy 0.14.0
# input: x(array), mean(array), cov_inv(matrix) output: probability of x
# covariance has to be inverted to reduce calculation time
def norm_pdf_multivariate(x, mean, cov_inv):
    size = len(x)
    if size == len(mean) and (size, size) == cov_inv.shape:
        inv_det = numpy.linalg.det(cov_inv)
        if not inv_det > 0:
            rospy.logwarn("Determinant of inverse cov matrix {0} is equal or smaller than zero".format(inv_det))
            return 0.0
        norm_const = math.pow((2 * numpy.pi), float(size) / 2) * math.pow(1 / inv_det, 1.0 / 2) # determinant of inverse matrix is reciprocal
        if not norm_const > 0 :
            rospy.logwarn("Norm const {0} is equal or smaller than zero".format(norm_const))
            return 0.0
        x_mean = numpy.matrix(x - mean)
        exponent = -0.5 * (x_mean * cov_inv * x_mean.T)
        if exponent > 0:
            rospy.logwarn("Exponent {0} is larger than zero".format(exponent))
            exponent = 0
        result = math.pow(math.e, exponent)
        return result / norm_const
    else:
        rospy.logwarn("The dimensions of the input don't match")
        return 0.0

# tf.transformations.euler_from_quaternion is slow because the function calculates matrix inside.
# cf. https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def transform_quaternion_to_euler(quat):
    zero_thre = numpy.finfo(float).eps * 4.0 # epsilon for testing whether a number is close to zero
    roll_numerator = 2 * (quat[3] * quat[0] + quat[1] * quat[2])
    if abs(roll_numerator) < zero_thre:
        roll_numerator = numpy.sign(roll_numerator) * 0.0
    yaw_numerator = 2 * (quat[3] * quat[2] + quat[0] * quat[1])
    if abs(yaw_numerator) < zero_thre:
        yaw_numerator = numpy.sign(yaw_numerator) * 0.0
    return (numpy.arctan2(roll_numerator, 1 - 2 * (quat[0] ** 2 + quat[1] ** 2)),
            numpy.arcsin(2 * (quat[3] * quat[1] - quat[2] * quat[0])),
            numpy.arctan2(yaw_numerator, 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)))
    
class ParticleOdometry(object):
    ## initialize
    def __init__(self):
        # init node
        rospy.init_node("ParticleOdometry", anonymous=True)
        # instance valiables
        self.rate = float(rospy.get_param("~rate", 100))
        self.particle_num = float(rospy.get_param("~particle_num", 100))
        self.odom_frame = rospy.get_param("~odom_frame", "feedback_odom")
        self.base_link_frame = rospy.get_param("~base_link_frame", "BODY")
        self.odom_init_frame = rospy.get_param("~odom_init_frame", "odom_init")
        self.z_error_sigma = rospy.get_param("~z_error_sigma", 0.01) # z error probability from source. z is assumed not to move largely from source odometry
        self.use_imu = rospy.get_param("~use_imu", False)
        self.use_imu_yaw = rospy.get_param("~use_imu_yaw", False) # referenced only when use_imu is True
        self.roll_error_sigma = rospy.get_param("~roll_error_sigma", 0.05) # roll error probability from imu. (referenced only when use_imu is True)
        self.pitch_error_sigma = rospy.get_param("~pitch_error_sigma", 0.05) # pitch error probability from imu. (referenced only when use_imu is True)
        self.yaw_error_sigma = rospy.get_param("~yaw_error_sigma", 0.1) # yaw error probability from imu. (referenced only when use_imu and use_imu_yaw are both True)
        self.min_weight = rospy.get_param("~min_weight", 1e-10)
        self.source_skip_dt = rospy.get_param("~source_skip_dt", 0.1)
        self.r = rospy.Rate(self.rate)
        self.lock = threading.Lock()
        self.odom = None
        self.source_odom = None
        self.measure_odom = None
        self.particles = None
        self.weights = []
        self.measurement_updated = False
        self.init_sigma = [rospy.get_param("~init_sigma_x", 0.1),
                           rospy.get_param("~init_sigma_y", 0.1),
                           rospy.get_param("~init_sigma_z", 0.0001),
                           rospy.get_param("~init_sigma_roll", 0.0001),
                           rospy.get_param("~init_sigma_pitch", 0.0001),
                           rospy.get_param("~init_sigma_yaw", 0.05)]
        # tf
        self.listener = tf.TransformListener(True, rospy.Duration(10))
        self.broadcast = tf.TransformBroadcaster()
        self.publish_tf = rospy.get_param("~publish_tf", True)
        self.invert_tf = rospy.get_param("~invert_tf", True)
        # publisher
        self.pub = rospy.Publisher("~output", Odometry, queue_size = 1)
        # subscriber
        self.source_odom_sub = rospy.Subscriber("~source_odom", Odometry, self.source_odom_callback, queue_size = 10)
        self.measure_odom_sub = rospy.Subscriber("~measure_odom", Odometry, self.measure_odom_callback, queue_size = 10)
        self.imu_sub = rospy.Subscriber("~imu", Imu, self.imu_callback, queue_size = 10)
        self.init_signal_sub = rospy.Subscriber("~init_signal", Empty, self.init_signal_callback, queue_size = 10)
        # init
        self.initialize_odometry()

    def initialize_odometry(self):
        try:
            (trans,rot) = self.listener.lookupTransform(self.odom_init_frame, self.base_link_frame, rospy.Time(0))
        except:
            rospy.logwarn("[%s] failed to solve tf in initialize_odometry: %s to %s", rospy.get_name(), self.odom_init_frame, self.base_link_frame)
            trans = [0.0, 0.0, 0.0]
            rot = [0.0, 0.0, 0.0, 1.0]
        rospy.loginfo("[%s]: initiailze odometry ", rospy.get_name())
        with self.lock:
            self.particles = self.initial_distribution(Pose(Point(*trans), Quaternion(*rot)))
            self.weights = [1.0 / self.particle_num] * int(self.particle_num)
            self.odom = Odometry()
            mean, cov = self.guess_normal_distribution(self.particles, self.weights)
            self.odom.pose.pose = self.convert_list_to_pose(mean)
            self.odom.pose.covariance = list(itertools.chain(*cov))
            self.odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
            self.odom.twist.covariance = numpy.diag([0.001**2]*6).reshape(-1,).tolist()
            self.odom.header.stamp = rospy.Time.now()
            self.odom.header.frame_id = self.odom_frame
            self.odom.child_frame_id = self.base_link_frame
            self.source_odom = None
            self.measure_odom = None
            self.measurement_updated = False
            self.imu = None
            
    ## particle filter functions
    # input: particles(list of pose), source_odom(control input)  output: list of sampled particles(pose)
    def sampling(self, particles, source_odom):
        global_twist_with_covariance = self.transform_twist_with_covariance_to_global(source_odom.pose, source_odom.twist)
        sampled_velocities = self.state_transition_probability_rvs(global_twist_with_covariance.twist, global_twist_with_covariance.covariance) # make sampeld velocity at once because multivariate_normal calculates invert matrix and it is slow
        dt = (source_odom.header.stamp - self.odom.header.stamp).to_sec()
        # return self.calculate_pose_transform(prev_x, Twist(Vector3(*vel_list[0:3]), Vector3(*vel_list[3:6])), dt)
        return [self.calculate_pose_transform(prt, Twist(Vector3(*vel[0:3]), Vector3(*vel[3:6])), dt) for prt, vel in zip(particles, sampled_velocities)]
        
    # input: particles(list of pose), min_weight(float) output: list of weights
    def weighting(self, particles, min_weight):
        if not self.measure_odom:
            rospy.logwarn("[%s] measurement does not exist.", rospy.get_name())
            weights = [1.0 / self.particle_num] * int(self.particle_num) # use uniform weights when measure_odom has not been subscribed yet
        else:
            measure_to_source_dt = (self.source_odom.header.stamp - self.measure_odom.header.stamp).to_sec() # adjust timestamp of pose in measure_odom to source_odom
            current_measure_pose_with_covariance = self.update_pose_with_covariance(self.measure_odom.pose, self.measure_odom.twist, measure_to_source_dt) # assuming dt is small and measure_odom.twist is do not change in dt
            measurement_pose_array = numpy.array(self.convert_pose_to_list(current_measure_pose_with_covariance.pose))
            try:
                measurement_cov_matrix_inv = numpy.linalg.inv(numpy.matrix(current_measure_pose_with_covariance.covariance).reshape(6, 6)) # calculate inverse matrix first to reduce computation cost
                weights = [max(min_weight, self.calculate_weighting_likelihood(prt, measurement_pose_array, measurement_cov_matrix_inv)) for prt in particles]
            except numpy.linalg.LinAlgError:
                rospy.logwarn("[%s] covariance matrix is not singular.", rospy.get_name())
                weights = [min_weight] * len(particles)
            if all([x == min_weight for x in weights]):
                rospy.logwarn("[%s] likelihood is too small and all weights are limited by min_weight.", rospy.get_name())
            normalization_coeffs = sum(weights) # normalization and each weight is assumed to be larger than 0
            weights = [w / normalization_coeffs for w in weights]
        return weights

    def calculate_weighting_likelihood(self, prt, measurement_pose_array, measurement_cov_matrix_inv):
        measurement_likelihood = self.measurement_pdf(prt, measurement_pose_array, measurement_cov_matrix_inv)
        z_error_likelihood = self.z_error_pdf(prt.position.z) # consider difference from ideal z height to prevent drift
        if self.use_imu:
            imu_likelihood = self.imu_error_pdf(prt)
            return measurement_likelihood * z_error_likelihood * imu_likelihood
        else:
            return  measurement_likelihood * z_error_likelihood

    # input: list of particles, list of weights output: list of particles
    def resampling(self, particles, weights):
        uniform_probability = 1.0 / self.particle_num
        ret_particles = []
        probability_seed = numpy.random.rand() * uniform_probability
        weight_amount = self.weights[0]
        index = 0
        # for i in range(int(self.particle_num)):
        for i in range(int(self.particle_num)):
            selector = probability_seed + i * uniform_probability
            while selector > weight_amount and index < len(weights):
                index += 1
                weight_amount += weights[index]
            ret_particles.append(particles[index])
        return ret_particles

    ## probability functions
    # input: u(twist), u_cov(twist.covariance)  output: sampled velocity
    def state_transition_probability_rvs(self, u, u_cov): # rvs = Random Varieties Sampling
        u_mean = [u.linear.x, u.linear.y, u.linear.z,
                  u.angular.x, u.angular.y, u.angular.z]
        u_cov_matrix = zip(*[iter(u_cov)]*6)
        return numpy.random.multivariate_normal(u_mean, u_cov_matrix, int(self.particle_num)).tolist()

    # input: x(pose), mean(array), cov_inv(matrix), output: pdf value for x
    def measurement_pdf(self, x, measure_mean_array, measure_cov_matrix_inv): # pdf = Probability Dencity Function
        # w ~ p(z(t)|x(t))
        x_array = numpy.array(self.convert_pose_to_list(x))
        pdf_value = norm_pdf_multivariate(x_array, measure_mean_array, measure_cov_matrix_inv) # ~ p(x(t)|z(t))
        return pdf_value

    def z_error_pdf(self, particle_z):
        z_error = particle_z - self.source_odom.pose.pose.position.z
        return scipy.stats.norm.pdf(z_error, loc = 0.0, scale = self.z_error_sigma) # scale is standard divasion

    def imu_error_pdf(self, prt):
        if not self.imu:
            rospy.logwarn("[%s]: use_imu is True but imu is not subscribed", rospy.get_name())
            return 1.0 # multiply 1.0 make no effects to weight
        prt_euler = self.convert_pose_to_list(prt)[3:6]
        imu_euler = transform_quaternion_to_euler([self.imu.orientation.x, self.imu.orientation.y, self.imu.orientation.z, self.imu.orientation.w]) # imu.orientation is assumed to be global
        roll_pitch_pdf = scipy.stats.norm.pdf(prt_euler[0] - imu_euler[0], loc = 0.0, scale = self.roll_error_sigma) * scipy.stats.norm.pdf(prt_euler[1] - imu_euler[1], loc = 0.0, scale = self.pitch_error_sigma)
        if self.use_imu_yaw:
            return roll_pitch_pdf * scipy.stats.norm.pdf(prt_euler[2] - imu_euler[2], loc = 0.0, scale = self.yaw_error_sigma)
        else:
            return roll_pitch_pdf

    # input: init_pose(pose), output: initial distribution of pose(list of pose)
    def initial_distribution(self, init_pose):
        pose_list = numpy.random.multivariate_normal(numpy.array(self.convert_pose_to_list(init_pose)), numpy.diag([x ** 2 for x in self.init_sigma]), int(self.particle_num))
        return [self.convert_list_to_pose(pose) for pose in pose_list]

    ## top odometry calculations 
    def calc_odometry(self):
        # sampling
        self.particles = self.sampling(self.particles, self.source_odom)
        if self.measurement_updated:
            # weighting
            self.weights = self.weighting(self.particles, self.min_weight)
            # resampling
            self.particles = self.resampling(self.particles, self.weights)
            # wait next measurement
            self.measurement_updated = False

    def publish_odometry(self):
        # relfect source_odom information
        self.odom.header.stamp = self.source_odom.header.stamp
        self.odom.twist = self.source_odom.twist
        # estimate gaussian distribution for Odometry msg 
        mean, cov = self.guess_normal_distribution(self.particles, self.weights)
        self.odom.pose.pose = self.convert_list_to_pose(mean)
        self.odom.pose.covariance = list(itertools.chain(*cov))
        self.pub.publish(self.odom)
        if self.publish_tf:
            self.broadcast_transform()

    ## callback functions
    def source_odom_callback(self, msg):        
        with self.lock:
            vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                   msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
            if any([abs(v) > 0.5 for v in vel]):
                rospy.logwarn("[%s]: ignore source_odom because velocity over 0.5", rospy.get_name())
                return
            else:
                self.source_odom = msg
            # if self.source_odom != None:
            #     dt = (msg.header.stamp - self.source_odom.header.stamp).to_sec()
            # else:
            #     dt = 0
            # if dt > self.source_skip_dt:
            #     rospy.logwarn("[%s]: ignore source_odom because there is a suspicion that has stopped. elapsed time is %f [sec]", rospy.get_name(), dt)
            #     self.source_odom.header.stamp = msg.header.stamp
            # else:
            #     self.source_odom = msg

    def measure_odom_callback(self, msg):
        with self.lock:
            self.measure_odom = msg
            self.measurement_updated = True # raise measurement flag

    def init_signal_callback(self, msg):
        # time.sleep(1) # wait to update odom_init frame
        self.initialize_odometry()

    def imu_callback(self, msg):
        with self.lock:
            self.imu = msg
        
    # main functions
    def update(self):
        if not self.odom or not self.particles or not self.source_odom:
            rospy.logwarn("[%s]: odometry is not initialized", rospy.get_name())
            return
        else:
            self.calc_odometry()
            self.publish_odometry()

    def execute(self):
        while not rospy.is_shutdown():
            with self.lock:
                self.update() # call update() when control input is subscribed
            self.r.sleep()
        
    ## utils
    def convert_pose_to_list(self, pose):
        euler = transform_quaternion_to_euler((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
        return [pose.position.x, pose.position.y, pose.position.z, euler[0], euler[1], euler[2]]

    def convert_list_to_pose(self, lst):
        return Pose(Point(*lst[0:3]), Quaternion(*tf.transformations.quaternion_from_euler(*lst[3:6])))

    def guess_normal_distribution(self, particles, weights):
        # particles_lst = [self.convert_pose_to_list(prt) for prt in particles]
        # mean = numpy.mean(particles_lst, axis = 0)
        # cov = numpy.cov(particles_lst, rowvar = 0)
        
        # particles_list = [numpy.array(self.convert_pose_to_list(prt)) for prt in particles]
        # mean = None
        # cov = None
        # w2_sum = 0.0
        # mean = numpy.average(particles_list, axis = 0, weights = weights)
        # for prt, w in zip(particles_list, weights):
        #     if cov == None:
        #         cov = w * numpy.vstack(prt - mean) * (prt - mean)
        #     else:
        #         cov += w * numpy.vstack(prt - mean) * (prt - mean)
        #     w2_sum += w ** 2
        # cov = (1.0 / (1.0 - w2_sum)) * cov # unbiased covariance

        # calculate weighted mean and covariance (cf. https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance)        
        particle_array = numpy.array([self.convert_pose_to_list(prt) for prt in particles])
        weights_array = numpy.array(weights)
        mean = numpy.average(particle_array, axis = 0, weights = weights_array)
        diffs = particle_array - mean # array of x - mean
        cov = numpy.dot((numpy.vstack(weights_array) * diffs).T, diffs) # sum(w * (x - mean).T * (x - mean))
        cov = (1.0 / (1.0 - sum([w ** 2 for w in weights]))) * cov # unbiased covariance
        
        return (mean.tolist(), cov.tolist())

    def broadcast_transform(self):
        if not self.odom:
            return
        position = [self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z]
        orientation = [self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]
        if self.invert_tf:
            homogeneous_matrix = tf.transformations.quaternion_matrix(orientation)
            homogeneous_matrix[:3, 3] = numpy.array(position).reshape(1, 3)
            homogeneous_matrix_inv = numpy.linalg.inv(homogeneous_matrix)
            position = list(homogeneous_matrix_inv[:3, 3])
            orientation = list(tf.transformations.quaternion_from_matrix(homogeneous_matrix_inv))
            parent_frame = self.odom.child_frame_id
            target_frame = self.odom.header.frame_id
        else:
            parent_frame = self.odom.header.frame_id
            target_frame = self.odom.child_frame_id
        self.broadcast.sendTransform(position, orientation, rospy.Time.now(), target_frame, parent_frame)

    def transform_twist_with_covariance_to_global(self, pose, twist):
        trans = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
        rot = [pose.pose.orientation.x, pose.pose.orientation.y,
               pose.pose.orientation.z, pose.pose.orientation.w]
        rotation_matrix = tf.transformations.quaternion_matrix(rot)[:3, :3]
        twist_cov_matrix = numpy.matrix(twist.covariance).reshape(6, 6)
        global_velocity = numpy.dot(rotation_matrix, numpy.array([[twist.twist.linear.x],
                                                                  [twist.twist.linear.y],
                                                                  [twist.twist.linear.z]]))
        global_omega = numpy.dot(rotation_matrix, numpy.array([[twist.twist.angular.x],
                                                               [twist.twist.angular.y],
                                                               [twist.twist.angular.z]]))
        global_twist_cov_matrix = numpy.zeros((6, 6))
        global_twist_cov_matrix[:3, :3] = (rotation_matrix.T).dot(twist_cov_matrix[:3, :3].dot(rotation_matrix))
        global_twist_cov_matrix[3:6, 3:6] = (rotation_matrix.T).dot(twist_cov_matrix[3:6, 3:6].dot(rotation_matrix))

        return TwistWithCovariance(Twist(Vector3(*global_velocity[:, 0]), Vector3(*global_omega[:, 0])),
                                   global_twist_cov_matrix.reshape(-1,).tolist())

    def update_pose_with_covariance(self, pose_with_covariance, twist_with_covariance, dt):
        global_twist_with_covariance = self.transform_twist_with_covariance_to_global(pose_with_covariance, twist_with_covariance)
        # calculate current pose as integration
        ret_pose = self.calculate_pose_transform(pose_with_covariance.pose, global_twist_with_covariance.twist, dt)
        # update covariance
        ret_pose_cov = self.calculate_pose_covariance_transform(pose_with_covariance.covariance, global_twist_with_covariance.covariance, dt)
        return PoseWithCovariance(ret_pose, ret_pose_cov)

    def calculate_pose_transform(self, pose, global_twist, dt):
        ret_pose = Pose()
        # calculate current pose as integration
        ret_pose.position.x = pose.position.x + global_twist.linear.x * dt
        ret_pose.position.y = pose.position.y + global_twist.linear.y * dt
        ret_pose.position.z = pose.position.z + global_twist.linear.z * dt
        ret_pose.orientation = self.calculate_quaternion_transform(pose.orientation, global_twist.angular, dt)
        return ret_pose

    def calculate_quaternion_transform(self, orientation, angular, dt): # angular is assumed to be global
        # quaternion calculation
        quat_vec = numpy.array([[orientation.x],
                                [orientation.y],
                                [orientation.z],
                                [orientation.w]])
        # skew_omega = numpy.matrix([[0, angular.z, -angular.y, angular.x],
        #                            [-angular.z, 0, angular.x, angular.y],
        #                            [angular.y, -angular.x, 0, angular.z],
        #                            [-angular.x, -angular.y, -angular.z, 0]])
        skew_omega = numpy.matrix([[0, -angular.z, angular.y, angular.x],
                                   [angular.z, 0, -angular.x, angular.y],
                                   [-angular.y, angular.x, 0, angular.z],
                                   [-angular.x, -angular.y, -angular.z, 0]])
        new_quat_vec = quat_vec + 0.5 * numpy.dot(skew_omega, quat_vec) * dt
        norm = numpy.linalg.norm(new_quat_vec)
        if norm == 0:
            rospy.logwarn("norm of quaternion is zero")
        else:
            new_quat_vec = new_quat_vec / norm # normalize
        return Quaternion(*numpy.array(new_quat_vec).reshape(-1,).tolist())

    def calculate_pose_covariance_transform(self, pose_cov, global_twist_cov, dt):
        ret_pose_cov = []
        # make matirx from covariance array
        prev_pose_cov_matrix = numpy.matrix(pose_cov).reshape(6, 6)
        global_twist_cov_matrix = numpy.matrix(global_twist_cov).reshape(6, 6)
        # jacobian matrix
        # elements in pose and twist are assumed to be independent on global coordinates
        jacobi_pose = numpy.diag([1.0] * 6)
        jacobi_twist = numpy.diag([dt] * 6)
        # covariance calculation
        pose_cov_matrix = jacobi_pose.dot(prev_pose_cov_matrix.dot(jacobi_pose.T)) + jacobi_twist.dot(global_twist_cov_matrix.dot(jacobi_twist.T))
        # update covariances as array type (twist is same as before)
        ret_pose_cov = numpy.array(pose_cov_matrix).reshape(-1,).tolist()
        return ret_pose_cov
        

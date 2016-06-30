#! /usr/bin/env python

import rospy
from jsk_robot_startup.OdometryImuTwistCompensation import *

if __name__ == '__main__':
    try:
        node = OdometryImuTwistCompensation()
        node.execute()
    except rospy.ROSInterruptException: pass

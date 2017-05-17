#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(init-unit-test)

(load "package://baxtereus/baxter-interface.l")

(ros::load-ros-manifest "baxtereus")
(ros::roseus "baxter-moveit")

(deftest test-baxter-moveit ()
  (let ((waiting-count 0) ri robot av)
    (while
      (null (and (one-shot-subscribe "/clock" rosgraph_msgs::Clock :timeout 3000)
                 (one-shot-subscribe "/robot/state" baxter_core_msgs::AssemblyState :timeout 3000)))
      (ros::ros-info "waiting gazebo")
      (setq waiting-count (incf waiting-count))
      (assert (> 300 waiting-count) "gazebo did not start.")
      (unix::sleep 3))
    (ros::ros-info "gazebo ready")

    (setq waiting-count 0)
    (while
      (null (one-shot-subscribe "/move_group/status" actionlib_msgs::GoalStatusArray :timeout 3000))
      (ros::ros-info "waiting moveit")
      (setq waiting-count (incf waiting-count))
      (assert (> 300 waiting-count) "moveit did not start.")
      (unix::sleep 3))
    (ros::ros-info "moveit ready")

    (setq ri (instance baxter-interface :init))
    (setq robot (instance baxter-robot :init))
    (send ri :angle-vector-raw (send robot :reset-pose))
    (send ri :wait-interpolation)

    (setq av (send ri :angle-vector (send robot :init-pose) 3000))
    (send ri :wait-interpolation)
    (assert av "failed: (send ri :angle-vector (send robot :init-pose) 3000)")

    (setq av (send ri :angle-vector (send robot :untuck-pose) 3000 :rarm-controller 0))
    (send ri :wait-interpolation)
    (assert av "failed: (send ri :angle-vector (send robot :untuck-pose) 3000 :rarm-controller 0)")

    (setq av (send ri :angle-vector (send robot :untuck-pose) 3000 :larm-controller 0))
    (send ri :wait-interpolation)
    (assert av "failed: (send ri :angle-vector (send robot :untuck-pose) 3000 :larm-controller 0)")

    (setq av (send ri :angle-vector (send robot :init-pose) 3000 :rarm-controller 0 :move-arm :rarm))
    (send ri :wait-interpolation)
    (assert av "failed: (send ri :angle-vector (send robot :init-pose) 3000 :rarm-controller 0 :move-arm :rarm)")

    (setq av (send ri :angle-vector (send robot :init-pose) 3000 :larm-controller 0 :move-arm :larm))
    (send ri :wait-interpolation)
    (assert av "failed: (send ri :angle-vector (send robot :init-pose) 3000 :larm-controller 0 :move-arm :larm)")

    (setq av (send ri :angle-vector (send robot :untuck-pose) 3000 nil 0 :move-arm :arms))
    (send ri :wait-interpolation)
    (assert av "failed: (send ri :angle-vector (send robot :untuck-pose) 3000 nil 0 :move-arm :arms)")
    ))

(run-all-tests)
(exit)
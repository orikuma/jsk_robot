(load "package://pr2eus/robot-interface.l")
(require :baxter "package://baxtereus//baxter-util.l")
(load "package://pr2eus/speak.l")
(ros::load-ros-manifest "control_msgs")
(ros::load-ros-manifest "baxter_core_msgs")

(defparameter *suction* 65537) ;;vacuum
(defparameter *electoric* 65538)
(defvar *wait-for-suction* 5000000)

(defclass baxter-interface
  :super robot-interface
  :slots (gripper-sequence-id right-gripper-action left-gripper-action
			      right-gripper-type left-gripper-type))
(defmethod baxter-interface
  (:init (&rest args)
   (prog1 (send-super* :init :robot baxter-robot :joint-states-topic "/robot/joint_states" :groupname "baxter_interface" args)
     (send self :add-controller :larm-controller)
     (send self :add-controller :rarm-controller)
     (send self :add-controller :head-controller)
     (ros::advertise "/robot/end_effector/right_gripper/command" baxter_core_msgs::EndEffectorCommand 5) 
     (ros::advertise "/robot/end_effector/left_gripper/command" baxter_core_msgs::EndEffectorCommand 5) 
     (ros::advertise "/robot/xdisplay" sensor_msgs::Image 1)
     (ros::advertise "/robot/head/command_head_nod" std_msgs::Bool 1)
     (ros::subscribe "/robot/end_effector/right_gripper/properties" baxter_core_msgs::EndEffectorProperties #'send self :right-property-cb :groupname groupname)
     (ros::subscribe "/robot/end_effector/left_gripper/properties" baxter_core_msgs::EndEffectorProperties #'send self :left-property-cb :groupname groupname)

     (setq right-gripper-action (instance ros::simple-action-client :init
					  "robot/end_effector/right_gripper/gripper_action"
					  control_msgs::GripperCommandAction
					  :groupname groupname))
     (setq left-gripper-action (instance ros::simple-action-client :init
					 "robot/end_effector/left_gripper/gripper_action"
					 control_msgs::GripperCommandAction
					 :groupname groupname
					 ))
     (if (ros::has-param "~wait_for_suction")
	 (setq *wait-for-suction* (read-from-string (ros::get-param "~wait_for_suction"))))
     (dolist (action (list right-gripper-action left-gripper-action))
       (unless (and joint-action-enable (send action :wait-for-server 3))
	 (ros::ros-warn "~A is not respond" action)
	 (ros::ros-info "*** if you do not have gripper, you can ignore this message ***")
	 (return)))

     (setq gripper-sequence-id 0)
     (ros::spin-once)
     ))
  (:right-property-cb (msg)
   (setq right-gripper-type (send msg :id))
   )
  (:left-property-cb (msg)
   (setq left-gripper-type (send msg :id))
   )
  (:default-controller ()
   (append
    (send self :larm-controller)
    (send self :rarm-controller)
    (send self :head-controller)))
  (:larm-controller ()
   (list
    (list
     (cons :controller-action "/robot/limb/left/follow_joint_trajectory")
     (cons :controller-state "/robot/limb/left/state")
     (cons :action-type control_msgs::FollowJointTrajectoryAction)
     (cons :joint-names (list "left_s0" "left_s1" "left_e0" "left_e1" "left_w0" "left_w1" "left_w2")))))
  (:rarm-controller ()
   (list
    (list
     (cons :controller-action "/robot/limb/right/follow_joint_trajectory")
     (cons :controller-state "/robot/limb/right/state")
     (cons :action-type control_msgs::FollowJointTrajectoryAction)
     (cons :joint-names (list "right_s0" "right_s1" "right_e0" "right_e1" "right_w0" "right_w1" "right_w2")))))
  (:head-controller ()
   (list
    (list
     (cons :controller-action "/robot/head/head_action")
     (cons :controller-state "/robot/head/head_state")
     (cons :action-type control_msgs::SingleJointPositionAction)
     (cons :joint-names (list "head_pan")))))
  (:close-head-camera ()
    (send self :close-camera "head_camera")
    )
  (:close-right-hand-camera ()
    (send self :close-camera "right_hand_camera")
    )
  (:close-left-hand-camera ()
    (send self :close-camera "left_hand_camera")
    )
  (:close-camera (camera)
    (let ((req (instance baxter_core_msgs::CloseCameraRequest :init)))
      (send req :name camera)
      (ros::service-call "/cameras/close" req)
      )
    )
  (:open-head-camera ()
    (send self :open-camera "head_camera")
    )
  (:open-right-hand-camera ()
    (send self :open-camera "right_hand_camera")
    )
  (:open-left-hand-camera ()
    (send self :open-camera "left_hand_camera")
    )
  (:open-camera (camera)
    (let ((camera-setting (instance baxter_core_msgs::CameraSettings :init :width 1280 :height 800 :fps 20))
          (req (instance baxter_core_msgs::OpenCameraRequest :init))
          )
      (send req :name camera)
      (send req :settings camera-setting)
      (ros::service-call "/cameras/open" req)
      )
    )
  (:get-cameras-list()
    (ros::wait-for-service "/cameras/list")
    (let ((req (instance baxter_core_msgs::ListCamerasRequest :init)) res)
      (setq res (ros::service-call "/cameras/list" req))
      (send res :cameras)
      ))

  (:clear-calib-grasp
   (&optional (arm :arms))
   (dolist (a (if (eq arm :arms) (list "left" "right") (list (if (eq arm :rarm) "right" "left"))))
     (send self :command-grasp baxter_core_msgs::EndEffectorCommand::*CMD_CLEAR_CALIBRATION* a)
     )
   )

  (:calib-grasp
   (&optional (arm :arms) &key ((:pos pos) 100))
   (dolist (a (if (eq arm :arms) (list "left" "right") (list (if (eq arm :rarm) "right" "left"))))
     (send self :command-grasp baxter_core_msgs::EndEffectorCommand::*cmd_calibrate* a)
     )
   )

  (:start-grasp
   (&optional (arm :arms) &key (effort 50))
   (send self :go-grasp arm :pos 0 :effort effort)
   (if (or ( and (equal arm :arms) (or (equal right-gripper-type *suction*) (equal left-gripper-type *suction*)))
	   ( and (equal arm :rarm) (equal right-gripper-type *suction*))
	   ( and (equal arm :larm) (equal left-gripper-type *suction*)))
       (unix::usleep *wait-for-suction*)
     )
   )

  (:stop-grasp
   (&optional (arm :arms) &key (effort 50) (wait nil))
   (send self :go-grasp arm :pos 100 :effort effort :wait wait)
   (if (or ( and (equal arm :arms) (or (equal right-gripper-type *suction*) (equal left-gripper-type *suction*)))
	   ( and (equal arm :rarm) (equal right-gripper-type *suction*))
	   ( and (equal arm :larm) (equal left-gripper-type *suction*)))
       (unix::usleep *wait-for-suction*)
     )
   )

  (:go-grasp
   (&optional (arm :arms) &key (pos 0) (effort 50) (wait t))
   (let (goal
	 (clients (case arm
		    (:larm (list left-gripper-action))
		    (:rarm (list right-gripper-action))
		    (:arms (list right-gripper-action left-gripper-action))
		    (t (ros::ros-warn "arm[~a] in :move-gripper is invalid type" arm))))
	 (results (list)))
     (dolist (client clients)
       (ros::ros-info "send-goal to ~A" client)
       (setq goal  (instance control_msgs::GripperCommandActionGoal :init))
       (send goal :goal :command :position pos)
       (send goal :goal :command :max_effort effort)
       (send client :send-goal goal)
       )
     
     (if wait (send-all clients :wait-for-result)) 
     (dolist (client clients)
       (push (send client :get-result) results)
       )
     results
     )
   )

  (:command-grasp
   (cmd arm &key ((:pos pos) nil))
   (ros::spin-once)
   (let ((ee-cmd (instance baxter_core_msgs::EndEffectorCommand :init)))
     (if (equal arm "right")
	 (send ee-cmd :id right-gripper-type)
       (send ee-cmd :id left-gripper-type)
       )
     (send ee-cmd :command cmd)
     (send ee-cmd :sender "/baxter_interface")
     (send ee-cmd :sequence gripper-sequence-id)
     (if pos
         (send ee-cmd :args (format nil "{\"position\": ~A}" pos)))
     (ros::publish (format nil "/robot/end_effector/~A_gripper/command" arm) ee-cmd)
     (setq gripper-sequence-id (1+ gripper-sequence-id))
     )
   )

  (:set-baxter-face 
   (filepath)
   (let (img msg)
     (setq img (read-jpeg-file filepath))
     
     ;;convert image bgr to rgb
     (dotimes (x (send img :width))
       (dotimes (y (send img :height))
	 (let (r b bd)
	   (setq bd (send img :byte-depth))
	   (setq b (aref img y (* bd x)))
	   (setq r (aref img y (+ (* bd x) 2)))
	   (setf (aref img y (* bd x)) r)
	   (setf (aref img y (+ (* bd x) 2)) b)
	   )
	 )
       )
     (setq msg (instance sensor_msgs::Image :init))
     (send msg :height (send img :height))
     (send msg :width (send img :width))
     (send msg :encoding "rgb8")
     (send msg :step (* (send img :byte-depth) (send img :width)))
     (send msg :data (send img :entity))
     (send msg :header :stamp (ros::time-now))
     (ros::publish "/robot/xdisplay" msg)
     )
   )

  (:nod
   ()
   (let ((msg (instance std_msgs::Bool :init :data t)))
     (ros::publish "/robot/head/command_head_nod" msg)
     )
   )
  (:angle-vector (av &optional (tm :fast) (ctype controller-type) (start-time 0) &rest args)
		 (send* self :angle-vector-sequence (list av) (list tm) ctype start-time args))
  (:angle-vector-sequence (avs &optional (tms :fast) (ctype controller-type) (start-time 0) &key (scale 2.2) (min-time 0.05))
		 ;; force add current position to the top of avs
		 (if (atom tms) (setq tms (list tms)))
		 (push (send self :state :potentio-vector) avs)
		 (push 50 tms)
		 (when (= (length avs) 2) ;; when input avs is 1
		   (setq avs (list (elt avs 0) (midpoint 0.5 (elt avs 0) (elt avs 1)) (elt avs 1)))
		   (cond ((numberp (elt tms 1))
			  (setq tms (list (elt tms 0) (/ (elt tms 1) 2) (/ (elt tms 1) 2))))
			 (t
			  (setq tms (list (elt tms 0) (elt tms 1) (elt tms 1))))))
		 (when (= (length avs) 3) ;; when input avs is 1 or 2
		   (setq avs (append avs (list (elt avs 2))))
		   (setq tms (append tms (list 50))))
     (setq ctype (or ctype controller-type))  ;; use default if ctype is nil
		 (send-super :angle-vector-sequence avs tms ctype start-time :scale scale :min-time min-time))
  
  (:ros-state-callback
   (msg)
   (let ((robot-state-names (cdr (assoc :name robot-state))) (robot-msg-names (send msg :name)) (torso-index))
     ;;Remove toros_t0 . We think this is not the rotational-joint
     (setq torso-index (position "torso_t0" robot-msg-names :test #'string=))
     (send msg :name ( remove "torso_t0" robot-state-names :test #'string=))
     (when torso-index
       (dolist (key '(:position :velocity :effort) )
         (send msg key (concatenate float-vector 
                                    (subseq (send msg key) 0 torso-index)
                                    (subseq (send msg key) (+ torso-index 1))))))
     ;;End of Removing torso_t0
     (send-super :ros-state-callback msg)
     ))
  ) ;; defclass baxter-interface


(defun baxter-init (&key (safe t) (type :default-controller))
  (if (not (boundp '*ri*))
      (setq *ri* (instance baxter-interface :init :type type)))
  (if (not (boundp '*baxter*))
      (if safe
	  (setq *baxter* (instance baxter-robot-safe :init))
	(setq *baxter* (instance baxter-robot :init))))
  (send *ri* :calib-grasp :arms)
  )

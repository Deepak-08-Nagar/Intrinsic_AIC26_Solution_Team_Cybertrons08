from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)

from aic_task_interfaces.msg import Task
 




class MyPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.parent_node = parent_node
        self.parent_node.get_logger().info("Cable Insertion Policy Initialized")

        # Imports 
        import numpy as np
        import cv2
        import math
        import os
        from ament_index_python.packages import get_package_share_directory
        from ultralytics import YOLO
        import torch
        from sensor_msgs.msg import Image
        from rclpy.duration import Duration




        # YOLO Model
        pkg_path = get_package_share_directory('my_policy_node')
        model_path = os.path.join(pkg_path, 'models', 'obb_model.pt')
        model_blue_path = os.path.join(pkg_path, 'models', 'blue.pt')
        model_roll_path = os.path.join(pkg_path, 'models', 'roll_model.pkl')
        model_roll_path_sfp = os.path.join(pkg_path, 'models', 'roll_model1.pkl')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.parent_node.get_logger().info(f"[YOLO] Using device: {self.device}")
    
        self.model = YOLO(model_path).to(self.device)
        self.model_blue = YOLO(model_blue_path).to(self.device)

        # Load Roll Error Model
        import pickle
        try:
            with open(model_roll_path, "rb") as f:
                self.roll_model = pickle.load(f)
            self.parent_node.get_logger().info("Loaded roll_model.pkl successfully")
        except Exception as e:
            self.parent_node.get_logger().error(f"Failed to load roll_model.pkl: {e}")
            self.roll_model = None

        try:
            with open(model_roll_path_sfp, "rb") as f:
                self.roll_model_sfp = pickle.load(f)
            self.parent_node.get_logger().info("Loaded roll_model_sfp.pkl successfully")
        except Exception as e:
            self.parent_node.get_logger().error(f"Failed to load roll_model_sfp.pkl: {e}")
            self.roll_model_sfp = None

        # Load Rail Classifier Coefficients (JSON)
        rail_model_path = os.path.join(pkg_path, 'models', 'rail_classifier.json')
        try:
            import json
            with open(rail_model_path, "r") as f:
                self.rail_coefficients = json.load(f)
            self.parent_node.get_logger().info("Loaded rail_classifier.json successfully")
        except Exception as e:
            self.parent_node.get_logger().error(f"Failed to load rail_classifier.json: {e}")
            self.rail_coefficients = None

        # Publishers
        self.pub_output_cam_center = self.parent_node.create_publisher(Image, '/output_cam_center', 10)
        self.pub_output_cam_left   = self.parent_node.create_publisher(Image, '/output_cam_left', 10)

        # Vision state
        self.ports_center             = []
        self.cable_center             = []
        self.current_best_port_center = None
        self.current_cable_center     = []
        self.port1_center_pt          = None
        self.port2_center_pt          = None
        self.prev_best_port_center    = None
        self.max_pixel_jump           = 100.0

        # Yaw fix state
        self.port_starting_yaw = None
        self.once_fix_yaw      = True
        self.wait_yaw_fix      = True
        self.yaw_fixed         = False
        self.yaw_180_fixed     = True # Assuming for now, that it will not be 180 degree flipped


        # Blue Yaw fix state
        self.once_fix_yaw_blue      = True
        self.wait_yaw_fix_blue      = True
        self.yaw_fixed_blue         = False
        self.yaw_180_fixed_blue     = False

        self.sc_aligned = False
        self.blue_fix_roll = False

        # Virtual target / ratio lock
        self.last_known_cable_center = None
        self.last_known_cable_pts    = None
        
        self.dx_ratio_history_blue        = []
        self.dy_ratio_history_blue        = []
        self.required_offset_samples_blue = 15
        self.saved_normalized_offset_blue = None
        self.is_ratio_locked_blue         = False
        self.nearest_corner_idx_blue      = None
        self.prev_target_port_center_blue = None
        self.last_known_cable_blue        = None

        # Virtual target / ratio lock (SFP)
        self.dx_ratio_history        = []
        self.dy_ratio_history        = []
        self.required_offset_samples = 15
        self.saved_normalized_offset = None
        self.is_ratio_locked         = False

        # Stiffness / damping
        self.current_stiffness = [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
        self.current_damping   = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]

        self.current_target_rail = 0

        self.target_port_index   = 0

        # Contact flag (set externally when contact is detected)
        self.is_contact_made = False

        self.target_port_index   = 0
        self.is_contact_made     = False

        # Vision and State
        self.active_port_type = None
        self.lock_hold_counter = 0
        self.xy_locked = False
        self.lock_hold_counter = 0
        self.lock_hold_counter = 0
        self.tip_missing_counter = 0
        from geometry_msgs.msg import Twist
        self.last_valid_twist = Twist()
        self.wiggle_counter = 0
        self.still_counter = 0
        self.prev_cable_center = None
        self.once_reset = True
        self.taskboard_gap_ready = False
        self.pink_region_found = False
        self.ready_to_scan = False
        self.target_port_found = False
        self.target_rail_y = None
        self.identified_rails = {}
        self.rails_identified = False
        self.target_top_edge_offset = None
        self.snapshot_saved = False

        
        # Occlusion Prediction Variables (from anany_node.py)
        self.dx_ratio_history = []
        self.dy_ratio_history = []
        self.required_offset_samples = 15
        self.saved_normalized_offset = None
        self.is_ratio_locked = False
        self.prev_target_top_edge = None
        self.MAX_PORT_JUMP = 60.0
        self.wait_initial = 0

        
        self.prev_predicted_error_pitch = None
        self.pitch_counter = 0
        self.target_rail_yaw_fixed = False
        self.initial_pitch = 0
        self.Energy_Stored = True



    # ================================================================
    # RESET STATE BETWEEN TRIALS
    # ================================================================
    def reset_state(self):
        from geometry_msgs.msg import Twist
        self.once_reset = True
        self.taskboard_gap_ready = False
        self.pink_region_found = False
        self.ready_to_scan = False
        self.wait_initial = 0


        # Vision state
        self.ports_center             = []
        self.cable_center             = []
        self.current_best_port_center = None
        self.current_cable_center     = []
        self.port1_center_pt          = None
        self.port2_center_pt          = None
        self.prev_best_port_center    = None
        self.max_pixel_jump           = 100.0

        # Yaw fix state
        self.port_starting_yaw = None
        self.once_fix_yaw      = True
        self.wait_yaw_fix      = True
        self.yaw_fixed         = False
        self.yaw_180_fixed     = True

        # Blue Yaw fix state
        self.once_fix_yaw_blue      = True
        self.wait_yaw_fix_blue      = True
        self.yaw_fixed_blue         = False
        self.yaw_180_fixed_blue     = False

        # SC state
        self.sc_aligned    = False
        self.blue_fix_roll = False

        # Virtual target / ratio lock (SC)
        self.last_known_cable_center       = None
        self.last_known_cable_pts          = None
        self.dx_ratio_history_blue         = []
        self.dy_ratio_history_blue         = []
        self.saved_normalized_offset_blue  = None
        self.is_ratio_locked_blue          = False
        self.nearest_corner_idx_blue       = None
        self.prev_target_port_center_blue  = None
        self.last_known_cable_blue         = None

        # Virtual target / ratio lock (SFP)
        self.dx_ratio_history        = []
        self.dy_ratio_history        = []
        self.saved_normalized_offset = None
        self.is_ratio_locked         = False
        self.last_known_ref_center   = None
        self.last_known_target_center= None
        self.virtual_target_debug_point = None

        # Stiffness / damping
        self.current_stiffness = [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
        self.current_damping   = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]

        # self.current_target_rail          = 0

        # self.target_port_index = 0

        # Contact flag
        self.is_contact_made = False

        # Servoing control state
        self.xy_locked           = False
        self.lock_hold_counter   = 0
        self.still_counter       = 0
        self.prev_cable_center   = None
        self.tip_missing_counter = 0
        self.last_valid_twist    = Twist()
        self.port_search_counter = 0

        self.tip_missing_counter = 0

        # EMA smoothing state
        self.smoothed_ex_sfp       = None
        self.smoothed_ey_sfp       = None
        self.smoothed_ex_sc        = None
        self.smoothed_ey_sc        = None

        self.target_nic_polygon  = None
        self.virtual_target_debug_point = None
        
        self.ema_alpha = 0.5

        self.last_seen_stamp = None
        self.zero_vel_start_time = None

        self.target_port_found = False
        self.target_rail_y = None
        self.identified_rails = {}
        self.rails_identified = False
        self.target_top_edge_offset = None
        self.snapshot_saved = False

        
        # Reset Occlusion logic
        self.dx_ratio_history = []
        self.dy_ratio_history = []
        self.saved_normalized_offset = None
        self.is_ratio_locked = False
        self.prev_target_top_edge = None
        self.pitch_counter = 0
        self.target_rail_yaw_fixed = False



        
        # SFP Cable Locking logic
        self.cable_collection_list = []
        self.fixed_cable_pts       = None
        self.fixed_cable_center    = None
        self.initial_pitch = 0
        self.Energy_Stored = True
    # ================================================================
    # MAIN ENTRY POINT
    # ================================================================
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.reset_state()
        
        # Imports
        import numpy as np
        import cv2
        import math

        from aic_control_interfaces.msg import (
    MotionUpdate,
    JointMotionUpdate,
    TrajectoryGenerationMode,
)
        from rclpy.duration import Duration
        from geometry_msgs.msg import Twist, Vector3, Wrench

        self.parent_node.get_logger().info("Starting Insertion Loop")
        
        start_time = self.time_now()
        
        timeout = Duration(seconds=60.0)
        if task.time_limit <= 60.0:
            timeout = Duration(seconds= task.time_limit)

        self.active_port_type = task.port_type
        
        # ── SAFETY LIFT ──────────────────────────────────────────────────────────
        obs = get_observation()
        if obs is not None:
            tcp_z = obs.controller_state.tcp_pose.position.z
            if tcp_z < 0.29:
                self.parent_node.get_logger().warn(f"[SAFETY] Gripper too low ({tcp_z:.3f}m). Lifting to safety...")
                mu = MotionUpdate()
                mu.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
                mu.velocity.linear.z = 0.05
                lift_start = self.time_now()
                while tcp_z < 0.30 and (self.time_now() - lift_start).nanoseconds / 1e9 < 2.0:
                    move_robot(motion_update=mu)
                    obs = get_observation()
                    if obs is None: break
                    tcp_z = obs.controller_state.tcp_pose.position.z
                    self.sleep_for(0.01)
                # Stop motion
                mu.velocity.linear.z = 0.0
                move_robot(motion_update=mu)
                self.parent_node.get_logger().info("[SAFETY] Lift complete.")

        if self.Energy_Stored:
            obs = get_observation()
            if obs is not None:
                self.flush_stored_energy(obs, move_robot)
                self.Energy_Stored = False

        if task.port_type == "sfp":
            self.current_target_rail = int(task.target_module_name[-1])
            self.target_port_index   = int(task.port_name[-1])
            self.parent_node.get_logger().info(f"Target Rail: {self.current_target_rail}, Target Port: {self.target_port_index}")
            while self.time_now() - start_time < timeout:
                if self.time_now()- start_time > timeout - Duration(seconds=1.0):
                    obs = get_observation()
                    if obs is not None:
                        self.flush_stored_energy(obs, move_robot)
                    return True
                if self.once_reset:
                    self.reset_state()
                    self.once_reset = False
                

                self.sleep_for(0.01) # Faster control loop (100Hz)

                obs = get_observation()
                if obs is None:
                    continue



                c_img_cv = self._msg_to_cv2(obs.center_image)
                
                # Synchronous Detection
                c_img_output_cv, p_center, c_center, c_pts, yaw = self._detect(c_img_cv)
                
                self.ports_center = p_center
                all_detected_ports = p_center.copy()

                self.cable_center = c_center
                self.cable_pts = c_pts
                if self.once_fix_yaw:
                    self.port_starting_yaw = yaw
                
                # 1. Group ports into rails (Available for all phases)
                groups = []
                if len(p_center) > 0:
                    sorted_p = sorted(p_center, key=lambda p: p["center"][1])
                    curr_group = [sorted_p[0]]
                    for i in range(1, len(sorted_p)):
                        if sorted_p[i]["center"][1] - curr_group[-1]["center"][1] < 15:
                            curr_group.append(sorted_p[i])
                        else:
                            groups.append(curr_group)
                            curr_group = [sorted_p[i]]
                    groups.append(curr_group)
                
                #--------------------------------------------------------------------------------------

                # Cable Caching Logic
                if len(self.cable_center) > 0:
                    self.current_cable_center = self.cable_center
                    self.last_known_cable_center = self.cable_center
                    self.last_known_cable_pts = self.cable_pts
                elif self.last_known_cable_center is not None:
                    self.current_cable_center = self.last_known_cable_center
                    if hasattr(self, 'last_known_cable_pts') and self.last_known_cable_pts is not None:
                        pts_cached = self.last_known_cable_pts.astype(int)
                        cv2.polylines(c_img_output_cv, [pts_cached], True, (0, 165, 255), 2)
                        cv2.putText(c_img_output_cv, "C_CACHED", (pts_cached[0][0], pts_cached[0][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                else:
                    self.current_cable_center = []

                #-------------------------------------------------------------------------------------------------------------

            

                # ── PHASE 0: YAW FIX (joint-space) ──────────────────────────────────
                if self.once_fix_yaw:
                    if self.wait_initial <=50:
                        self.wait_initial += 1
                        
                    if self.port_starting_yaw is not None and self.wait_initial >= 50:                        
                        # self.parent_node.get_logger().info(f"Yaw= {self.port_starting_yaw:.4f}")
                        if abs(self.port_starting_yaw) <= 0.001:
                            self.wait_yaw_fix = False

                        if not self.wait_yaw_fix:
                            # Yaw within tolerance — send zero vel to stop, then advance
                            jmu = self._make_joint_vel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                            move_robot(joint_motion_update=jmu)
                            self.parent_node.get_logger().info(
                                f"[YAW FIX] Complete. yaw={self.port_starting_yaw:.4f} rad"
                            )
                            self.wait_yaw_fix = True
                            self.once_fix_yaw = False
                            self.yaw_fixed = True
                        else:
                            # Actively rotate wrist-3 using smooth proportional control
                            rot_vel = self.port_starting_yaw * 1.5
                            rot_vel = float(np.clip(rot_vel, -0.2, 0.2))
                            jmu = self._make_joint_vel([0.0, 0.0, 0.0, 0.0, 0.0, rot_vel])
                            move_robot(joint_motion_update=jmu)
                    # Publish during Phase 0
                    c_img_output = self._cv2_to_msg(c_img_output_cv, obs.center_image.header)
                    self.pub_output_cam_center.publish(c_img_output)
                        
                    continue
                
                

                #----------------------------------------------------------------------------------------------

                # phase 0.5: initialize vels: ------------------------------------------
                if True:
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                    wx = 0.0
                    wy = 0.0
                    wz = 0.0
                    ex = 0.0
                    ey = 0.0
                    error = 0.0
                #-----------------------------------------------------------------------------------

                # Phase 1.5 : port prediction 

                    # first make sure that yaw is fixed, and then there is 
                    # ittle gap between cable and taskboard and the pink region is visible in center
                
                if self.yaw_fixed and self.yaw_180_fixed and not self.taskboard_gap_ready:

                    if self.current_cable_center is not None and len(self.current_cable_center) >= 2:
                        gray = cv2.cvtColor(c_img_cv, cv2.COLOR_BGR2GRAY)
                        _, black_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                        
                        kernel = np.ones((5, 5), np.uint8)
                        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
                        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
                        
                        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Target the black region directly above the cable tip
                            cable_tip_x = int(self.current_cable_center[0])
                            cable_tip_y = int(self.current_cable_center[1])
                            # 1. Filter: Keep only contours that have at least some part above the cable tip
                            # (Remember: Y increases downwards, so "above" means smaller Y)
                            valid_contours = [cnt for cnt in contours if np.min(cnt[:, :, 1]) < cable_tip_y]
                            
                            taskboard_contour = None
                            if valid_contours:
                                # 2. Try to find the specific contour directly above the tip
                                search_pt = (float(cable_tip_x), float(cable_tip_y - 40))
                                for cnt in valid_contours:
                                    if cv2.pointPolygonTest(cnt, search_pt, False) >= 0:
                                        taskboard_contour = cnt
                                        break
                                
                                # 3. Fallback to the largest valid contour
                                if taskboard_contour is None:
                                    taskboard_contour = max(valid_contours, key=cv2.contourArea)

                            if taskboard_contour is not None and cv2.contourArea(taskboard_contour) > 1000:
                                # self.parent_node.get_logger().info("Taskboard region targeted above cable.")
                                # Semi-transparent overlay of the full mask
                                overlay = c_img_output_cv.copy()
                                cv2.drawContours(overlay, [taskboard_contour], -1, (0, 255, 255), -1)
                                cv2.addWeighted(overlay, 0.3, c_img_output_cv, 0.7, 0, c_img_output_cv)
                                # Outline for clarity
                                cv2.drawContours(c_img_output_cv, [taskboard_contour], -1, (0, 255, 255), 2)
                                
                                cable_tip = (int(self.current_cable_center[0]), int(self.current_cable_center[1]))
                                cv2.circle(c_img_output_cv, cable_tip, 7, (255, 0, 255), -1)
                                
                                # ── NEW LEDGE DETECTION LOGIC (RIGHT 1/4th of IMAGE) ──
                                # 1. Get image width
                                img_w = c_img_cv.shape[1]
                                right_threshold = img_w * 0.75
                                
                                # 2. Filter for points in the right 1/4th of the image
                                pts = taskboard_contour.reshape(-1, 2)
                                right_mask = (pts[:, 0] > right_threshold)
                                right_points = pts[right_mask]
                                
                                if len(right_points) > 5:
                                    # 3. Find the stable horizontal "ledge" in the right section
                                    y_vals = np.sort(right_points[:, 1])
                                    num_pts = min(len(y_vals), 10)
                                    lowest_black_y = np.mean(y_vals[-num_pts:])
                                    
                                    # Draw the detected ledge for visualization
                                    cv2.line(c_img_output_cv, (int(right_threshold), int(lowest_black_y)), 
                                             (img_w, int(lowest_black_y)), (255, 255, 0), 2)
                                    cv2.putText(c_img_output_cv, "RIGHT LEDGE", (int(right_threshold), int(lowest_black_y) - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                else:
                                    # Fallback to absolute max of the whole contour if right side is empty
                                    lowest_black_y = np.max(taskboard_contour[:, :, 1])

                                gap = cable_tip[1] - lowest_black_y
                                
                                # Draw points and lines
                                cv2.circle(c_img_output_cv, cable_tip, 5, (255, 0, 255), -1) # Magenta Tip
                                cv2.putText(c_img_output_cv, "TIP", (cable_tip[0] + 5, cable_tip[1] - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                                
                                taskboard_min_pt = (cable_tip[0], int(lowest_black_y))
                                cv2.circle(c_img_output_cv, taskboard_min_pt, 5, (0, 0, 255), -1) # Red Min Point
                                cv2.putText(c_img_output_cv, "BOARD_MIN", (taskboard_min_pt[0] + 5, taskboard_min_pt[1] - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                
                                cv2.line(c_img_output_cv, taskboard_min_pt, cable_tip, (0, 255, 0), 2)
                                cv2.putText(c_img_output_cv, f"GAP: {gap}", (cable_tip[0] + 10, cable_tip[1] + 15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                                if gap > 20:
                                    pass
                                    # self.parent_node.get_logger().info("[PHASE 1.5] Valid white gap detected")
                                    # self.taskboard_gap_ready = True
                                else:
                                    # self.parent_node.get_logger().info("[PHASE 1.5] Cable too close to taskboard")
                                    self.taskboard_gap_ready = False
                                    vy = 0.051

                                # Move to the right of the rightmost port by some offset
                                if len(self.ports_center) > 0:
                                    rightmost_port_x = max([p["center"][0] for p in self.ports_center])
                                    target_x = rightmost_port_x + 100  # 60 pixel offset to the right
                                    ex_h = target_x - cable_tip[0]
                                
                                    vx = float(np.clip(0.01 * ex_h, -0.051, 0.051))
                                    
                                    # Visual debugging for horizontal target
                                    cv2.line(c_img_output_cv, (int(cable_tip[0]), int(cable_tip[1])), (int(target_x), int(cable_tip[1])), (0, 0, 255), 2)
                                    cv2.circle(c_img_output_cv, (int(target_x), int(cable_tip[1])), 5, (0, 0, 255), -1)
                                    cv2.putText(c_img_output_cv, "RIGHT_TARGET", (int(target_x) + 5, int(cable_tip[1])), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                                if ex_h < 20 and gap > 20:
                                    self.taskboard_gap_ready = True

                if self.taskboard_gap_ready and not self.pink_region_found:
                    # rotate untill pink region is found in the image
                    hsv = cv2.cvtColor(c_img_cv, cv2.COLOR_BGR2HSV)
                    lower_pink = np.array([135, 100, 100])
                    upper_pink = np.array([165, 255, 255])
                    mask = cv2.inRange(hsv, lower_pink, upper_pink)
                    
                    if cv2.countNonZero(mask) > 10:
                        self.parent_node.get_logger().info("[PHASE 1.5] Pink region detected. Transitioning to Phase 2.")
                        self.pink_region_found = True
                        wx = 0.0
                    else:
                        wx = 0.3 # Gentle roll to find the pink markers
                        self.initial_pitch += 1
                        
                    # Visualization of pink mask for debugging (manual blending to avoid OpenCV errors)
                    if cv2.countNonZero(mask) > 0:
                        pink_color = np.array([255, 0, 255], dtype=np.uint8)
                        c_img_output_cv[mask > 0] = (c_img_output_cv[mask > 0] * 0.5 + pink_color * 0.5).astype(np.uint8)

                if self.taskboard_gap_ready and self.pink_region_found:
                    self.ready_to_scan = True
                    
                    if self.ready_to_scan:
                        # 1. Get features and group ports
                        if self.current_cable_center is not None and len(self.current_cable_center) >= 2:
                            cable_tip_y = self.current_cable_center[1]
                            tcp_z = obs.controller_state.tcp_pose.position.z
                            

                            # 2. ONE-SHOT IDENTIFICATION (Requires Pink Marker)
                            if not self.rails_identified and groups:
                                pink_coords = np.where(mask > 0)
                                if pink_coords[0].size > 0:
                                    pink_lowest_y = np.max(pink_coords[0])
                                    d_val = cable_tip_y - pink_lowest_y
                                    obs_dists = []
                                    group_info = []
                                    for group in groups:
                                        gy = np.mean([p["center"][1] for p in group])
                                        gx = np.mean([p["center"][0] for p in group])
                                        dist = gy - cable_tip_y
                                        obs_dists.append(dist)
                                        group_info.append((gx, gy))
                                    
                                    if self.rail_coefficients is not None:
                                        # Use internal predict method
                                        preds = self.predict_rail(tcp_z, self.current_target_rail, obs_dists)
                                        for i, rid in enumerate(preds):
                                            self.identified_rails[int(rid)] = group_info[i]
                                    
                                    if len(self.identified_rails) > 0:
                                        self.rails_identified = True



                            # 3. CONTINUOUS TRACKING (Only Requires Ports)
                            if self.rails_identified and groups:
                                new_assignments = {}
                                for group in groups:
                                    gy = np.mean([p["center"][1] for p in group])
                                    gx = np.mean([p["center"][0] for p in group])
                                    best_rid = None
                                    min_dist = 1000.0
                                    for rid, last_coords in self.identified_rails.items():
                                        dist = abs(gy - last_coords[1])
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_rid = rid
                                    if best_rid is not None and min_dist < 50:
                                        new_assignments[best_rid] = (gx, gy)
                                
                                for rid, coords in new_assignments.items():
                                    self.identified_rails[rid] = coords
                                    if rid == self.current_target_rail:
                                        self.target_port_found = True
                                        self.target_rail_y = coords[1]


                                # 3. TARGET PORT TRACKING & OCCLUSION HANDLING
                                if self.target_port_found:
                                    # Get current rail center from tracking
                                    rgx, rgy = self.identified_rails[self.current_target_rail]
                                    
                                    # Find ports belonging to the target rail
                                    ports_on_target_rail = [p for p in all_detected_ports if abs(p["center"][1] - self.target_rail_y) < 20]
                                    
                                    if len(ports_on_target_rail) > 0:
                                        # Sort by X coordinate: [Rightmost, ..., Leftmost]
                                        sorted_p = sorted(ports_on_target_rail, key=lambda x: x["center"][0], reverse=True)
                                        
                                        target_port = None
                                        ref_port = None
                                        
                                        if len(sorted_p) == 2:
                                            # Both visible: Collect ratio samples
                                            if self.target_port_index == 0: # Right
                                                target_port, ref_port = sorted_p[0], sorted_p[1]
                                            else: # Left
                                                target_port, ref_port = sorted_p[1], sorted_p[0]
                                            
                                            t_top = np.array(target_port["center"], dtype=np.float32)
                                            r_top = np.array(ref_port["center"], dtype=np.float32)
                                            
                                            # Calculate current ratio relative to reference port height
                                            # Height = Top-Left to Bottom-Left distance
                                            ref_height = np.linalg.norm(ref_port["corners"][3] - ref_port["corners"][0])
                                            
                                            if ref_height > 0 and not self.is_ratio_locked:
                                                dx = t_top[0] - r_top[0]
                                                dy = t_top[1] - r_top[1]
                                                self.dx_ratio_history.append(dx / ref_height)
                                                self.dy_ratio_history.append(dy / ref_height)
                                                
                                                if len(self.dx_ratio_history) >= self.required_offset_samples:
                                                    avg_dx = sum(self.dx_ratio_history) / len(self.dx_ratio_history)
                                                    avg_dy = sum(self.dy_ratio_history) / len(self.dy_ratio_history)
                                                    self.saved_normalized_offset = [avg_dx, avg_dy]
                                                    self.is_ratio_locked = True
                                                    self.parent_node.get_logger().info(f"[LOCKED] Virtual Target Ratio: dx={avg_dx:.3f}, dy={avg_dy:.3f}")
                                            
                                            self.current_target_top_edge = t_top
                                            
                                        elif len(sorted_p) == 1:
                                            p = sorted_p[0]
                                            p_top = np.array(p["center"], dtype=np.float32)
                                            
                                            # Use distance from previous frame to identify the port
                                            is_target_candidate = False
                                            if self.prev_target_top_edge is not None:
                                                dist_to_last = np.linalg.norm(p_top - self.prev_target_top_edge)
                                                if dist_to_last < self.MAX_PORT_JUMP:
                                                    is_target_candidate = True
                                            else:
                                                # Fallback to rail-center logic only for the very first frame
                                                is_right_of_center = (p["center"][0] > rgx)
                                                if self.target_port_index == 0 and is_right_of_center: is_target_candidate = True
                                                if self.target_port_index == 1 and not is_right_of_center: is_target_candidate = True

                                            if is_target_candidate:
                                                target_port = p
                                                self.current_target_top_edge = p_top
                                            elif self.saved_normalized_offset is not None:
                                                # This must be the reference port! Predict the target.
                                                ref_port = p
                                                r_top = p_top
                                                curr_ref_height = np.linalg.norm(ref_port["corners"][3] - ref_port["corners"][0])
                                                
                                                pred_x = r_top[0] + (self.saved_normalized_offset[0] * curr_ref_height)
                                                pred_y = r_top[1] + (self.saved_normalized_offset[1] * curr_ref_height)
                                                self.current_target_top_edge = np.array([pred_x, pred_y])
                                                
                                                # Mark as predicted (Orange circle)
                                                cv2.circle(c_img_output_cv, (int(pred_x), int(pred_y)), 8, (0, 165, 255), -1)
                                                cv2.putText(c_img_output_cv, "V-TARGET", (int(pred_x) + 10, int(pred_y)),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                            else:
                                                # We have only one port but no ratio yet - stay on this port but don't lock
                                                self.current_target_top_edge = p_top

                                        # Update state for next frame
                                        if hasattr(self, 'current_target_top_edge'):
                                            self.prev_target_top_edge = self.current_target_top_edge
                                            self.current_best_port_center = {"center": self.current_target_top_edge}
                                            
                    # Persistent visualization of tracked rails
                    img_w = c_img_output_cv.shape[1]
                    for rid, coords in self.identified_rails.items():
                        gx, gy = coords
                        is_target = (rid == self.current_target_rail)
                        # Darker red for target rail line (0, 0, 150)
                        color = (0, 0, 150) if is_target else (0, 255, 0)
                        thickness = 3 if is_target else 1
                        
                        cv2.line(c_img_output_cv, (0, int(gy)), (img_w, int(gy)), color, thickness)
                        label = f"TRACKED: CARD {rid}" if self.rails_identified else f"PREDICTING: CARD {rid}"
                        if is_target: label = "TARGET: " + label
                        cv2.putText(c_img_output_cv, label, (10, int(gy) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                                    
                    # Draw a mark on the target port's top edge center if identified
                    if hasattr(self, 'current_target_top_edge') and self.target_port_found:
                        tx, ty = self.current_target_top_edge
                        cv2.drawMarker(c_img_output_cv, (int(tx), int(ty)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)



                    # Draw cable and pink references for visual confirmation
                    if self.current_cable_center is not None and len(self.current_cable_center) >= 2:
                        cable_tip_y = self.current_cable_center[1]
                        pink_coords = np.where(mask > 0)
                        if pink_coords[0].size > 0:
                            pink_lowest_y = np.max(pink_coords[0])
                            cv2.line(c_img_output_cv, (0, int(cable_tip_y)), (img_w, int(cable_tip_y)), (255, 0, 0), 2)
                            cv2.line(c_img_output_cv, (0, int(pink_lowest_y)), (img_w, int(pink_lowest_y)), (255, 0, 255), 2)

                # ── PHASE 1.8: TARGET RAIL YAW FIX ─────────────────────────────────
                if self.yaw_fixed and self.yaw_180_fixed and self.target_port_found and not self.target_rail_yaw_fixed:
                    # Find ports belonging to the target rail
                    ports_on_target_rail = [p for p in self.ports_center if abs(p["center"][1] - self.target_rail_y) < 20]
                    
                    if len(ports_on_target_rail) == 2:
                        p1 = ports_on_target_rail[0]["center"]
                        p2 = ports_on_target_rail[1]["center"]
                        target_yaw = self.calculate_two_port_angle(p1, p2)
                        
                        self.parent_node.get_logger().info(f"[PHASE 1.8] Target Rail Yaw: {target_yaw:.4f}")
                        
                        if abs(target_yaw) <= 0.005:
                            self.target_rail_yaw_fixed = True
                            self.parent_node.get_logger().info("[PHASE 1.8] Target Rail Yaw Fix complete.")
                        else:
                            wz = target_yaw * 1.5 # Proportional control
                            wz = float(np.clip(wz, -0.3, 0.3))
                    else:
                        # If we don't see both ports, we can't fix yaw based on the rail line.
                        # Skip for now or wait?
                        # User said "find angle made by the line joining the ports of the target rail"
                        # So we must see both.
                        pass

                # ── PHASE 2: PLANAR VISUAL SERVOING ─────────────────────────────────
                if self.yaw_fixed and self.yaw_180_fixed and self.target_port_found and self.target_rail_yaw_fixed:
                    ex, ey = None, None
                    
                    # TRACKING LOGIC FROM TEMPU.PY
                    
                    # 1. Update rail Y-tracking (Continuous)
                    if self.rails_identified and groups:
                        new_assignments = {}
                        for group in groups:
                            gy = np.mean([p["center"][1] for p in group])
                            gx = np.mean([p["center"][0] for p in group])
                            best_rid = None
                            min_dist = 1000.0
                            for rid, last_coords in self.identified_rails.items():
                                dist = abs(gy - last_coords[1])
                                if dist < min_dist:
                                    min_dist = dist
                                    best_rid = rid
                            if best_rid is not None and min_dist < 50:
                                new_assignments[best_rid] = (gx, gy)
                        
                        for rid, coords in new_assignments.items():
                            self.identified_rails[rid] = coords
                            if rid == self.current_target_rail:
                                self.target_rail_y = coords[1]

                    # 2. Find target port top edge
                    if hasattr(self, 'target_rail_y'):
                        ports_on_target_rail = [p for p in all_detected_ports if abs(p["center"][1] - self.target_rail_y) < 25]
                        
                        if len(ports_on_target_rail) > 0:
                            # Sort by X: [Rightmost, ..., Leftmost]
                            sorted_p = sorted(ports_on_target_rail, key=lambda x: x["center"][0], reverse=True)
                            
                            if len(sorted_p) == 2:
                                # Both visible: Ratio lock
                                if self.target_port_index == 0: # Right
                                    target_port, ref_port = sorted_p[0], sorted_p[1]
                                else: # Left
                                    target_port, ref_port = sorted_p[1], sorted_p[0]
                                
                                t_top = np.array(target_port["center"], dtype=np.float32)
                                r_top = np.array(ref_port["center"], dtype=np.float32)
                                ref_height = np.linalg.norm(ref_port["corners"][3] - ref_port["corners"][0])
                                
                                if ref_height > 0 and not self.is_ratio_locked:
                                    dx = t_top[0] - r_top[0]
                                    dy = t_top[1] - r_top[1]
                                    self.dx_ratio_history.append(dx / ref_height)
                                    self.dy_ratio_history.append(dy / ref_height)
                                    if len(self.dx_ratio_history) >= self.required_offset_samples:
                                        self.saved_normalized_offset = [np.mean(self.dx_ratio_history), np.mean(self.dy_ratio_history)]
                                        self.is_ratio_locked = True
                                        self.parent_node.get_logger().info(f"[SFP] Locked offset: {self.saved_normalized_offset}")
                                
                                self.current_target_top_edge = t_top
                                
                            elif len(sorted_p) == 1:
                                p = sorted_p[0]
                                p_top = np.array(p["center"], dtype=np.float32)
                                
                                is_target_candidate = False
                                if self.prev_target_top_edge is not None:
                                    dist_to_last = np.linalg.norm(p_top - self.prev_target_top_edge)
                                    if dist_to_last < self.MAX_PORT_JUMP:
                                        is_target_candidate = True
                                else:
                                    # Fallback to rail center logic
                                    rgx, rgy = self.identified_rails[self.current_target_rail]
                                    if (self.target_port_index == 0 and p["center"][0] > rgx) or (self.target_port_index == 1 and p["center"][0] < rgx):
                                        is_target_candidate = True

                                if is_target_candidate:
                                    self.current_target_top_edge = p_top
                                elif self.saved_normalized_offset is not None:
                                    # Prediction!
                                    r_top = p_top
                                    curr_ref_height = np.linalg.norm(p["corners"][3] - p["corners"][0])
                                    pred_x = r_top[0] + (self.saved_normalized_offset[0] * curr_ref_height)
                                    pred_y = r_top[1] + (self.saved_normalized_offset[1] * curr_ref_height)
                                    self.current_target_top_edge = np.array([pred_x, pred_y])
                                    cv2.circle(c_img_output_cv, (int(pred_x), int(pred_y)), 8, (0, 165, 255), -1)
                                    cv2.putText(c_img_output_cv, "V-TARGET", (int(pred_x) + 10, int(pred_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                else:
                                    self.current_target_top_edge = p_top

                            # Update state for next frame
                            self.prev_target_top_edge = self.current_target_top_edge
                            self.virtual_target_debug_point = self.current_target_top_edge

                    # Calculate Errors
                    if self.virtual_target_debug_point is not None and len(self.current_cable_center) >= 2:
                        ex = float(self.virtual_target_debug_point[0] - self.current_cable_center[0])
                        ey = float(self.virtual_target_debug_point[1] - self.current_cable_center[1])
                        
                        # Draw error vector
                        cv2.line(c_img_output_cv, 
                                (int(self.current_cable_center[0]), int(self.current_cable_center[1])),
                                (int(self.virtual_target_debug_point[0]), int(self.virtual_target_debug_point[1])),
                                (0, 255, 0), 2)
                        cv2.putText(c_img_output_cv, f"ex:{ex:.1f} ey:{ey:.1f}", 
                                    (int(self.current_cable_center[0]), int(self.current_cable_center[1]) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    Kp_xy        = 0.001
                    XY_TOLERANCE = 15.0  # Generous tolerance so Z-descent actually triggers
                    vx = vy = vz = 0.0
                    wx = wy = wz = 0.0

                    if ex is not None and ey is not None:
                        
                        error = math.sqrt(ex**2 + ey**2)
                        # Dynamic Speed Profile: Fast approach, slow and precise final alignment
                        max_speed = 0.05
                        kp_xy     = 0.0005
                        max_speed_z = 0.01
                        vx = float(np.clip(Kp_xy * ex, -max_speed, max_speed))
                        vy = float(np.clip(Kp_xy * ey, -max_speed, max_speed))
                        if error < 20:
                            kp_z = 0.001
                            vz = float(np.clip(kp_z * (50 - error), -max_speed_z, max_speed_z))
                        else:
                            vz = 0.0

                        # =====================================================
                        # SFP ROLL/PITCH MODEL INFERENCE
                        # =====================================================
                        if self.is_contact_made:
                            try:
                                from transforms3d.euler import quat2euler
                                q = obs.controller_state.tcp_pose.orientation
                                curr_roll, curr_pitch, _ = quat2euler(
                                    [q.w, q.x, q.y, q.z]
                                )
                                test_features = [
                                    np.sin(curr_roll),
                                    np.cos(curr_roll),
                                ]
                                predicted_error = self.roll_model_sfp.predict(
                                    [test_features]
                                )[0]
                                p_raw = predicted_error
                                predicted_error = math.trunc(predicted_error * 10000) / 10000
                                self.parent_node.get_logger().info(f"[SFP] Predicted error: {predicted_error}")

                                # if abs(predicted_error) < 0.0005: 
                                #     self.parent_node.get_logger().info("[SFP] Predicted error < 0.0005. Returning.")
                                #     return True
                                if self.prev_predicted_error_pitch == predicted_error:
                                    self.pitch_counter = self.pitch_counter + 1
                                else:
                                    self.pitch_counter = 0
                                if self.pitch_counter > 100:
                                    self.flush_stored_energy(obs, move_robot)
                                    self.sleep_for(0.25)
                                    return True
                                self.prev_predicted_error_pitch = predicted_error
                                wx = -0.08
                                if abs(p_raw) < 0.1:
                                    vz = 0.01
                                    vy = -0.05
                                    # wx = -0.03
                                else:
                                    vz= 0.02
                                    vy = -0.001
                                self.parent_node.get_logger().info(
                                    f"[SFP MODEL] pitch_error={predicted_error:.4f}, wx={wx:.4f}"
                                )
                            except Exception as e:
                                self.parent_node.get_logger().warn(
                                    f"[SFP MODEL ERROR] {e}"
                                )

                                
                            # ----------------------------
                # ── Send Cartesian command ───────────────────────────────────────────
                twist          = Twist()
                twist.linear.x = vx
                twist.linear.y = vy
                twist.linear.z = vz
                twist.angular.x = wx
                twist.angular.y = wy
                twist.angular.z = wz
                self.last_valid_twist = twist
                #self.parent_node.get_logger().info(f"[SFP] Servoing: ex={ex:.2f}, ey={ey:.2f}, error={error:.2f}, vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f}")

                # Publish final annotated image
                c_img_output = self._cv2_to_msg(c_img_output_cv, obs.center_image.header)
                self.pub_output_cam_center.publish(c_img_output)

                mu = MotionUpdate()
                mu.header.stamp                    = self.parent_node.get_clock().now().to_msg()
                mu.header.frame_id                 = "gripper/tcp"
                mu.velocity                        = twist
                mu.target_stiffness                = np.diag(self.current_stiffness).flatten().tolist()
                mu.target_damping                  = np.diag(self.current_damping).flatten().tolist()
                mu.feedforward_wrench_at_tip       = Wrench()
                mu.wrench_feedback_gains_at_tip    = [0.0] * 6
                mu.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

                move_robot(motion_update=mu)
                
                tcp_vel = obs.controller_state.tcp_velocity
                sum_vel = abs(tcp_vel.linear.x) + abs(tcp_vel.linear.y) + abs(tcp_vel.linear.z) + \
                          abs(tcp_vel.angular.x) + abs(tcp_vel.angular.y) + abs(tcp_vel.angular.z)


                if sum_vel <= 0.001:
                    #return True
                    if self.zero_vel_start_time is None:
                        self.zero_vel_start_time = self.time_now()
                    elif (self.time_now() - self.zero_vel_start_time).nanoseconds / 1e9 >= 1.0:
                        self.parent_node.get_logger().info("[SFP] Zero velocity sustained for 2s. Returning.")
                        if self.yaw_fixed and self.yaw_180_fixed and self.target_port_found and self.target_rail_yaw_fixed:
                            self.is_contact_made = True
                            self.parent_node.get_logger().info("[SFP] Contact Made.")
                        elif (self.time_now() - self.zero_vel_start_time).nanoseconds / 1e9 >= 3.0:
                            self.flush_stored_energy(obs, move_robot)
                            self.sleep_for(0.25)
                            return True

                        # self.reset_state()
                        # return True
                else:
                    self.zero_vel_start_time = None
            obs = get_observation()
            if obs is not None:
                self.flush_stored_energy(obs, move_robot)
            return True
        
        elif task.port_type == "sc":
            self.parent_node.get_logger().info("Starting SC (Blue) Insertion Loop")
            while self.time_now() - start_time < timeout:
                obs = get_observation()
                if obs is None:
                    self.sleep_for(0.01)
                    continue
    
                current_stamp = obs.center_image.header.stamp
                if getattr(self, "last_seen_stamp", None) is not None:
                    if current_stamp.sec == self.last_seen_stamp.sec and \
                       current_stamp.nanosec == self.last_seen_stamp.nanosec:
                        self.sleep_for(0.01)
                        continue
                    
                self.last_seen_stamp = current_stamp
            
                c_img_cv = self._msg_to_cv2(obs.center_image)
                c_img_output_cv = c_img_cv.copy()
            
                # Detect Blue Ports and Cable on Center Camera
                results_blue = self.model_blue(c_img_cv, conf=0.4, verbose=False, device=self.device)
                obb_blue = results_blue[0].obb
                
                # Detect and Publish on Left Camera
                if hasattr(obs, 'left_image') and obs.left_image is not None:
                    l_img_cv = self._msg_to_cv2(obs.left_image)
                    l_img_output_cv = l_img_cv.copy()
                    results_blue_left = self.model_blue(l_img_cv, conf=0.4, verbose=False, device=self.device)
                    obb_blue_left = results_blue_left[0].obb
                    
                    if obb_blue_left is not None and len(obb_blue_left) > 0:
                        for i in range(len(obb_blue_left)):
                            pts_l = obb_blue_left.xyxyxyxy[i].cpu().numpy().astype(int)
                            cls_l = int(obb_blue_left.cls[i])
                            if cls_l == 1:  # Port
                                import cv2
                                cv2.polylines(l_img_output_cv, [pts_l], True, (0, 255, 0), 2)
                            elif cls_l == 0 and getattr(self, 'last_known_cable_blue', None) is None:  # Cable
                                import cv2
                                cv2.polylines(l_img_output_cv, [pts_l], True, (255, 0, 0), 2)
                                
                    l_img_output = self._cv2_to_msg(l_img_output_cv, obs.left_image.header)
                    self.pub_output_cam_left.publish(l_img_output)
            
                ports_blue = []
                cable_blue = None
            
                if obb_blue is not None and len(obb_blue) > 0:
                    for i in range(len(obb_blue)):
                        pts = obb_blue.xyxyxyxy[i].cpu().numpy().astype(int)
                        cls = int(obb_blue.cls[i])
                        conf = float(obb_blue.conf[i])
                        if cls == 1:  # Port
                            ports_blue.append(pts)
                            cv2.polylines(c_img_output_cv, [pts], True, (0, 255, 0), 2)
                        elif cls == 0 and getattr(self, 'last_known_cable_blue', None) is None:  # Cable
                            cable_blue = pts
                            cv2.polylines(c_img_output_cv, [pts], True, (255, 0, 0), 2)
                        
                # Cable Caching Logic for SC
                if cable_blue is not None:
                    self.last_known_cable_blue = cable_blue
                elif self.last_known_cable_blue is not None:
                    cable_blue = self.last_known_cable_blue
                    cv2.polylines(c_img_output_cv, [cable_blue], True, (0, 165, 255), 2)
                    cv2.putText(c_img_output_cv, "C_CACHED", (cable_blue[0][0], cable_blue[0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                        
                # ── PHASE 0: YAW FIX (Align short edge horizontally) ──
                if self.once_fix_yaw_blue:
                    if len(ports_blue) > 0:
                        pts = ports_blue[0]
                        # Find the angle of the shorter edge
                        d01 = np.linalg.norm(pts[0] - pts[1])
                        d12 = np.linalg.norm(pts[1] - pts[2])
                    
                        if d01 < d12:
                            pA, pB = pts[0], pts[1]
                        else:
                            pA, pB = pts[1], pts[2]
                        
                        dx = pB[0] - pA[0]
                        dy = pB[1] - pA[1]
                        ang = math.atan2(dy, dx)
                    
                        # Normalize angle to [-pi/2, pi/2]
                        if ang > math.pi / 2: ang -= math.pi
                        elif ang < -math.pi / 2: ang += math.pi
                    
                        self.parent_node.get_logger().info(f"[BLUE] Short edge angle: {ang:.4f} rad")
                    
                        if abs(ang) <= 0.02:
                            self.wait_yaw_fix_blue = False
    
                        if not self.wait_yaw_fix_blue:
                            jmu = self._make_joint_vel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                            move_robot(joint_motion_update=jmu)
                            self.parent_node.get_logger().info("[BLUE YAW FIX] Complete.")
                            self.wait_yaw_fix_blue = True
                            self.once_fix_yaw_blue = False
                            self.yaw_fixed_blue = True
                        else:
                            # Rotate wrist-3 to minimize angle
                            # Proportional control for rotation
                            rot_vel = float(np.clip(-1.5 * ang, -0.31, 0.31))
                            jmu = self._make_joint_vel([0.0, 0.0, 0.0, 0.0, 0.0, -rot_vel])
                            move_robot(joint_motion_update=jmu)
                    else:
                        self.parent_node.get_logger().info("[BLUE YAW FIX] Waiting for port detection...")
                    
                    c_img_output = self._cv2_to_msg(c_img_output_cv, obs.center_image.header)
                    self.pub_output_cam_center.publish(c_img_output)
                    continue
    
                # ── PHASE 1: YAW 180 FIX (using pink region) ──
                if self.yaw_fixed_blue and not self.yaw_180_fixed_blue:
                    hsv = cv2.cvtColor(c_img_cv, cv2.COLOR_BGR2HSV)
                    lower_pink = np.array([135, 100, 100])
                    upper_pink = np.array([165, 255, 255])
                    mask = cv2.inRange(hsv, lower_pink, upper_pink)
                    M = cv2.moments(mask)
                
                    if M["m00"] > 0:
                        pink_cy = int(M["m01"] / M["m00"])
                    
                        if len(ports_blue) > 0:
                            port_cy = self.points_center(ports_blue[0])[1]
                        
                            if pink_cy > port_cy:
                                self.parent_node.get_logger().info("[BLUE] Upside down! Rotating 180 degrees.")
                                current_joints = list(obs.joint_states.position)
                                current_names = list(obs.joint_states.name)
                            
                                arm_joint_names = [
                                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
                                ]
                            
                                target_positions = []
                                for j_name in arm_joint_names:
                                    if j_name in current_names:
                                        target_positions.append(current_joints[current_names.index(j_name)])
                                    else:
                                        target_positions.append(0.0)
                                    
                                target_positions[5] -= math.pi
                                jmu = self._make_joint_pos(target_positions)
                                move_robot(joint_motion_update=jmu)
                                self.sleep_for(4.0)
                                self.parent_node.get_logger().info("[BLUE] 180 degree correction complete.")
                            else:
                                self.parent_node.get_logger().info("[BLUE] Orientation is correct.")
                            
                            self.yaw_180_fixed_blue = True
                        else:
                            self.parent_node.get_logger().info("[BLUE 180 YAW] Waiting for port...")
                    else:
                        self.parent_node.get_logger().info("[BLUE 180 YAW] Pink region not found, assuming ok.")
                        self.yaw_180_fixed_blue = True
                    
                    c_img_output = self._cv2_to_msg(c_img_output_cv, obs.center_image.header)
                    self.pub_output_cam_center.publish(c_img_output)
                
                    if not self.yaw_180_fixed_blue:
                        continue
    
                # ── PHASE 2: PLANAR VISUAL SERVOING ──
                if self.yaw_fixed_blue and self.yaw_180_fixed_blue:
                    # 1. Find pink region corners and size
                    hsv = cv2.cvtColor(c_img_cv, cv2.COLOR_BGR2HSV)
                    lower_pink = np.array([135, 100, 100])
                    upper_pink = np.array([165, 255, 255])
                    mask = cv2.inRange(hsv, lower_pink, upper_pink)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                    pink_corners = None
                    pink_size = None
                    if len(contours) > 0:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) > 50:
                            rect = cv2.minAreaRect(c)
                            box = cv2.boxPoints(rect)
                            pink_corners = self.order_corners(np.asarray(box, dtype=np.float32))
                            pink_size = max(rect[1][0], rect[1][1])
                            if pink_size < 1: pink_size = None
                            # Draw pink corners for debugging
                            for pt in pink_corners:
                                cv2.circle(c_img_output_cv, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
    
                    target_port_pts = None
                    if len(ports_blue) == 1:
                        target_port_pts = ports_blue[0]
                    elif len(ports_blue) >= 2:
                        ports_blue_sorted = sorted(ports_blue, key=lambda p: self.points_center(p)[1])
                        if task.port_name == "sc_port_0":
                            target_port_pts = ports_blue_sorted[-1]  # Bottom one
                        else:
                            target_port_pts = ports_blue_sorted[0]   # Top one
                        
                    port_target = None
                    nearest_corner = None
                
                    if target_port_pts is not None:
                        ordered_port = self.order_corners(target_port_pts)
                        # Aim 20% down from the top edge center towards the bottom edge center to handle camera tilt
                        top_c = self.top_edge_center(ordered_port)
                        bottom_c = (ordered_port[2] + ordered_port[3]) / 2.0
                        actual_port_target = top_c + 0.65 * (bottom_c - top_c)
                    
                        # Distance jump check to prevent snapping to wrong port during occlusion
                        if self.prev_target_port_center_blue is not None:
                            dist = np.linalg.norm(actual_port_target - self.prev_target_port_center_blue)
                            if dist > 60.0:  # 60 pixels jump threshold
                                target_port_pts = None
                                self.parent_node.get_logger().warn(f"[BLUE] Large jump ({dist:.1f}px) detected! Forcing predict mode.")
                            
                    if target_port_pts is not None:
                        self.prev_target_port_center_blue = actual_port_target
                        port_target = actual_port_target
                    
                        if pink_corners is not None and len(pink_corners) >= 4:
                            p_bl = pink_corners[3]
                            p_br = pink_corners[2]
                            D = np.linalg.norm(p_br - p_bl)
                        
                            if D > 5.0:
                                dx = actual_port_target[0] - p_bl[0]
                                dy = actual_port_target[1] - p_bl[1]
                            
                                self.dx_ratio_history_blue.append(dx / D)
                                self.dy_ratio_history_blue.append(dy / D)
                            
                                if len(self.dx_ratio_history_blue) > self.required_offset_samples_blue:
                                    self.dx_ratio_history_blue.pop(0)
                                    self.dy_ratio_history_blue.pop(0)
                                
                                avg_dx = sum(self.dx_ratio_history_blue) / len(self.dx_ratio_history_blue)
                                avg_dy = sum(self.dy_ratio_history_blue) / len(self.dy_ratio_history_blue)
                                self.saved_normalized_offset_blue = [avg_dx, avg_dy]
                            
                            nearest_corner = p_bl
                    else:
                        # Switch to predicting mode
                        if pink_corners is not None and len(pink_corners) >= 4 and self.saved_normalized_offset_blue is not None:
                            p_bl = pink_corners[3]
                            p_br = pink_corners[2]
                            D = np.linalg.norm(p_br - p_bl)
                            pred_x = p_bl[0] + self.saved_normalized_offset_blue[0] * D
                            pred_y = p_bl[1] + self.saved_normalized_offset_blue[1] * D
                            self.locked_port_target_blue = [pred_x, pred_y]
                            self.port_occluded_blue = True
                            self.parent_node.get_logger().info(f"[BLUE] Port occluded! Locking target at ({pred_x:.1f}, {pred_y:.1f})")
                    
                        if self.locked_port_target_blue is not None:
                            port_target = self.locked_port_target_blue
                            cv2.circle(c_img_output_cv, (int(port_target[0]), int(port_target[1])), 8, (0, 0, 255), -1)
                            cv2.putText(c_img_output_cv, "LOCKED", (int(port_target[0]) + 10, int(port_target[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            self.parent_node.get_logger().warn("[BLUE] Port lost, and no locked target available!")
    
                    if port_target is not None and cable_blue is not None:
                        ordered_cable = self.order_corners(cable_blue)
                        cab_center = self.top_edge_center(ordered_cable)
                    
                        ex = port_target[0] - cab_center[0]
                        ey = port_target[1] - cab_center[1]
                        error_xy = math.sqrt(ex**2 + ey**2)
                            
                        if not self.sc_aligned:
                            Kp_xy = 0.00051
                            vx = float(np.clip(Kp_xy * ex, -0.05, 0.05))
                            vy = float(np.clip(Kp_xy * ey, -0.05, 0.05))
                        else:
                            Kp_xy = 0.001
                            vx = float(np.clip(Kp_xy * ex, -0.05, 0.05))
                            vy = float(np.clip(Kp_xy * ey, -0.05, 0.05))
    
                            
                        # Once error is small enough, lock the alignment flag
                        if error_xy <= 40.0:
                            self.sc_aligned = True
                            
                        # Calculate vz: 
                        # 1. Stop if contact is made
                        # 2. If aligned, descend! Speed is max 0.02, scaling down to a minimum of 0.005 if error gets large
                        # 3. If not yet aligned, hover (0.0)
                        if self.is_contact_made:
                            #return
                            #vx = float(np.sign(ex)*0.5)
                            #vy = float(np.sign(ey)*0.5)
                            vz = -0.0001
                            self.blue_fix_roll = True
                            self.is_contact_made = False
                        elif self.sc_aligned:
                            # Modulate vz: drops gracefully as error grows, but never stops descending
                            vz = float(np.clip(0.15 - (error_xy * 0.005), 0.001, 0.02))
                        else:
                            vz = 0.0
                            
                        # --- ROLL MODEL INFERENCE ---
                        
                        wx = 0.0
                        if self.roll_model is not None:
                            from transforms3d.euler import quat2euler
                            q = obs.controller_state.tcp_pose.orientation
                            curr_roll, _, _ = quat2euler([q.w, q.x, q.y, q.z])
                            
                            test_sin = np.sin(curr_roll)
                            test_cos = np.cos(curr_roll)
                            predicted_error = self.roll_model.predict([[test_sin, test_cos]])[0]
                            
                            
                            Kp_roll = 0.5
                            wx = float(np.clip(Kp_roll * predicted_error, -0.1, 0.1))
                            self.parent_node.get_logger().info(f"[BLUE] Predicted roll error: {predicted_error:.4f} wx: {wx:.4f}")
                        # ----------------------------
                    
                        twist = Twist()
                        twist.linear.x = vx
                        twist.linear.y = vy
                        twist.linear.z = vz
                        twist.angular.x = wx
                    
                        mu = MotionUpdate()
                        mu.header.stamp = self.parent_node.get_clock().now().to_msg()
                        mu.header.frame_id = "gripper/tcp"
                        mu.velocity = twist
                        mu.target_stiffness = np.diag(self.current_stiffness).flatten().tolist()
                        mu.target_damping = np.diag(self.current_damping).flatten().tolist()
                        mu.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
                    
                        move_robot(motion_update=mu)
                        
                        
                        tcp_vel = obs.controller_state.tcp_velocity
                        sum_vel = abs(tcp_vel.linear.x) + abs(tcp_vel.linear.y) + abs(tcp_vel.linear.z) + \
                                  abs(tcp_vel.angular.x) + abs(tcp_vel.angular.y) + abs(tcp_vel.angular.z)
                        if sum_vel <= 0.001:
                            if self.zero_vel_start_time is None:
                                self.zero_vel_start_time = self.time_now()
                            elif (self.time_now() - self.zero_vel_start_time).nanoseconds / 1e9 >= 2.0:
                                self.is_contact_made = True
                                self.parent_node.get_logger().info("[SC] Zero velocity sustained for 2s. Returning.")
                                self.flush_stored_energy(obs, move_robot)
                                self.sleep_for(0.25)
                                return True
                        else:
                            self.zero_vel_start_time = None
                        #self.parent_node.get_logger().info(f"[BLUE] Servoing: ex={ex:.2f}, ey={ey:.2f}, vz={vz}")
                    
                        cv2.circle(c_img_output_cv, (int(port_target[0]), int(port_target[1])), 5, (0, 255, 0), -1)
                        cv2.circle(c_img_output_cv, (int(cab_center[0]), int(cab_center[1])), 5, (255, 0, 0), -1)
                        if nearest_corner is not None:
                            cv2.line(c_img_output_cv, (int(cab_center[0]), int(cab_center[1])), (int(nearest_corner[0]), int(nearest_corner[1])), (0, 255, 255), 2)
                        else:
                            cv2.line(c_img_output_cv, (int(cab_center[0]), int(cab_center[1])), (int(port_target[0]), int(port_target[1])), (0, 255, 255), 2)
    
                    else:
                        self.parent_node.get_logger().info("[BLUE] Waiting for target port or cable...")
                    
                    c_img_output = self._cv2_to_msg(c_img_output_cv, obs.center_image.header)
                    self.pub_output_cam_center.publish(c_img_output)
        
        obs = get_observation()
        if obs is not None:
            self.flush_stored_energy(obs, move_robot)
        return True

    # ==========================================
    # 2. PORT & CABLE DETECTION
    # ==========================================
    def _detect(self, img):
        # Imports
        import numpy as np
        import cv2
        
        ports_center_local = []
        cable_center_local = []
        cable_pts_local    = None
        port_starting_yaw_local = None
        output = img.copy()

        # ==========================================
        # 2. PORT & CABLE DETECTION
        # ==========================================
        results = self.model(img, conf=0.25, verbose=False, device = self.device)
        obb     = results[0].obb

        if obb is not None:
            for i in range(len(obb)):
                pts  = obb.xyxyxyxy[i].cpu().numpy().astype(int)
                conf = float(obb.conf[i])
                cls  = int(obb.cls[i])

                if cls == 0:  # PORT
                    ordered_pts = self.order_corners(pts)
                    # Use geometric center for filtering and targeting
                    cx, cy = self.points_center(ordered_pts)
                    
                    top_c = self.top_edge_center(ordered_pts)
                    mid_c = self.points_center(ordered_pts)

                    target_pt = (   
                        0.6 * top_c
                        + 0.4 * mid_c
                    )

                    tcx, tcy = float(target_pt[0]), float(target_pt[1])

                    ports_center_local.append({
                        "corners": ordered_pts,
                        "center":  [tcx, tcy],  # This is what V-Target uses
                    })
                    cv2.polylines(output, [ordered_pts.astype(int)], True, (0, 255, 255), 2)
                    for j, p in enumerate(ordered_pts):
                        cv2.circle(output, tuple(p.astype(int)), 3, (0, 0, 255), -1)
                        cv2.putText(output, str(j), tuple(p.astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                elif cls == 1:  # CABLE
                    ordered_pts = self.order_corners(pts)
                    
                    # Use top center directly (removed smoothing)
                    cable_center_local = self.top_edge_center(ordered_pts)
                    cable_pts_local    = ordered_pts

                    # CABLE LOCKING LOGIC (SFP ONLY)
                    if self.active_port_type == "sfp" and self.fixed_cable_pts is None:
                        area = cv2.contourArea(ordered_pts.astype(np.int32))
                        self.cable_collection_list.append({
                            "area": area,
                            "pts": ordered_pts.copy(),
                            "center": cable_center_local.copy(),
                        })
                        
                        if len(self.cable_collection_list) >= 20:
                            # Use median area instead of max
                            areas = [x["area"] for x in self.cable_collection_list]
                            median_area = np.median(areas)
                            # Find the sample closest to the median area
                            best = min(self.cable_collection_list, key=lambda x: abs(x["area"] - median_area))
                            
                            self.fixed_cable_pts = best["pts"]
                            self.fixed_cable_center = best["center"]
                            self.parent_node.get_logger().info(f"[CABLE LOCK] Locked cable with median area {best['area']:.2f}")

                    # Visualization
                    cv2.polylines(output, [ordered_pts.astype(int)], True, (255, 0, 0), 2)
                    cv2.putText(output, f"C:{conf:.2f}", (int(ordered_pts[0][0]), int(ordered_pts[0][1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Yaw tracking: Angle of any detected port relative to horizontal
        if self.once_fix_yaw and len(ports_center_local) > 0:
            # Pick the first detected port and find its angle relative to horizontal
            first_port_corners = ports_center_local[0]["corners"]
            port_starting_yaw_local = self.calculate_port_angle(first_port_corners)

        # Apply Cable Lock if available
        if self.active_port_type == "sfp" and self.fixed_cable_pts is not None:
            cable_pts_local = self.fixed_cable_pts
            cable_center_local = self.fixed_cable_center
            # Visual feedback for locked cable (Orange)
            cv2.polylines(output, [cable_pts_local.astype(int)], True, (0, 165, 255), 2)
            cv2.putText(output, "LOCKED", (int(cable_pts_local[0][0]), int(cable_pts_local[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        return self._draw_debug_info(output, ports_center_local, cable_center_local), ports_center_local, cable_center_local, cable_pts_local, port_starting_yaw_local
    def get_planar_errors(self):

        # Imports
        import numpy as np
        import math

        cable = self.current_cable_center
        if cable is None or len(cable) == 0:
            return None, None

        port_target  = None
        sorted_ports = []
        if self.ports_center:
            # Robust sorting: project onto the NIC card's long axis instead of just X-coordinate!
            if self.target_nic_polygon is not None and len(self.target_nic_polygon) == 4:
                theta = self.calculate_port_angle(self.target_nic_polygon)
                # calculate_port_angle returns angle of longest edge.
                # Project centers onto this vector to sort them robustly regardless of rotation!
                v = np.array([math.cos(theta), math.sin(theta)])
                sorted_ports = sorted(
                    self.ports_center,
                    key=lambda p: np.dot(self.points_center(p["corners"]), v)
                )
            else:
                # Fallback to X if NIC polygon is somehow missing
                sorted_ports = sorted(
                    self.ports_center,
                    key=lambda p: self.points_center(p["corners"])[0]
                )

        if not self.is_ratio_locked:
            if len(sorted_ports) >= 2:
                # 0 is RIGHT (largest projection), 1 is LEFT (smallest projection)
                if self.target_port_index == 0:
                    target_port = sorted_ports[-1] # Right
                    ref_port    = sorted_ports[0]  # Left
                else:
                    target_port = sorted_ports[0]  # Left
                    ref_port    = sorted_ports[-1] # Right

                ref_target    = np.array(ref_port["center"], dtype=np.float32)
                target_center = np.array(target_port["center"], dtype=np.float32)

                dx = target_center[0] - ref_target[0]
                dy = target_center[1] - ref_target[1]

                ref_height = np.linalg.norm(ref_port["corners"][3] - ref_port["corners"][0])
                if ref_height > 0:
                    self.dx_ratio_history.append(dx / ref_height)
                    self.dy_ratio_history.append(dy / ref_height)

                    if len(self.dx_ratio_history) >= self.required_offset_samples:
                        avg_dx = sum(self.dx_ratio_history) / len(self.dx_ratio_history)
                        avg_dy = sum(self.dy_ratio_history) / len(self.dy_ratio_history)
                        self.saved_normalized_offset = [avg_dx, avg_dy]
                        self.is_ratio_locked         = True
                        self.last_known_ref_center   = ref_target
                        self.last_known_target_center= target_center
                        self.parent_node.get_logger().info(f"[SFP] Target locked! Using virtual port tracking.")

                port_target = target_center

            elif len(sorted_ports) == 1:
                # If only 1 port is visible before lock, we just track it directly (fallback)
                port_target = np.array(sorted_ports[0]["center"], dtype=np.float32)

        else:
            # LOCKED MODE: Use reference port to project virtual target
            if sorted_ports and self.saved_normalized_offset is not None:
                if len(sorted_ports) >= 2:
                    if self.target_port_index == 0:
                        ref_port = sorted_ports[0]  # Left is reference for Right target
                    else:
                        ref_port = sorted_ports[-1] # Right is reference for Left target
                    ref_target = np.array(ref_port["center"], dtype=np.float32)
                    target_port = sorted_ports[-1] if self.target_port_index == 0 else sorted_ports[0]
                    self.last_known_ref_center = ref_target
                    self.last_known_target_center = np.array(target_port["center"], dtype=np.float32)
                    
                    # Both ports directly visible - use detected center directly, no projection drift!
                    port_target = self.last_known_target_center
                    self.parent_node.get_logger().debug(f"[SFP][LOCK] Both visible, using direct center.")
                    
                else:
                    # Only 1 port is visible. Is it the target port or the reference port?
                    visible_center = np.array(sorted_ports[0]["center"], dtype=np.float32)
                    
                    dist_to_ref = math.sqrt((visible_center[0] - self.last_known_ref_center[0])**2 + (visible_center[1] - self.last_known_ref_center[1])**2)
                    dist_to_target = math.sqrt((visible_center[0] - self.last_known_target_center[0])**2 + (visible_center[1] - self.last_known_target_center[1])**2)
                    
                    if dist_to_target < dist_to_ref:
                        # The visible port is the TARGET port itself! We don't need virtual tracking.
                        port_target = visible_center
                        self.last_known_target_center = visible_center
                        # Infer ref center for future frames
                        cur_height = np.linalg.norm(sorted_ports[0]["corners"][3] - sorted_ports[0]["corners"][0])
                        self.last_known_ref_center = [visible_center[0] - self.saved_normalized_offset[0] * cur_height, 
                                                      visible_center[1] - self.saved_normalized_offset[1] * cur_height]
                    else:
                        # The visible port is the REFERENCE port. Project the virtual target!
                        ref_target = visible_center
                        self.last_known_ref_center = ref_target
                        cur_height = np.linalg.norm(sorted_ports[0]["corners"][3] - sorted_ports[0]["corners"][0])
                        pred_x = ref_target[0] + self.saved_normalized_offset[0] * cur_height
                        pred_y = ref_target[1] + self.saved_normalized_offset[1] * cur_height
                        port_target = [pred_x, pred_y]
                        self.last_known_target_center = port_target
            else:
                self.parent_node.get_logger().warn(
                        "[CRITICAL] Reference port lost — cannot compute virtual target!"
                )

        if port_target is None:
            self.virtual_target_debug_point = None
            return None, None

        self.virtual_target_debug_point = port_target
        return port_target[0] - cable[0], port_target[1] - cable[1]

    # ================================================================
    # GEOMETRY HELPERS
    # ================================================================

    def calculate_port_angle(self, corners):

        # Imports
        import math
        import numpy as np

        max_dist = -1
        best_edge = None
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i+1)%4]
            dist = np.linalg.norm(p2 - p1)
            if dist > max_dist:
                max_dist = dist
                best_edge = (p1, p2)
                
        p1, p2 = best_edge
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
            
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        ang = math.atan2(dy, dx)
        if ang > math.pi / 2: ang -= math.pi
        elif ang < -math.pi / 2: ang += math.pi
        
        return ang

    def calculate_two_port_angle(self, p1, p2):
        import math
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        ang = math.atan2(dy, dx)
        if ang > math.pi / 2: ang -= math.pi
        elif ang < -math.pi / 2: ang += math.pi
        return ang

    def order_corners(self, pts):

        # Imports
        import numpy as np
        

        pts    = pts[np.argsort(pts[:, 1])]
        top    = pts[:2][np.argsort(pts[:2, 0])]
        bottom = pts[2:][np.argsort(pts[2:, 0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]])

    def top_edge_center(self, corners):
        import numpy as np
        return np.mean(np.asarray(corners, dtype=np.float32)[:2], axis=0)

    def bottom_edge_center(self, corners):
        import numpy as np
        return np.mean(np.asarray(corners, dtype=np.float32)[2:4], axis=0)

    def upper_half_center(self, pts):
        import numpy as np
        top_c = self.top_edge_center(pts)
        mid_c = self.points_center(pts)
        return (top_c + mid_c) / 2.0

    def points_center(self, pts):
        import numpy as np
        return np.mean(np.asarray(pts, dtype=np.float32), axis=0)

    def get_best_port(self, ports, prev_center=None, max_jump=100.0):
        import numpy as np
        import math
        if not ports:
            return None

        candidate = max(ports, key=lambda p: np.max(p["corners"][:, 0]))

        if len(ports) >= 2:
            ref = min(ports, key=lambda p: np.min(p["corners"][:, 0]))

            def area(p):
                w = np.linalg.norm(p["corners"][1] - p["corners"][0])
                h = np.linalg.norm(p["corners"][2] - p["corners"][1])
                return w * h

            if area(candidate) < 0.95 * area(ref):
                return None

        if prev_center is not None:
            c = self.points_center(candidate["corners"])
            if math.sqrt((c[0]-prev_center[0])**2 + (c[1]-prev_center[1])**2) > max_jump:
                return None

        return candidate

    # ================================================================
    # JOINT FACTORIES
    # ================================================================
    def _make_joint_pos(self, positions):
        from aic_control_interfaces.msg import (
            JointMotionUpdate,
            TrajectoryGenerationMode,
        )
        jmu = JointMotionUpdate()
        jmu.target_state.positions          = positions
        jmu.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        jmu.target_stiffness                = self.current_stiffness[:]
        jmu.target_damping                  = self.current_damping[:]
        return jmu

    def _make_joint_vel(self, velocities):
        from aic_control_interfaces.msg import (
    MotionUpdate,
    JointMotionUpdate,
    TrajectoryGenerationMode,
)

        jmu = JointMotionUpdate()
        jmu.target_state.velocities         = velocities
        jmu.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        jmu.target_stiffness                = self.current_stiffness[:]
        jmu.target_damping                  = self.current_damping[:]
        return jmu

    # ================================================================
    # IMAGE HELPERS
    # ================================================================
    def _msg_to_cv2(self, img_msg):
        import numpy as np
        import cv2
        img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
        cv_img    = img_array.reshape((img_msg.height, img_msg.width, 3))
        if img_msg.encoding == 'rgb8':
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        return cv_img

    def _cv2_to_msg(self, cv_img, header):
        from sensor_msgs.msg import Image

        msg              = Image()
        msg.header       = header
        msg.height       = cv_img.shape[0]
        msg.width        = cv_img.shape[1]
        msg.encoding     = "bgr8"
        msg.is_bigendian = False
        msg.step         = cv_img.shape[1] * 3
        msg.data         = cv_img.tobytes()
        return msg

    # ================================================================
    # DEBUG VISUALISER
    # ================================================================
    def _draw_debug_info(self, frame, ports_center_local=None, cable_center_local=None):
        from sensor_msgs.msg import Image

        import numpy as np
        import cv2
        output = frame.copy()
        
        if ports_center_local is None:
            ports_center_local = self.ports_center
        if cable_center_local is None:
            cable_center_local = self.cable_center

        if ports_center_local:
            best = self.current_best_port_center
            if isinstance(best, dict):
                if "corners" in best:
                    t = self.points_center(best["corners"])
                    cv2.circle(output, (int(t[0]), int(t[1])), 6, (0, 255, 255), -1)
                    pts = best["corners"].astype(int)
                    cv2.line(output, tuple(pts[0]), tuple(pts[1]), (255, 255, 0), 2)
                    cv2.line(output, tuple(pts[2]), tuple(pts[3]), (255, 255, 0), 2)
                elif "center" in best:
                    t = best["center"]
                    cv2.circle(output, (int(t[0]), int(t[1])), 6, (0, 255, 255), -1)

        if self.port1_center_pt is not None and self.port2_center_pt is not None:
            cv2.line(output,
                     tuple(self.port1_center_pt.astype(int)),
                     tuple(self.port2_center_pt.astype(int)),
                     (0, 255, 255), 2)

        if cable_center_local is not None and len(cable_center_local) > 0:
            cx, cy = int(cable_center_local[0]), int(cable_center_local[1])
            cv2.circle(output, (cx, cy), 6, (255, 0, 255), -1)
            cv2.putText(output, "CABLE_TOP_EDGE", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return output

    def predict_rail(self, tcp_z, target_rail_index, di_values):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        if not self.rail_coefficients: return []
        
        refs = []
        for coeffs in self.rail_coefficients:
            poly = np.poly1d(coeffs)
            refs.append(poly(tcp_z))
        
        num_obs = len(di_values)
        num_rails = len(refs)
        cost_matrix = np.zeros((num_obs, num_rails))
        for i in range(num_obs):
            for j in range(num_rails):
                cost_matrix[i, j] = (di_values[i] - refs[j])**2
        
        obs_idx, rail_idx = linear_sum_assignment(cost_matrix)
        assigned_rails = list(rail_idx)
        
        if target_rail_index not in assigned_rails:
            min_increase = float('inf')
            best_obs_to_switch = -1
            for i in range(num_obs):
                current_rail = assigned_rails[i]
                increase = cost_matrix[i, target_rail_index] - cost_matrix[i, current_rail]
                if increase < min_increase:
                    min_increase = increase
                    best_obs_to_switch = i
            if best_obs_to_switch != -1:
                assigned_rails[best_obs_to_switch] = target_rail_index
        return assigned_rails
    def flush_stored_energy(self, obs, move_robot):
        from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
        from geometry_msgs.msg import Wrench
        import numpy as np

        # 1. Get current physical pose
        curr_pose = obs.controller_state.tcp_pose

        # 2. Send a POSITION command to the CURRENT pose
        mu = MotionUpdate()
        mu.header.stamp = self.parent_node.get_clock().now().to_msg()
        mu.header.frame_id = "base_link" # Pose is usually in base_link
        mu.pose = curr_pose 
        
        # Use zero velocity
        mu.velocity.linear.x = 0.0
        mu.velocity.linear.y = 0.0
        mu.velocity.linear.z = 0.0
        
        mu.target_stiffness = np.diag(self.current_stiffness).flatten().tolist()
        mu.target_damping = np.diag(self.current_damping).flatten().tolist()
        mu.feedforward_wrench_at_tip = Wrench()
        
        # This is the key: Switch to POSITION mode for one frame to reset the integrator
        mu.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        
        move_robot(motion_update=mu)
        self.parent_node.get_logger().info("[CONTROL] Flushed stored energy/wind-up.")

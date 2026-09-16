# MCAP inspection: rec_20260821_025500_780ebcbd

- file: `/n/holylabs/hankyang_lab/Lab/jackbjed/dexteleop-vla/data/raw/rec_20260821_025500_780ebcbd/rec_20260821_025500_780ebcbd_0.mcap` (776.5 MiB)
- bag start (ns since epoch): 1787252100557995575
- duration: 347.779 s, messages: 46951
- decode pass: 3 s

## Topics (timing from MCAP log_time)

| topic | schema | n | first s | last s | Hz | dt med/p99/max ms | gaps>2x | hdr−log ms (mean) |
|---|---|---|---|---|---|---|---|---|
| `/can0/error_code` | ErrorCode | 0 | | | | | | |
| `/can0/motor_cmd` | JointState | 0 | | | | | | |
| `/can0/motor_states` | JointState | 0 | | | | | | |
| `/can1/error_code` | ErrorCode | 0 | | | | | | |
| `/can1/motor_cmd` | JointState | 0 | | | | | | |
| `/can1/motor_states` | JointState | 0 | | | | | | |
| `/can2/error_code` | ErrorCode | 0 | | | | | | |
| `/can2/motor_cmd` | JointState | 0 | | | | | | |
| `/can2/motor_states` | JointState | 0 | | | | | | |
| `/can3/error_code` | ErrorCode | 0 | | | | | | |
| `/can3/motor_cmd` | JointState | 0 | | | | | | |
| `/can3/motor_states` | JointState | 0 | | | | | | |
| `/can4/error_code` | ErrorCode | 0 | | | | | | |
| `/can4/motor_cmd` | JointState | 0 | | | | | | |
| `/can4/motor_states` | JointState | 0 | | | | | | |
| `/chassis/error_code` | ErrorCode | 0 | | | | | | |
| `/chassis/joint_cmd` | JointState | 0 | | | | | | |
| `/chassis/joint_states` | JointState | 0 | | | | | | |
| `/chassis_target_vel` | Float32MultiArray | 0 | | | | | | |
| `/fsm_state` | Int32 | 0 | | | | | | |
| `/head_position` | Pose | 0 | | | | | | |
| `/kinco/actual_position` | Float64 | 0 | | | | | | |
| `/kinco/actual_velocity` | Float64 | 0 | | | | | | |
| `/kinco/cmd_enable` | Bool | 0 | | | | | | |
| `/kinco/cmd_mode` | String | 0 | | | | | | |
| `/kinco/cmd_velocity` | Float64 | 0 | | | | | | |
| `/kinco/motor_state` | String | 0 | | | | | | |
| `/left/color/image_raw/ffmpeg` | FFMPEGPacket | 15651 | 0.000 | 347.779 | 45.0 | 22.3/25.2/35 | 0 | 5076.8 |
| `/left_arm/current_ee_pose` | Pose | 0 | | | | | | |
| `/left_arm/error_code` | ErrorCode | 0 | | | | | | |
| `/left_arm/joint_cmd` | JointState | 0 | | | | | | |
| `/left_arm/joint_states` | JointState | 0 | | | | | | |
| `/left_arm/target_ee_pose` | Pose | 0 | | | | | | |
| `/left_gripper/error_code` | ErrorCode | 0 | | | | | | |
| `/left_gripper/joint_cmd` | JointState | 0 | | | | | | |
| `/left_gripper/joint_states` | JointState | 0 | | | | | | |
| `/left_hand_position` | Pose | 0 | | | | | | |
| `/left_shoulder_position` | Pose | 0 | | | | | | |
| `/left_target_ee_pose` | Pose | 0 | | | | | | |
| `/left_wrist_position` | Pose | 0 | | | | | | |
| `/right/color/image_raw/ffmpeg` | FFMPEGPacket | 15650 | 0.020 | 347.776 | 45.0 | 22.2/25.2/41 | 0 | 5076.3 |
| `/right_arm/current_ee_pose` | Pose | 0 | | | | | | |
| `/right_arm/error_code` | ErrorCode | 0 | | | | | | |
| `/right_arm/joint_cmd` | JointState | 0 | | | | | | |
| `/right_arm/joint_states` | JointState | 0 | | | | | | |
| `/right_arm/target_ee_pose` | Pose | 0 | | | | | | |
| `/right_gripper/error_code` | ErrorCode | 0 | | | | | | |
| `/right_gripper/joint_cmd` | JointState | 0 | | | | | | |
| `/right_gripper/joint_states` | JointState | 0 | | | | | | |
| `/right_hand_position` | Pose | 0 | | | | | | |
| `/right_shoulder_position` | Pose | 0 | | | | | | |
| `/right_target_ee_pose` | Pose | 0 | | | | | | |
| `/right_wrist_position` | Pose | 0 | | | | | | |
| `/tf` | TFMessage | 0 | | | | | | |
| `/tf_static` | TFMessage | 0 | | | | | | |
| `/xr/hmd_pose` | PoseStamped | 0 | | | | | | |
| `/xr/left_aim_pose` | PoseStamped | 0 | | | | | | |
| `/xr/left_hand_inputs` | Joy | 0 | | | | | | |
| `/xr/right_aim_pose` | PoseStamped | 0 | | | | | | |
| `/xr/right_hand_inputs` | Joy | 0 | | | | | | |
| `/xr_video_topic/ffmpeg` | FFMPEGPacket | 15650 | 0.010 | 347.767 | 45.0 | 22.3/26.2/31 | 0 | 5065.1 |

## JointState topics: names + per-field stats

## Video topics

- `/left/color/image_raw/ffmpeg`: {"encoding": ["hevc"], "width": [2560], "height": [800], "frame_id": [""], "n_packets": 15651, "n_keyframes": 0, "flags_unique": [0], "pts_first": 7672, "pts_last": 347786500, "pts_dt_unique": [9139, 11041, 11479, 13604, 13975, 15611, 18454, 18508, 18525, 18561, 18597, 18610, 18628, 18629, 18707, 18709, 18726, 18738, 18750, 18759], "pts_nonmonotonic": 0, "bytes_mean": 18820.3729474155, "bytes_max": 109894, "keyframe_interval_packets": null}
- `/right/color/image_raw/ffmpeg`: {"encoding": ["hevc"], "width": [2560], "height": [800], "frame_id": [""], "n_packets": 15650, "n_keyframes": 0, "flags_unique": [0], "pts_first": 27318, "pts_last": 347784044, "pts_dt_unique": [3334, 13509, 13721, 16033, 16536, 18538, 18558, 18565, 18608, 18610, 18628, 18664, 18672, 18675, 18714, 18734, 18741, 18742, 18748, 18767], "pts_nonmonotonic": 0, "bytes_mean": 10636.668051118211, "bytes_max": 91847, "keyframe_interval_packets": null}
- `/xr_video_topic/ffmpeg`: {"encoding": ["hevc"], "width": [3840], "height": [1920], "frame_id": [""], "n_packets": 15650, "n_keyframes": 0, "flags_unique": [0], "pts_first": 17651, "pts_last": 347774388, "pts_dt_unique": [14017, 15051, 15105, 15590, 15712, 16445, 17127, 17131, 17161, 17163, 17199, 17209, 17216, 17229, 17233, 17235, 17250, 17252, 17257, 17267], "pts_nonmonotonic": 0, "bytes_mean": 22222.856869009585, "bytes_max": 230447, "keyframe_interval_packets": null}

## Other decoded topics

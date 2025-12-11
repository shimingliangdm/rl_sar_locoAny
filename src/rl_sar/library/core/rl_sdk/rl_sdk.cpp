/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"
#include <random>
#include <algorithm>


void RL::StateController(const RobotState<float>* state, RobotCommand<float>* command)
{
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    fsm.Run();

    this->motiontime++;

    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
    }
}

std::vector<float> RL::UniformNoise(const std::vector<float>& x, float min, float max)
{
    std::vector<float> result = x;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (auto& val : result) 
    {
        val += dis(gen);
    }

    return result;
}

std::vector<float> RL::ApplyNoise(const std::vector<float>& x, float min, float max)
{
    return UniformNoise(x, min, max);
}

std::vector<float> RL::ComputeObservation()
{
    std::vector<std::vector<float>> obs_list;

    for (const std::string &observation : this->params.Get<std::vector<std::string>>("observations"))
    {
        // ============= Base Observations =============
        if (observation == "lin_vel")
        {
            obs_list.push_back(this->obs.lin_vel * this->params.Get<float>("lin_vel_scale"));
        }
        else if (observation == "ang_vel")
        {
            // In ROS1 Gazebo, the coordinate system for angular velocity is in the world coordinate system.
            // In ROS2 Gazebo, mujoco and real robot, the coordinate system for angular velocity is in the body coordinate system.
            if (this->ang_vel_axis == "body")
            {
                obs_list.push_back(this->obs.ang_vel * this->params.Get<float>("ang_vel_scale"));
            }
            else if (this->ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.Get<float>("ang_vel_scale"));
            }
        }
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec));
        }
        else if (observation == "commands")
        {
            obs_list.push_back(this->obs.commands * this->params.Get<std::vector<float>>("commands_scale"));
        }
        else if (observation == "dof_pos")
        {
            std::vector<float> dof_pos_rel = this->obs.dof_pos - this->params.Get<std::vector<float>>("default_dof_pos");
            for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
            {
                dof_pos_rel[i] = 0.0f;
            }
            obs_list.push_back(dof_pos_rel * this->params.Get<float>("dof_pos_scale"));
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.Get<float>("dof_vel_scale"));
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        // ============= Other Observations =============
        else if (observation == "whole_body_tracking/motion_command")
        {
            std::vector<float> motion_cmd;
            if (this->motion_loader)
            {
                auto joint_pos_sdk = this->motion_loader->GetJointPos();
                auto joint_vel_sdk = this->motion_loader->GetJointVel();
                auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
                std::vector<float> joint_pos_training(joint_mapping.size());
                std::vector<float> joint_vel_training(joint_mapping.size());
                for (size_t i = 0; i < joint_mapping.size(); ++i)
                {
                    joint_pos_training[i] = joint_pos_sdk[joint_mapping[i]];
                    joint_vel_training[i] = joint_vel_sdk[joint_mapping[i]];
                }
                motion_cmd.insert(motion_cmd.end(), joint_pos_training.begin(), joint_pos_training.end());
                motion_cmd.insert(motion_cmd.end(), joint_vel_training.begin(), joint_vel_training.end());
            }
            else
            {
                motion_cmd.resize(this->params.Get<int>("num_of_dofs") * 2, 0.0f);
            }
            obs_list.push_back(motion_cmd);
        }
        else if (observation == "whole_body_tracking/motion_anchor_ori_b")
        {
            std::vector<float> anchor_ori(6, 0.0f);
            if (this->motion_loader)
            {
                auto waist_sdk_indices = this->params.Get<std::vector<int>>("waist_joint_indices");
                std::vector<float> waist_angles = {
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[0])],
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[1])],
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[2])]
                };
                std::vector<float> robot_torso_quat_w = MotionLoader::ComputeTorsoQuat(this->obs.base_quat, waist_angles);
                std::vector<float> ref_torso_quat_w = this->motion_loader->GetAnchorQuat();
                std::vector<float> init_quat = this->motion_loader->GetInitQuat();
                std::vector<float> motion_anchor_quat_w = QuaternionMultiply(init_quat, ref_torso_quat_w);
                std::vector<float> robot_quat_inv = QuaternionConjugate(robot_torso_quat_w);
                std::vector<float> relative_quat = QuaternionMultiply(robot_quat_inv, motion_anchor_quat_w);
                std::vector<float> rot_matrix = QuaternionToRotationMatrix(relative_quat);
                anchor_ori = MatrixFirstTwoColumns(rot_matrix);
            }
            obs_list.push_back(anchor_ori);
        }
        else if (observation == "RoboMimic_Deploy/phase")
        {
            float motion_time = this->episode_length_buf * this->params.Get<float>("dt") * this->params.Get<int>("decimation");
            float count = motion_time;
            float phase = count / this->motion_length;
            std::vector<float> phase_vec = {phase};
            obs_list.push_back(phase_vec);
        }
        // ============= lafan Observations =============
        if (observation == "lafan_motion_command")
        {
            std::vector<float> motion_cmd;
            if (this->video_mimic_motion_loader)
            {
                auto joint_pos_sdk = this->video_mimic_motion_loader->GetJointPos();
                auto joint_vel_sdk = this->video_mimic_motion_loader->GetJointVel();
                auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
                std::vector<float> joint_pos_training(joint_mapping.size());
                std::vector<float> joint_vel_training(joint_mapping.size());
                for (size_t i = 0; i < joint_mapping.size(); ++i)
                {
                    //joint_pos_training[i] = joint_pos_sdk[joint_mapping[i]];
                    //joint_vel_training[i] = joint_vel_sdk[joint_mapping[i]];
                    joint_pos_training[i] = joint_pos_sdk[i];
                    joint_vel_training[i] = joint_vel_sdk[i];
                }
                motion_cmd.insert(motion_cmd.end(), joint_pos_training.begin(), joint_pos_training.end());
                motion_cmd.insert(motion_cmd.end(), joint_vel_training.begin(), joint_vel_training.end());

                if (cur_joint_pos_action.size() == 0)
                {
                    // which means it's first frame
                    cur_joint_pos_action.insert(cur_joint_pos_action.end(), joint_pos_training.begin(), joint_pos_training.end());
                    cur_joint_vel_action.insert(cur_joint_vel_action.end(), joint_vel_training.begin(), joint_vel_training.end());

                    last_joint_pos_action.assign(29, 0.0);
                    last_joint_vel_action.assign(29, 0.0);
                }
                else
                {
                    last_joint_pos_action.clear();
                    last_joint_pos_action.insert(last_joint_pos_action.end(), cur_joint_pos_action.begin(), cur_joint_pos_action.end());
                    
                    last_joint_vel_action.clear();
                    last_joint_vel_action.insert(last_joint_vel_action.end(), cur_joint_vel_action.begin(), cur_joint_vel_action.end());

                    cur_joint_pos_action.clear();
                    cur_joint_vel_action.clear();
                    cur_joint_pos_action.insert(cur_joint_pos_action.end(), joint_pos_training.begin(), joint_pos_training.end());
                    cur_joint_vel_action.insert(cur_joint_vel_action.end(), joint_vel_training.begin(), joint_vel_training.end());
                }
            }
            else
            {
                motion_cmd.resize(this->params.Get<int>("num_of_dofs") * 2, 0.0f);
            }
            std::vector<float> ref_root_pos_10_30_60_w = video_mimic_motion_loader->GetFutureRootPos();
            Eigen::Vector3d p_ref_future_10(ref_root_pos_10_30_60_w[0], ref_root_pos_10_30_60_w[1], ref_root_pos_10_30_60_w[2]);
            Eigen::Vector3d p_ref_future_30(ref_root_pos_10_30_60_w[3], ref_root_pos_10_30_60_w[4], ref_root_pos_10_30_60_w[5]);
            Eigen::Vector3d p_ref_future_60(ref_root_pos_10_30_60_w[6], ref_root_pos_10_30_60_w[7], ref_root_pos_10_30_60_w[8]);
            std::vector<float> real_root_pos_w = root_world_joint_translation;
            // wxyz
            std::vector<float> real_root_quat_w = root_world_joint_quat;
            Eigen::Vector3d p_real(real_root_pos_w[0], real_root_pos_w[1], real_root_pos_w[2]);
            Eigen::Quaterniond q_real(real_root_quat_w[0], real_root_quat_w[1], real_root_quat_w[2], real_root_quat_w[3]);
            Eigen::Vector3d pos_in_robot_frame_future_10 = q_real.inverse() * (p_ref_future_10 - p_real);
            Eigen::Vector3d pos_in_robot_frame_future_30 = q_real.inverse() * (p_ref_future_30 - p_real);
            Eigen::Vector3d pos_in_robot_frame_future_60 = q_real.inverse() * (p_ref_future_60 - p_real);
            std::vector<float> future_ref_10 = 
            {
                pos_in_robot_frame_future_10.x(), 
                pos_in_robot_frame_future_10.y(), 
                pos_in_robot_frame_future_10.z()
            };
            std::vector<float> future_ref_30 = 
            {
                pos_in_robot_frame_future_30.x(), 
                pos_in_robot_frame_future_30.y(), 
                pos_in_robot_frame_future_30.z()
            };
            std::vector<float> future_ref_60 = 
            {
                pos_in_robot_frame_future_60.x(), 
                pos_in_robot_frame_future_60.y(), 
                pos_in_robot_frame_future_60.z()
            };
            motion_cmd.insert(motion_cmd.end(), future_ref_10.begin(), future_ref_10.end());
            motion_cmd.insert(motion_cmd.end(), future_ref_30.begin(), future_ref_30.end());
            motion_cmd.insert(motion_cmd.end(), future_ref_60.begin(), future_ref_60.end());

            obs_list.push_back(motion_cmd);
        }
        else if (observation == "lafan_motion_anchor_pos_b")
        {
            std::vector<float> root_pos_in_robot_frame_vec(3, 0.0);
            if (this->video_mimic_motion_loader)
            {
                std::vector<float> ref_root_pos_w = video_mimic_motion_loader->GetRootPos();
                std::vector<float> ref_root_quat_w = video_mimic_motion_loader->GetRootQuat();

                std::vector<float> real_root_pos_w = root_world_joint_translation;
                // wxyz
                std::vector<float> real_root_quat_w = root_world_joint_quat;

                Eigen::Vector3d p_ref(ref_root_pos_w[0], ref_root_pos_w[1], ref_root_pos_w[2]);
                Eigen::Quaterniond q_ref(ref_root_quat_w[0], ref_root_quat_w[1], ref_root_quat_w[2], ref_root_quat_w[3]); // w, x, y, z
                
                Eigen::Vector3d p_real(real_root_pos_w[0], real_root_pos_w[1], real_root_pos_w[2]);
                Eigen::Quaterniond q_real(real_root_quat_w[0], real_root_quat_w[1], real_root_quat_w[2], real_root_quat_w[3]); // w, x, y, z

                q_ref.normalize();
                q_real.normalize();

                Eigen::Vector3d pos_in_robot_frame = q_real.inverse() * (p_ref - p_real);

                root_pos_in_robot_frame_vec[0] = pos_in_robot_frame.x();
                root_pos_in_robot_frame_vec[1] = pos_in_robot_frame.y();
                root_pos_in_robot_frame_vec[2] = pos_in_robot_frame.z();
            }
            obs_list.push_back(root_pos_in_robot_frame_vec);
        }
        else if (observation == "lafan_motion_anchor_ori_b")
        {
            std::vector<float> root_quat_in_robot_frame_vec;
            if (this->video_mimic_motion_loader)
            {
                std::vector<float> ref_root_pos_w = video_mimic_motion_loader->GetRootPos();
                std::vector<float> ref_root_quat_w = video_mimic_motion_loader->GetRootQuat();

                std::vector<float> real_root_pos_w = root_world_joint_translation;
                // wxyz
                std::vector<float> real_root_quat_w = root_world_joint_quat;

                Eigen::Vector3d p_ref(ref_root_pos_w[0], ref_root_pos_w[1], ref_root_pos_w[2]);
                Eigen::Quaterniond q_ref(ref_root_quat_w[0], ref_root_quat_w[1], ref_root_quat_w[2], ref_root_quat_w[3]); // w, x, y, z
                
                Eigen::Vector3d p_real(real_root_pos_w[0], real_root_pos_w[1], real_root_pos_w[2]);
                Eigen::Quaterniond q_real(real_root_quat_w[0], real_root_quat_w[1], real_root_quat_w[2], real_root_quat_w[3]); // w, x, y, z

                q_ref.normalize();
                q_real.normalize();

                Eigen::Quaterniond quat_in_robot_frame = q_real.inverse() * q_ref;
                
                // we take first 2 columns of the rotation matrix
                Eigen::Matrix3d mat = quat_in_robot_frame.toRotationMatrix();
                for (int i=0; i<3; i++)
                {
                    root_quat_in_robot_frame_vec.push_back(mat(i, 0));
                    root_quat_in_robot_frame_vec.push_back(mat(i, 1));
                }
            }
            else
            {
                root_quat_in_robot_frame_vec = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            }
            obs_list.push_back(root_quat_in_robot_frame_vec);
        }
        else if (observation == "lafan_joint_pos_history")
        {
            // information from locoAny printing that
            // length of JointPosHistory is 167 !!!

            std::vector<float> joint_pos_history;
            std::vector<float> noise_lin_vel_b = ApplyNoise(root_local_joint_lin_vel, noise_lin_vel_b_min, noise_lin_vel_b_max);
            joint_pos_history.insert(joint_pos_history.end(), noise_lin_vel_b.begin(), noise_lin_vel_b.end());

            std::vector<float> noise_ang_vel_b = ApplyNoise(root_local_joint_ang_vel, noise_ang_vel_b_min, noise_ang_vel_b_max);
            joint_pos_history.insert(joint_pos_history.end(), noise_ang_vel_b.begin(), noise_ang_vel_b.end());
            // this is how default joint pos looks like
            /******
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.76),
                joint_pos={
                    ".*_hip_pitch_joint": -0.312,
                    ".*_knee_joint": 0.669,
                    ".*_ankle_pitch_joint": -0.363,
                    ".*_elbow_joint": 0.6,
                    "left_shoulder_roll_joint": 0.2,
                    "left_shoulder_pitch_joint": 0.2,
                    "right_shoulder_roll_joint": -0.2,
                    "right_shoulder_pitch_joint": 0.2,
                },
                joint_vel={".*": 0.0},
            )
            */
            std::vector<float> joint_pos_rels;
            for (int i=0; i<29; i++)
            {
                float joint_pos_rel = cur_joint_pos[i] - default_joint_pos[i];
                joint_pos_rels.push_back(joint_pos_rel);
            }
            std::vector<float> noise_joint_pos_rel = ApplyNoise(joint_pos_rels, noise_joint_pos_min, noise_joint_pos_max);
            joint_pos_history.insert(joint_pos_history.end(), noise_joint_pos_rel.begin(), noise_joint_pos_rel.end());

            std::vector<float> joint_vel_rels;
            for (int i=0; i<29; i++)
            {
                // joint vel is assigned becuase default joint vel is 0
                float joint_vel_rel = cur_joint_vel[i];
                joint_vel_rels.push_back(joint_vel_rel);
            }
            std::vector<float> noise_joint_vel_rel = ApplyNoise(joint_vel_rels, noise_joint_vel_min, noise_joint_vel_max);
            joint_pos_history.insert(joint_pos_history.end(), noise_joint_vel_rel.begin(), noise_joint_vel_rel.end());

            std::vector<float> imu = cur_imu;
            joint_pos_history.insert(joint_pos_history.end(), imu.begin(), imu.end());

            std::vector<float> last_action = last_joint_pos_action;
            joint_pos_history.insert(joint_pos_history.end(), last_action.begin(), last_action.end());

            std::vector<float> priv_explicit(9, 0.0);
            priv_explicit[0] = root_local_joint_lin_vel[0] * 2.0;
            priv_explicit[1] = root_local_joint_lin_vel[1] * 2.0;
            priv_explicit[2] = root_local_joint_lin_vel[2] * 2.0;
            joint_pos_history.insert(joint_pos_history.end(), priv_explicit.begin(), priv_explicit.end());

            /* mass information is fetched from mujoco model
            === Unitree G1 Body Masses ===
            ID: 0 | Name: world | Mass: 0 kg
            ID: 1 | Name: pelvis | Mass: 3.813 kg
            ID: 2 | Name: left_hip_pitch_link | Mass: 1.35 kg
            ID: 3 | Name: left_hip_roll_link | Mass: 1.52 kg
            ID: 4 | Name: left_hip_yaw_link | Mass: 1.702 kg
            ID: 5 | Name: left_knee_link | Mass: 1.932 kg
            ID: 6 | Name: left_ankle_pitch_link | Mass: 0.074 kg
            ID: 7 | Name: left_ankle_roll_link | Mass: 0.608 kg
            ID: 8 | Name: right_hip_pitch_link | Mass: 1.35 kg
            ID: 9 | Name: right_hip_roll_link | Mass: 1.52 kg
            ID: 10 | Name: right_hip_yaw_link | Mass: 1.702 kg
            ID: 11 | Name: right_knee_link | Mass: 1.932 kg
            ID: 12 | Name: right_ankle_pitch_link | Mass: 0.074 kg
            ID: 13 | Name: right_ankle_roll_link | Mass: 0.608 kg
            ID: 14 | Name: waist_yaw_link | Mass: 0.244 kg
            ID: 15 | Name: waist_roll_link | Mass: 0.047 kg
            ID: 16 | Name: torso_link | Mass: 9.598 kg
            ID: 17 | Name: left_shoulder_pitch_link | Mass: 0.718 kg
            ID: 18 | Name: left_shoulder_roll_link | Mass: 0.643 kg
            ID: 19 | Name: left_shoulder_yaw_link | Mass: 0.734 kg
            ID: 20 | Name: left_elbow_link | Mass: 0.6 kg
            ID: 21 | Name: left_wrist_roll_link | Mass: 0.085445 kg
            ID: 22 | Name: left_wrist_pitch_link | Mass: 0.48405 kg
            ID: 23 | Name: left_wrist_yaw_link | Mass: 0.254576 kg
            ID: 24 | Name: right_shoulder_pitch_link | Mass: 0.718 kg
            ID: 25 | Name: right_shoulder_roll_link | Mass: 0.643 kg
            ID: 26 | Name: right_shoulder_yaw_link | Mass: 0.734 kg
            ID: 27 | Name: right_elbow_link | Mass: 0.6 kg
            ID: 28 | Name: right_wrist_roll_link | Mass: 0.085445 kg
            ID: 29 | Name: right_wrist_pitch_link | Mass: 0.48405 kg
            ID: 30 | Name: right_wrist_yaw_link | Mass: 0.254576 kg
            ==============================
            */

            /* default joint stiffness of each joint
            [40.1792, 40.1792, 40.1792, 99.0984, 99.0984, 28.5012, 40.1792, 40.1792,
                28.5012, 99.0984, 99.0984, 14.2506, 14.2506, 28.5012, 28.5012, 14.2506,
                14.2506, 28.5012, 28.5012, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506,
                14.2506, 16.7783, 16.7783, 16.7783, 16.7783]
            */

            /* default joint damping of each joint
            [2.5579, 2.5579, 2.5579, 6.3088, 6.3088, 1.8144, 2.5579, 2.5579, 1.8144,
                6.3088, 6.3088, 0.9072, 0.9072, 1.8144, 1.8144, 0.9072, 0.9072, 1.8144,
                1.8144, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,
                1.0681, 1.0681]
            */

            /* friction in simulation is
            [0.4369, 1.1136]
            */


            // this is the center of mass in python
            //body_com = self.asset.data.com_pos_b[:,self.body_id,:].to(self.device).squeeze(1)
            // actually it's the position of root joint
            std::vector<float> mass_params(4, 0.0);
            mass_params[0] = 3.813;
            mass_params[1] = root_local_joint_translation[0];
            mass_params[2] = root_local_joint_translation[1];
            mass_params[3] = root_local_joint_translation[2];
            joint_pos_history.insert(joint_pos_history.end(), mass_params.begin(), mass_params.end());

            std::vector<float> friction_coeffs = {0.4369};
            joint_pos_history.insert(joint_pos_history.end(), friction_coeffs.begin(), friction_coeffs.end());

            std::vector<float> kp_ratio(29, 1.0);
            std::vector<float> kd_ratio(29, 1.0);
            joint_pos_history.insert(joint_pos_history.end(), kp_ratio.begin(), kp_ratio.end());
            joint_pos_history.insert(joint_pos_history.end(), kd_ratio.begin(), kd_ratio.end());

            obs_list.push_back(joint_pos_history);
        }
    }

    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
        this->obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    std::vector<float> clamped_obs = clamp(obs, -this->params.Get<float>("clip_obs"), this->params.Get<float>("clip_obs"));
    return clamped_obs;
}

void RL::InitObservations()
{
    this->obs.lin_vel = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};
    this->obs.commands = {0.0f, 0.0f, 0.0f};
    this->obs.base_quat = {0.0f, 0.0f, 0.0f, 1.0f};
    this->obs.dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.actions.clear();
    this->obs.actions.resize(this->params.Get<int>("num_of_dofs"), 0.0f);

    // for lafan motion mimic
    this->obs.lafan_motion_command.assign(67, 0.0);
    this->obs.lafan_motion_anchor_pos_b.assign(3, 0.0);
    this->obs.lafan_motion_anchor_ori_b.assign(6, 0.0);
    this->obs.lafan_joint_pos_history.assign(167, 0.0);

    this->ComputeObservation();
}

void RL::InitOutputs()
{
    int num_of_dofs = this->params.Get<int>("num_of_dofs");
    this->output_dof_tau.clear();
    this->output_dof_tau.resize(num_of_dofs, 0.0f);
    this->output_dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->output_dof_vel.clear();
    this->output_dof_vel.resize(num_of_dofs, 0.0f);
}

void RL::InitControl()
{
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}

void RL::InitJointNum(size_t num_joints)
{
    this->robot_state.motor_state.resize(num_joints);
    this->start_state.motor_state.resize(num_joints);
    this->now_state.motor_state.resize(num_joints);
    this->robot_command.motor_command.resize(num_joints);
}

void RL::InitRL(std::string robot_config_path)
{
    std::lock_guard<std::mutex> lock(this->model_mutex);

    this->ReadYaml(robot_config_path, "config.yaml");

    // init joint num first
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));

    if (!pinocchio_initialized)
    {
        std::string urdf_path = "/home/dm/rl_sar_locoAny/src/rl_sar/library/core/rl_sdk/g1_29dof.urdf";
        pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_pin);
        data_pin = pinocchio::Data(model_pin);

        root_world_joint_translation = {0.0, 0.0, 0.0};
        root_world_joint_quat = {0.0, 0.0, 0.0, 0.0};
        root_world_joint_lin_vel = {0.0, 0.0, 0.0};
        root_world_joint_ang_vel = {0.0, 0.0, 0.0};

        root_local_joint_translation = {0.0, 0.0, 0.0};
        root_local_joint_quat = {0.0, 0.0, 0.0, 0.0};
        root_local_joint_lin_vel = {0.0, 0.0, 0.0};
        root_local_joint_ang_vel = {0.0, 0.0, 0.0};

        cur_joint_pos.assign(29, 0.0);
        /*
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        }
        */

        /* joint names
        [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint", 
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
        ]
        */
        default_joint_pos.assign(29, 0.0);
        default_joint_pos[0] = -0.312;
        default_joint_pos[6] = -0.312;
        default_joint_pos[3] = 0.669;
        default_joint_pos[9] = 0.669;
        default_joint_pos[4] = -0.363;
        default_joint_pos[10] = -0.363;
        default_joint_pos[18] = 0.6;
        default_joint_pos[25] = 0.6;
        default_joint_pos[16] = 0.2;
        default_joint_pos[15] = 0.2;
        default_joint_pos[23] = -0.2;
        default_joint_pos[22] = 0.2;

        cur_joint_vel.assign(29, 0.0);

        default_kp.assign(29, 0.0);
        default_kd.assign(29, 0.0);
        
        default_kp[0] = 40.1792;
        default_kp[1] = 40.1792;
        default_kp[2] = 40.1792;
        default_kp[3] = 99.0984;
        default_kp[4] = 99.0984;
        default_kp[5] = 28.5012;
        default_kp[6] = 40.1792;
        default_kp[7] = 40.1792;
        default_kp[8] = 28.5012;
        default_kp[9] = 99.0984;
        default_kp[10] = 99.0984;
        default_kp[11] = 14.2506;
        default_kp[12] = 14.2506;
        default_kp[13] = 28.5012;
        default_kp[14] = 28.5012;
        default_kp[15] = 14.2506;
        default_kp[16] = 14.2506;
        default_kp[17] = 28.5012;
        default_kp[18] = 28.5012;
        default_kp[19] = 14.2506;
        default_kp[20] = 14.2506;
        default_kp[21] = 14.2506;
        default_kp[22] = 14.2506;
        default_kp[23] = 14.2506;
        default_kp[24] = 14.2506;
        default_kp[25] = 16.7783;
        default_kp[26] = 16.7783;
        default_kp[27] = 16.7783;
        default_kp[28] = 16.7783;

        default_kd[0] = 2.5579;
        default_kd[1] = 2.5579;
        default_kd[2] = 2.5579;
        default_kd[3] = 6.3088;
        default_kd[4] = 6.3088;
        default_kd[5] = 1.8144;
        default_kd[6] = 2.5579;
        default_kd[7] = 2.5579;
        default_kd[8] = 1.8144;
        default_kd[9] = 6.3088;
        default_kd[10] = 6.3088;
        default_kd[11] = 0.9072;
        default_kd[12] = 0.9072;
        default_kd[13] = 1.8144;
        default_kd[14] = 1.8144;
        default_kd[15] = 0.9072;
        default_kd[16] = 0.9072;
        default_kd[17] = 1.8144;
        default_kd[18] = 1.8144;
        default_kd[19] = 0.9072;
        default_kd[20] = 0.9072;
        default_kd[21] = 0.9072;
        default_kd[22] = 0.9072;
        default_kd[23] = 0.9072;
        default_kd[24] = 0.9072;
        default_kd[25] = 1.0681;
        default_kd[26] = 1.0681;
        default_kd[27] = 1.0681;
        default_kd[28] = 1.0681;

        cur_kp.assign(29, 0.0);
        cur_kd.assign(29, 0.0);

        pinocchio_initialized = true;
    }
    
    // init rl
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();

    // init obs history
    const auto& observations_history = this->params.Get<std::vector<int>>("observations_history");  // avoid dangling reference
    if (!observations_history.empty())
    {
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->params.Get<std::string>("observations_history_priority"));
    }

    // init model
    std::string model_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + this->params.Get<std::string>("model_name");
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
}

void RL::ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau)
{
    std::vector<float> actions_scaled = actions * this->params.Get<std::vector<float>>("action_scale");
    std::vector<float> pos_actions_scaled = actions_scaled;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
    {
        pos_actions_scaled[i] = 0.0f;
        vel_actions_scaled[i] = actions_scaled[i];
    }
    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output_dof_pos = pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos");
    output_dof_vel = vel_actions_scaled;
    output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (all_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos") - this->obs.dof_pos) - this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel;
    output_dof_tau = clamp(output_dof_tau, -this->params.Get<std::vector<float>>("torque_limits"), this->params.Get<std::vector<float>>("torque_limits"));
}

int RL::InverseJointMapping(int idx) const
{
    auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    for (size_t i = 0; i < joint_mapping.size(); ++i) {
        if (joint_mapping[i] == idx) return (int)i;
    }
    return -1;
}

void RL::TorqueProtect(const std::vector<float>& origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<float> out_of_range_values;
    for (size_t i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        float torque_value = origin_output_dof_tau[i];
        float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[i];
        float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (size_t i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            float value = out_of_range_values[i];
            float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[index];
            float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[index];

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

void RL::AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold)
{
    // Use QuaternionToEuler from vector_math.hpp
    std::vector<float> euler = QuaternionToEuler(quaternion);
    float roll = euler[0] * 57.2958f;   // Convert to degrees
    float pitch = euler[1] * 57.2958f;

    if (std::fabs(roll) > roll_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

static int kbhit()
{
    static bool initialized = false;
    static termios original_term;

    // Initialize terminal to non-canonical mode on first call
    if (!initialized)
    {
        tcgetattr(STDIN_FILENO, &original_term);

        termios new_term = original_term;
        new_term.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
        new_term.c_cc[VMIN] = 0;   // Non-blocking read
        new_term.c_cc[VTIME] = 0;  // No timeout

        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

        // Register cleanup function to restore terminal on exit
        static bool cleanup_registered = false;
        if (!cleanup_registered)
        {
            std::atexit([]() {
                tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
            });
            cleanup_registered = true;
        }

        initialized = true;
    }

    // Non-blocking read of a single character
    char c;
    int result = read(STDIN_FILENO, &c, 1);

    return (result == 1) ? (unsigned char)c : -1;
}

void RL::KeyboardInterface()
{
    int c = kbhit();
    if (c > 0)
    {
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27:  // Escape sequence (for arrow keys on Unix/Linux/macOS)
        {
            char seq[2];
            // Try to read escape sequence non-blockingly
            if (read(STDIN_FILENO, &seq[0], 1) == 1)
            {
                if (seq[0] == '[')
                {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1)
                    {
                        switch (seq[1])
                        {
                        case 'A': this->control.SetKeyboard(Input::Keyboard::Up); break;
                        case 'B': this->control.SetKeyboard(Input::Keyboard::Down); break;
                        case 'C': this->control.SetKeyboard(Input::Keyboard::Right); break;
                        case 'D': this->control.SetKeyboard(Input::Keyboard::Left); break;
                        default: break;
                        }
                    }
                }
                else
                {
                    // Plain escape key
                    this->control.SetKeyboard(Input::Keyboard::Escape);
                }
            }
            else
            {
                // Plain escape key
                this->control.SetKeyboard(Input::Keyboard::Escape);
            }
        } break;
        default:  break;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ReadYaml(const std::string& file_path, const std::string& file_name)
{
    std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[file_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    for (auto it = config.begin(); it != config.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        this->params.config_node[key] = it->second;
    }
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_cal_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_target_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}

void RL::CSVLogger(const std::vector<float>& torque, const std::vector<float>& tau_est, const std::vector<float>& joint_pos, const std::vector<float>& joint_pos_target, const std::vector<float>& joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos_target[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}

bool RLFSMState::Interpolate(
    float& percent,
    const std::vector<float>& start_pos,
    const std::vector<float>& target_pos,
    float duration_seconds,
    const std::string& description,
    bool use_fixed_gains)
{
    if (percent >= 1.0f)
    {
        return false;
    }

    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / rl.params.Get<float>("dt"))));
    float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    auto kp = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kp") : rl.params.Get<std::vector<float>>("rl_kp");
    auto kd = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kd") : rl.params.Get<std::vector<float>>("rl_kd");

    for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
    {
        fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        fsm_command->motor_command.dq[i] = 0;
        fsm_command->motor_command.kp[i] = kp[i];
        fsm_command->motor_command.kd[i] = kd[i];
        fsm_command->motor_command.tau[i] = 0;
    }

    if (!description.empty())
    {
        LOGGER::PrintProgress(percent, description);
    }

    if (percent >= 1.0f)
    {
        return false;
    }

    return true;
}

void RLFSMState::RLControl()
{
    std::vector<float> _output_dof_pos, _output_dof_vel;
    if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            if (!_output_dof_pos.empty())
            {
                fsm_command->motor_command.q[i] = _output_dof_pos[i];
            }
            if (!_output_dof_vel.empty())
            {
                fsm_command->motor_command.dq[i] = _output_dof_vel[i];
            }
            fsm_command->motor_command.kp[i] = rl.params.Get<std::vector<float>>("rl_kp")[i];
            fsm_command->motor_command.kd[i] = rl.params.Get<std::vector<float>>("rl_kd")[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }
}

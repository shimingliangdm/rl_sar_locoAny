#include "video_mimic_motion_loader.hpp"
#include "cnpy.h"
#include "math_struct.hpp"

VideoMimicMotionLoader::VideoMimicMotionLoader(const std::string& motion_file, float fps)
    : dt_(1.0f / fps), index_0_(0), index_1_(0), blend_(0.0f), index_10_future(0), index_30_future(0), index_60_future(0)
{
    dt_ = 1.0f / fps;
    index_0_ = 0;
    index_1_ = 0;
    blend_ = 0.0;
    index_10_future = 0;
    index_30_future = 0;
    index_60_future = 0;
    
    LoadVideoMimicCSV(motion_file);

    num_frames_ = root_positions_.size();
    duration_ = num_frames_ * dt_;
}

void VideoMimicMotionLoader::Init(const std::string& motion_file)
{
    dt_ = 1.0f / fps;
    index_0_ = 0;
    index_1_ = 0;
    blend_ = 0.0;
    index_10_future = 0;
    index_30_future = 0;
    index_60_future = 0;

    LoadVideoMimicCSV(motion_file);

    num_frames_ = root_positions_.size();
    duration_ = num_frames_ * dt_;
}

void VideoMimicMotionLoader::LoadVideoMimicCSV(const std::string& filename)
{
    
    cnpy::npz_t npz_data = cnpy::npz_load(filename);
    cnpy::NpyArray joint_poses_array = npz_data["joint_pos"];
    cnpy::NpyArray joint_vels_array = npz_data["joint_vel"];
    cnpy::NpyArray body_pos_w_array = npz_data["body_pos_w"];
    cnpy::NpyArray body_quat_w_array = npz_data["body_quat_w"];
    cnpy::NpyArray body_lin_vel_w_array = npz_data["body_lin_vel_w"];
    cnpy::NpyArray body_ang_vel_w_array = npz_data["body_ang_vel_w"];

    float* joint_poses = joint_poses_array.data<float>();
    float* joint_vels = joint_vels_array.data<float>();

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

    float* body_pos_w = body_pos_w_array.data<float>();
    float* body_quat_w = body_quat_w_array.data<float>();

    size_t frame_nb = joint_poses_array.shape[0];
    size_t joint_nb = joint_poses_array.shape[1];

    size_t body_nb = body_pos_w_array.shape[1];

    // target body names
    // ['pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 
    // 'right_knee_link', 'right_ankle_roll_link', 'torso_link', 'left_shoulder_roll_link', 'left_elbow_link', 
    // 'left_wrist_yaw_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link']

    // corresponding body idx
    // [ 0,  4, 10, 18,  5, 11, 19,  9, 16, 22, 28, 17, 23, 29]

    // anchor body name: torso_link

    // anchor body idx: 9

    std::vector<int> target_body_idx = {0,  4, 10, 18,  5, 11, 19,  9, 16, 22, 28, 17, 23, 29};
    int anchor_body_idx = 9;
    for (int i=0; i<frame_nb; i++)
    {
        std::vector<float> frame_joint_pos;
        std::vector<float> frame_joint_vel;
        for (int j=0; j<joint_nb; j++)
        {
            float joint_pos = joint_poses[i*joint_nb + j];
            float joint_vel = joint_vels[i*joint_nb + j];
            frame_joint_pos.push_back(joint_pos);
            frame_joint_vel.push_back(joint_vel);
        }

        std::vector<math_struct::Position<float>> frame_body_positions;
        std::vector<math_struct::Quat<float>> frame_body_quats;
        for (int j=0; j<body_nb; j++)
        {
            auto found = find(target_body_idx.begin(), target_body_idx.end(), j);
            if (found != target_body_idx.end())
            {
                // which means this is one of the target body
                float x = body_pos_w[(i*body_nb + j)*3];
                float y = body_pos_w[(i*body_nb + j)*3 + 1];
                float z = body_pos_w[(i*body_nb + j)*3 + 2];

                // quat should be in sequence of wxyz
                float quat_w = body_quat_w[(i*body_nb + j)*4];
                float quat_x = body_quat_w[(i*body_nb + j)*4 + 1];
                float quat_y = body_quat_w[(i*body_nb + j)*4 + 2];
                float quat_z = body_quat_w[(i*body_nb + j)*4 + 3];

                math_struct::Position<float> position(x, y, z);
                math_struct::Quat<float> quat(quat_x, quat_y, quat_z, quat_w);
                frame_body_positions.push_back(position);
                frame_body_quats.push_back(quat);

                if (j == anchor_body_idx)
                {                    
                    // which means this is anchor body
                    std::vector<float> anchor_pos;
                    anchor_pos.push_back(x);
                    anchor_pos.push_back(y);
                    anchor_pos.push_back(z);

                    std::vector<float> anchor_quat;
                    anchor_quat.push_back(quat_w);
                    anchor_quat.push_back(quat_x);
                    anchor_quat.push_back(quat_y);
                    anchor_quat.push_back(quat_z);
                    
                    root_positions_.push_back(anchor_pos);
                    root_quaternions_.push_back(anchor_quat);
                }
            }
            
        }
        joint_positions_.push_back(frame_joint_pos);
        joint_velocities_.push_back(frame_joint_vel);
    }
}

void VideoMimicMotionLoader::Update(float time)
{
    // Clamp time to valid range
    float phase = std::clamp(time / duration_, 0.0f, 1.0f);

    // Compute frame indices
    float frame_float = phase * (num_frames_ - 1);
    index_0_ = static_cast<int>(std::floor(frame_float));
    index_1_ = std::min(index_0_ + 1, num_frames_ - 1);

    index_10_future = static_cast<int>(std::floor(frame_float + 10));
    index_10_future = std::min(index_10_future, num_frames_ - 1);

    index_30_future = static_cast<int>(std::floor(frame_float + 30));
    index_30_future = std::min(index_30_future, num_frames_ - 1);

    index_60_future = static_cast<int>(std::floor(frame_float + 60));
    index_60_future = std::min(index_60_future, num_frames_ - 1);

    // Compute blend factor
    blend_ = frame_float - index_0_;
}

void VideoMimicMotionLoader::Reset(const std::vector<float>& robot_base_quat, const std::vector<float>& robot_waist_angles)
{
    Update(0.0f);

    std::vector<float> robot_torso = ComputeTorsoQuat(robot_base_quat, robot_waist_angles);
    std::vector<float> motion_torso = GetAnchorQuat();
    world_to_init_ = ComputeYawAlignment(robot_torso, motion_torso);
}

std::vector<float> VideoMimicMotionLoader::GetJointPos() const
{
    std::vector<float> result;
    const auto& pos0 = joint_positions_[index_0_];
    const auto& pos1 = joint_positions_[index_1_];

    for (size_t i = 0; i < pos0.size(); ++i)
    {
        result.push_back(pos0[i] * (1.0f - blend_) + pos1[i] * blend_);
    }
    return result;
}

std::vector<float> VideoMimicMotionLoader::GetJointVel() const
{
    std::vector<float> result;
    const auto& vel0 = joint_velocities_[index_0_];
    const auto& vel1 = joint_velocities_[index_1_];

    for (size_t i = 0; i < vel0.size(); ++i)
    {
        result.push_back(vel0[i] * (1.0f - blend_) + vel1[i] * blend_);
    }

    return result;
}

std::vector<float> VideoMimicMotionLoader::GetRootQuat() const
{
    std::vector<float> q = Slerp(root_quaternions_[index_0_], root_quaternions_[index_1_], blend_);
    return q;
}

std::vector<float> VideoMimicMotionLoader::ComputeTorsoQuat(const std::vector<float>& base_quat, 
    const std::vector<float>& waist_angles)
{
    std::vector<float> q_yaw = QuaternionFromAxisAngle({0.0f, 0.0f, 1.0f}, waist_angles[0]);
    std::vector<float> q_roll = QuaternionFromAxisAngle({1.0f, 0.0f, 0.0f}, waist_angles[1]);
    std::vector<float> q_pitch = QuaternionFromAxisAngle({0.0f, 1.0f, 0.0f}, waist_angles[2]);

    std::vector<float> torso_quat = QuaternionMultiply(base_quat, q_yaw);
    torso_quat = QuaternionMultiply(torso_quat, q_roll);
    torso_quat = QuaternionMultiply(torso_quat, q_pitch);

    return QuaternionNormalize(torso_quat);
}

std::vector<float> VideoMimicMotionLoader::ComputeYawAlignment(const std::vector<float>& robot_torso_quat, 
    const std::vector<float>& motion_torso_quat)
{
    std::vector<float> robot_yaw = QuaternionYawOnly(robot_torso_quat);
    std::vector<float> motion_yaw = QuaternionYawOnly(motion_torso_quat);
    return QuaternionMultiply(robot_yaw, QuaternionConjugate(motion_yaw));
}

std::vector<float> VideoMimicMotionLoader::GetRootPos() const
{
    std::vector<float> anchor_pos = Slerp(root_positions_[index_0_], root_positions_[index_1_], blend_);
    return anchor_pos;
}

std::vector<float> VideoMimicMotionLoader::GetFutureRootPos() const
{
    std::vector<float> future_10_pos = root_positions_[index_10_future];
    std::vector<float> future_30_pos = root_positions_[index_30_future];
    std::vector<float> future_60_pos = root_positions_[index_60_future];
    std::vector<float> combine_future_pos(9, 0.0);
    combine_future_pos[0] = future_10_pos[0];
    combine_future_pos[1] = future_10_pos[1];
    combine_future_pos[2] = future_10_pos[2];

    combine_future_pos[3] = future_30_pos[0];
    combine_future_pos[4] = future_30_pos[1];
    combine_future_pos[5] = future_30_pos[2];

    combine_future_pos[6] = future_60_pos[0];
    combine_future_pos[7] = future_60_pos[1];
    combine_future_pos[8] = future_60_pos[2];
    return combine_future_pos;
}

std::vector<float> VideoMimicMotionLoader::GetAnchorQuat() const
{
    std::vector<float> root_quat = Slerp(root_quaternions_[index_0_], root_quaternions_[index_1_], blend_);
    auto joint_pos = GetJointPos();

    const int WAIST_YAW_IDX = 12;
    const int WAIST_ROLL_IDX = 13;
    const int WAIST_PITCH_IDX = 14;

    std::vector<float> waist_angles = {joint_pos[WAIST_YAW_IDX], joint_pos[WAIST_ROLL_IDX], joint_pos[WAIST_PITCH_IDX]};
    return ComputeTorsoQuat(root_quat, waist_angles);
}

std::vector<float> VideoMimicMotionLoader::Slerp(const std::vector<float>& q0, const std::vector<float>& q1, float t) const
{
    // Compute dot product
    float dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];

    // If dot < 0, negate q1 to take shorter path
    std::vector<float> q1_adjusted = q1;
    if (dot < 0.0f)
    {
        q1_adjusted = {-q1[0], -q1[1], -q1[2], -q1[3]};
        dot = -dot;
    }

    // If quaternions are very close, use linear interpolation
    if (dot > 0.9995f)
    {
        return QuaternionNormalize({
            q0[0] + t * (q1_adjusted[0] - q0[0]),
            q0[1] + t * (q1_adjusted[1] - q0[1]),
            q0[2] + t * (q1_adjusted[2] - q0[2]),
            q0[3] + t * (q1_adjusted[3] - q0[3])
        });
    }

    // Spherical interpolation
    float theta = std::acos(std::clamp(dot, -1.0f, 1.0f));
    float sin_theta = std::sin(theta);

    float w0 = std::sin((1.0f - t) * theta) / sin_theta;
    float w1 = std::sin(t * theta) / sin_theta;

    return {
        q0[0] * w0 + q1_adjusted[0] * w1,
        q0[1] * w0 + q1_adjusted[1] * w1,
        q0[2] * w0 + q1_adjusted[2] * w1,
        q0[3] * w0 + q1_adjusted[3] * w1
    };
}

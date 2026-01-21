#ifndef VIDEO_MIMIC_MOTION_LOADER_HPP
#define VIDEO_MIMIC_MOTION_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "vector_math.hpp"
#include "../logger/logger.hpp"

#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

class VideoMimicMotionLoader
{
public:
    VideoMimicMotionLoader(const std::string& motion_file, float fps, pinocchio::Model model_pin,
        const std::vector<float>& root_pos,
        const std::vector<float>& root_quat,
        const std::vector<float>& joint_pos);

    void LoadVideoMimicCSV(const std::string& filename);

    void GenerateStaticStanding(pinocchio::Model model_pin, 
        const std::vector<float>& root_pos,
        const std::vector<float>& root_quat,
        const std::vector<float>& joint_pos);

    void Update(float time);

    std::vector<float> GetJointPos() const;

    std::vector<float> GetJointVel() const;

    std::vector<float> GetRootQuat() const;

    std::vector<float> GetRootPos() const;

    std::vector<float> GetFutureRootPos() const;

    std::vector<float> GetAnchorQuat() const;

    float GetDuration() const { return duration_; }

    std::vector<float> Lerp(const std::vector<float>& p0, const std::vector<float>& p1, float t) const;

    std::vector<float> Slerp(const std::vector<float>& q0, const std::vector<float>& q1, float t) const;

    void Reset(const std::vector<float>& robot_base_quat, const std::vector<float>& robot_waist_angles);

    static std::vector<float> ComputeTorsoQuat(const std::vector<float>& base_quat, const std::vector<float>& waist_angles);

    static std::vector<float> ComputeYawAlignment(const std::vector<float>& robot_torso_quat, const std::vector<float>& motion_torso_quat);

    std::vector<float> GetInitQuat() const { return world_to_init_; }

    // Motion data storage
    std::vector<std::vector<float>> root_positions_;     // [T, 3]
    std::vector<std::vector<float>> root_quaternions_;   // [T, 4] - each is [w, x, y, z]
    std::vector<std::vector<float>> joint_positions_;    // [T, N]
    std::vector<std::vector<float>> joint_velocities_;   // [T, N]

    float fps = 30;

    // Motion properties
    int num_frames_;
    int num_joints_;
    float dt_;           // Time between frames
    float duration_;     // Total duration

    // Current interpolation state
    int index_0_;        // Current frame index
    int index_1_;        // Next frame index
    float blend_;        // Interpolation factor [0, 1]

    int index_10_future;
    int index_30_future;
    int index_60_future;

    std::vector<float> world_to_init_;  // For yaw alignment between robot and motion [w, x, y, z]
};


#endif
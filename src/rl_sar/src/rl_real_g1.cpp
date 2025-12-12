/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_g1.hpp"

RL_Real::RL_Real(int argc, char **argv)
{
#if defined(USE_ROS1) && defined(USE_ROS)
    ros::NodeHandle nh;
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real::CmdvelCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    ros2_node = std::make_shared<rclcpp::Node>("rl_real_node");
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
#endif

    // read params from yaml
    this->ang_vel_axis = "body";
    this->robot_name = "g1";
    this->ReadYaml(this->robot_name, "base.yaml");

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
    this->mode_pr = Mode::PR;
    this->mode_machine = 0;
    this->InitLowCmd();
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();
    // init MotionSwitcherClient
    this->msc.SetTimeout(5.0f);
    this->msc.Init();
    // Shut down motion control-related service
    std::string form, name;
    while (this->msc.CheckMode(form, name), !name.empty())
    {
        if (this->msc.ReleaseMode())
        {
            std::cout << "Failed to switch to Release Mode" << std::endl;
        }
        sleep(5);
    }
    // create lowcmd publisher
    this->lowcmd_publisher.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    this->lowcmd_publisher->InitChannel();
    // create lowstate subscriber
    this->lowstate_subscriber.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    this->lowstate_subscriber->InitChannel(std::bind(&RL_Real::LowStateHandler, this, std::placeholders::_1), 1);
    // create imutorso subscriber
    this->imutorso_subscriber.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
    this->imutorso_subscriber->InitChannel(std::bind(&RL_Real::ImuTorsoHandler, this, std::placeholders::_1), 1);

    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::GetState(RobotState<float> *state)
{
    if (this->mode_machine != this->unitree_low_state.mode_machine())
    {
        if (this->mode_machine == 0)
        {
            std::cout << "G1 type: " << unsigned(this->unitree_low_state.mode_machine()) << std::endl;
        }
        this->mode_machine = this->unitree_low_state.mode_machine();
    }

    memcpy(this->remote_data_rx.buff, &unitree_low_state.wireless_remote()[0], 40);
    this->gamepad.update(this->remote_data_rx.RF_RX);

    if (this->gamepad.A.pressed) this->control.SetGamepad(Input::Gamepad::A);
    if (this->gamepad.B.pressed) this->control.SetGamepad(Input::Gamepad::B);
    if (this->gamepad.X.pressed) this->control.SetGamepad(Input::Gamepad::X);
    if (this->gamepad.Y.pressed) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->gamepad.R1.pressed) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->gamepad.L1.pressed) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->gamepad.F1.pressed) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->gamepad.F2.pressed) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->gamepad.up.pressed) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->gamepad.down.pressed) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->gamepad.left.pressed) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->gamepad.right.pressed) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->gamepad.L1.pressed && this->gamepad.A.pressed) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->gamepad.L1.pressed && this->gamepad.B.pressed) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->gamepad.L1.pressed && this->gamepad.X.pressed) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->gamepad.L1.pressed && this->gamepad.Y.pressed) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->gamepad.L1.pressed && this->gamepad.F1.pressed) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->gamepad.L1.pressed && this->gamepad.F2.pressed) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->gamepad.L1.pressed && this->gamepad.up.pressed) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->gamepad.L1.pressed && this->gamepad.down.pressed) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->gamepad.L1.pressed && this->gamepad.left.pressed) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->gamepad.L1.pressed && this->gamepad.right.pressed) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->gamepad.R1.pressed && this->gamepad.A.pressed) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->gamepad.R1.pressed && this->gamepad.B.pressed) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->gamepad.R1.pressed && this->gamepad.X.pressed) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->gamepad.R1.pressed && this->gamepad.Y.pressed) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->gamepad.R1.pressed && this->gamepad.F1.pressed) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->gamepad.R1.pressed && this->gamepad.F2.pressed) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->gamepad.R1.pressed && this->gamepad.up.pressed) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->gamepad.R1.pressed && this->gamepad.down.pressed) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->gamepad.R1.pressed && this->gamepad.left.pressed) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->gamepad.R1.pressed && this->gamepad.right.pressed) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->gamepad.L1.pressed && this->gamepad.R1.pressed) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->gamepad.ly;
    this->control.y = -this->gamepad.lx;
    this->control.yaw = -this->gamepad.rx;

    state->imu.quaternion[0] = this->unitree_low_state.imu_state().quaternion()[0]; // w
    state->imu.quaternion[1] = this->unitree_low_state.imu_state().quaternion()[1]; // x
    state->imu.quaternion[2] = this->unitree_low_state.imu_state().quaternion()[2]; // y
    state->imu.quaternion[3] = this->unitree_low_state.imu_state().quaternion()[3]; // z

    for (int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->unitree_low_state.imu_state().gyroscope()[i];
    }


    float w = state->imu.quaternion[0];
    float x = state->imu.quaternion[1];
    float y = state->imu.quaternion[2];
    float z = state->imu.quaternion[3];

    float t0 = 2.0 * (w * x + y * z);
    float t1 = 1.0 - 2.0 * (x * x + y * y);
    float roll = std::atan2(t0, t1);

    float t2 = 2.0 * (w * y - z * x);
    if (t2 < -0.1)
    {
        t2 = -1.0;
    }
    else if (t2 > 1.0)
    {
        t2 = 1.0;
    }
    float pitch = std::asin(t2);

    auto wrap_to_pi = [](float angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    };
        
    float final_roll = wrap_to_pi(roll);
    float final_pitch = wrap_to_pi(pitch);
        
    cur_imu = {final_roll, final_pitch};



    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        state->motor_state.q[i] = this->unitree_low_state.motor_state()[this->params.Get<std::vector<int>>("joint_mapping")[i]].q();
        state->motor_state.dq[i] = this->unitree_low_state.motor_state()[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq();
        state->motor_state.tau_est[i] = this->unitree_low_state.motor_state()[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est();
        if (cur_joint_pos.size() == 0)
        {
            cur_joint_pos.assign(29, 0.0);
        }
        cur_joint_pos[i] = state->motor_state.q[i];
    }

    /*
    if (model_pin.nq > 0)
    {
        Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_pin.nq);

        q_pin[0] = mj_data->qpos[0];
        q_pin[1] = mj_data->qpos[1];
        q_pin[2] = mj_data->qpos[2];

        q_pin[3] = mj_data->qpos[4];
        q_pin[4] = mj_data->qpos[5];
        q_pin[5] = mj_data->qpos[6];
        q_pin[6] = mj_data->qpos[3];

        int base_offset = 7;
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            if (base_offset + i < q_pin.size()) 
            {
                q_pin[base_offset + i] = state->motor_state.q[i];
            }
        }

        // size of q_pin is 29
        // after reading motion.npz, size of q_pin is 58
        pinocchio::forwardKinematics(model_pin, data_pin, q_pin);

        mjv_updateScene(sim->m_, sim->d_, &sim->opt, &sim->pert, &sim->cam, mjCAT_ALL, &sim->scn);
            
        sim->geoms_.clear();
        mjtNum geom_size[3] = {0.05, 0.0, 0.0};
        float geom_color[4] = {1.0, 0.0, 0.0, 1.0};
        float geom_green_color[4] = {0.0, 1.0, 0.0, 1.0};



        pinocchio::SE3 world_M_root;
        for (pinocchio::JointIndex joint_id = 1; joint_id < model_pin.joints.size(); ++joint_id)
        {
            const auto & world_transform = data_pin.oMi[joint_id];
            if (joint_id == 1)
            {
                world_M_root = data_pin.oMi[joint_id];
            }

            pinocchio::SE3 local_joint_transform = world_M_root.inverse() * world_transform;

            Eigen::Vector3d world_translation = world_transform.translation();
            Eigen::Quaterniond world_quat(world_transform.rotation());
            Eigen::Vector3d local_translation = local_joint_transform.translation();
            Eigen::Quaterniond local_quat(local_joint_transform.rotation());

            pinocchio::Motion v_local = data_pin.v[joint_id];
            pinocchio::Motion v_joint = data_pin.oMi[joint_id].act(v_local);
            pinocchio::Motion v_in_root_frame = local_joint_transform.act(v_local);

            Eigen::Vector3d world_linear_vel = v_joint.linear();
            Eigen::Vector3d world_angular_vel = v_joint.angular();
            Eigen::Vector3d local_linear_vel = v_in_root_frame.linear();
            Eigen::Vector3d local_angular_vel = v_in_root_frame.angular();

            if (joint_id == 1)
            {
                root_world_joint_translation[0] = world_translation.x();
                root_world_joint_translation[1] = world_translation.y();
                root_world_joint_translation[2] = world_translation.z();

                root_world_joint_quat[0] = world_quat.w();
                root_world_joint_quat[1] = world_quat.x();
                root_world_joint_quat[2] = world_quat.y();
                root_world_joint_quat[3] = world_quat.z();

                root_world_joint_lin_vel[0] = world_linear_vel.x();
                root_world_joint_lin_vel[1] = world_linear_vel.y();
                root_world_joint_lin_vel[2] = world_linear_vel.z();

                root_world_joint_ang_vel[0] = world_angular_vel.x();
                root_world_joint_ang_vel[1] = world_angular_vel.y();
                root_world_joint_ang_vel[2] = world_angular_vel.z();


                root_local_joint_translation[0] = local_translation.x();
                root_local_joint_translation[1] = local_translation.y();
                root_local_joint_translation[2] = local_translation.z();

                root_local_joint_quat[0] = local_quat.w();
                root_local_joint_quat[1] = local_quat.x();
                root_local_joint_quat[2] = local_quat.y();
                root_local_joint_quat[3] = local_quat.z();

                root_local_joint_lin_vel[0] = local_linear_vel.x();
                root_local_joint_lin_vel[1] = local_linear_vel.y();
                root_local_joint_lin_vel[2] = local_linear_vel.z();

                root_local_joint_ang_vel[0] = local_angular_vel.x();
                root_local_joint_ang_vel[1] = local_angular_vel.y();
                root_local_joint_ang_vel[2] = local_angular_vel.z();
            }

            mjtNum geom_pos[3];
            geom_pos[0] = world_translation.x();
            geom_pos[1] = world_translation.y();
            geom_pos[2] = world_translation.z();


            sim->geoms_.push_back({});
            auto & geom = sim->geoms_.back();
            mjv_initGeom(&geom, mjGEOM_SPHERE, geom_size, geom_pos, NULL, geom_color);


            mjtNum geom_local_pos[3];
            geom_local_pos[0] = local_translation.x();
            geom_local_pos[1] = local_translation.y();
            geom_local_pos[2] = local_translation.z();

            sim->geoms_.push_back({});
            auto & local_geom = sim->geoms_.back();
            mjv_initGeom(&local_geom, mjGEOM_SPHERE, geom_size, geom_local_pos, NULL, geom_green_color);
        }
    }
    */
}

void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    this->unitree_low_command.mode_pr() = static_cast<uint8_t>(this->mode_pr);
    this->unitree_low_command.mode_machine() = this->mode_machine;

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->unitree_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].mode() = 1; // 1:Enable, 0:Disable
        this->unitree_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].q() = command->motor_command.q[i];
        this->unitree_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq() = command->motor_command.dq[i];
        this->unitree_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp() = command->motor_command.kp[i];
        this->unitree_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd() = command->motor_command.kd[i];
        this->unitree_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau() = command->motor_command.tau[i];
    }

    this->unitree_low_command.crc() = Crc32Core((uint32_t *)&unitree_low_command, (sizeof(LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(unitree_low_command);
}

void RL_Real::RobotControl()
{
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
#if !defined(USE_CMAKE) && defined(USE_ROS)
        if (this->control.navigation_mode)
        {
            this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};

        }
#endif
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est = this->robot_state.motor_state.tau_est;
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Real::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (!this->params.Get<std::vector<int>>("observations_history").empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motor_state()[i].q());
        this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motor_cmd()[i].q());
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

uint32_t RL_Real::Crc32Core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; ++i)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }
            xbit >>= 1;
        }
    }

    return CRC32;
}

void RL_Real::InitLowCmd()
{
    for (int i = 0; i < 32; ++i)
    {
        this->unitree_low_command.motor_cmd()[i].mode() = (1); // 1:Enable, 0:Disable
        this->unitree_low_command.motor_cmd()[i].q() = (0);
        this->unitree_low_command.motor_cmd()[i].kp() = (0);
        this->unitree_low_command.motor_cmd()[i].dq() = (0);
        this->unitree_low_command.motor_cmd()[i].kd() = (0);
        this->unitree_low_command.motor_cmd()[i].tau() = (0);
    }
}

void RL_Real::LowStateHandler(const void *message)
{
    this->unitree_low_state = *(const LowState_ *)message;
}

void RL_Real::ImuTorsoHandler(const void *message)
{
    this->unitree_imu_torso = *(const IMUState_ *)message;
}

#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}
#endif

#if defined(USE_ROS1) && defined(USE_ROS)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " networkInterface" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    ChannelFactory::Instance()->Init(0, argv[1]);

#if defined(USE_ROS1) && defined(USE_ROS)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Real rl_sar(argc, argv);
    ros::spin();
#elif defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Real>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || !defined(USE_ROS)
    RL_Real rl_sar(argc, argv);
    while (1) { sleep(10); }
#endif

    return 0;
}

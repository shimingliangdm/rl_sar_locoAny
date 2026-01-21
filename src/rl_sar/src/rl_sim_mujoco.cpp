/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"
#include <cmath>


RL_Sim* RL_Sim::instance = nullptr;

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
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
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        
        state->imu.quaternion[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 0];
        state->imu.quaternion[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 1];
        state->imu.quaternion[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 2];
        state->imu.quaternion[3] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 3];

        state->imu.gyroscope[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 4];
        state->imu.gyroscope[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 5];
        state->imu.gyroscope[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 6];

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            state->motor_state.q[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            state->motor_state.dq[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")];
            state->motor_state.tau_est[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + 2 * this->params.Get<int>("num_of_dofs")];
        }

        state->real_root_pos[0] = mj_data->qpos[0];
        state->real_root_pos[1] = mj_data->qpos[1];
        state->real_root_pos[2] = mj_data->qpos[2];

        state->real_root_quat[0] = mj_data->qpos[3];
        state->real_root_quat[1] = mj_data->qpos[4];
        state->real_root_quat[2] = mj_data->qpos[5];
        state->real_root_quat[3] = mj_data->qpos[6];

    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
        }
    }
}

void RL_Sim::GetCurrentTestStates(std::vector<float>& root_pos, std::vector<float>& root_quat, std::vector<float>& out_joint_pos)
{
    std::vector<float> current_root_pos = 
    {
        (float)mj_data->qpos[0],
        (float)mj_data->qpos[1],
        (float)mj_data->qpos[2]
    };

    std::vector<float> current_root_quat = 
    {
        (float)mj_data->qpos[3],
        (float)mj_data->qpos[4],
        (float)mj_data->qpos[5],
        (float)mj_data->qpos[6]
    };

    std::vector<float> joint_pos;
    for (int i = 0; i < 29; ++i)
    {
        joint_pos.push_back((float)mj_data->qpos[7 + i]);
    }

    root_pos = current_root_pos;
    root_quat = current_root_quat;
    out_joint_pos = joint_pos;
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.base_pos = this->robot_state.real_root_pos;
        Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_pin.nq);
        // dof_pos is in order of IsaacLab
        q_pin[0] = this->robot_state.real_root_pos[0];
        q_pin[1] = this->robot_state.real_root_pos[1];
        q_pin[2] = this->robot_state.real_root_pos[2];

        q_pin[3] = this->robot_state.real_root_quat[1];
        q_pin[4] = this->robot_state.real_root_quat[2];
        q_pin[5] = this->robot_state.real_root_quat[3];
        q_pin[6] = this->robot_state.real_root_quat[0];

        auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
        for (int i = 0; i < joint_mapping.size(); ++i)
        {
            q_pin[7 + joint_mapping[i]] = this->obs.dof_pos[i];
        }
        pinocchio::forwardKinematics(model_pin, data_pin, q_pin);
        pinocchio::updateFramePlacements(model_pin, data_pin);
        int pelvis_id = model_pin.getFrameId("pelvis");

        Eigen::Vector3d eigen_anchor_pos_w = data_pin.oMf[pelvis_id].translation();
        Eigen::Matrix3d eigen_anchor_rot_w = data_pin.oMf[pelvis_id].rotation();
        if (this->obs.is_first_record)
        {
            this->obs.last_real_anchor_pos_w[0] = eigen_anchor_pos_w.x();
            this->obs.last_real_anchor_pos_w[1] = eigen_anchor_pos_w.y();
            this->obs.last_real_anchor_pos_w[2] = eigen_anchor_pos_w.z();
            this->obs.is_first_record = false;
        }

        std::vector<float> real_anchor_pos_w = {0.0, 0.0, 0.0};
        real_anchor_pos_w[0] = eigen_anchor_pos_w.x();
        real_anchor_pos_w[1] = eigen_anchor_pos_w.y();
        real_anchor_pos_w[2] = eigen_anchor_pos_w.z();

        float dt = this->params.Get<float>("dt");
        std::vector<float> real_anchor_lin_vel_w = {0.0, 0.0, 0.0};
        real_anchor_lin_vel_w[0] = (real_anchor_pos_w[0] - this->obs.last_real_anchor_pos_w[0])/dt;
        real_anchor_lin_vel_w[1] = (real_anchor_pos_w[1] - this->obs.last_real_anchor_pos_w[1])/dt;
        real_anchor_lin_vel_w[2] = (real_anchor_pos_w[2] - this->obs.last_real_anchor_pos_w[2])/dt;

        this->obs.last_real_anchor_pos_w = real_anchor_pos_w;


        Eigen::Vector3d eigen_anchor_lin_vel_w;
        eigen_anchor_lin_vel_w[0] = real_anchor_lin_vel_w[0];
        eigen_anchor_lin_vel_w[1] = real_anchor_lin_vel_w[1];
        eigen_anchor_lin_vel_w[2] = real_anchor_lin_vel_w[2];
        Eigen::Matrix3d mat_world2robot = eigen_anchor_rot_w.transpose();
        Eigen::Vector3d eigen_anchor_lin_vel_b = mat_world2robot * eigen_anchor_lin_vel_w;
        this->obs.lin_vel[0] = eigen_anchor_lin_vel_b.x();
        this->obs.lin_vel[1] = eigen_anchor_lin_vel_b.y();
        this->obs.lin_vel[2] = eigen_anchor_lin_vel_b.z();


        float imu_w = this->robot_state.imu.quaternion[0];
        float imu_x = this->robot_state.imu.quaternion[1];
        float imu_y = this->robot_state.imu.quaternion[2];
        float imu_z = this->robot_state.imu.quaternion[3];

        float t0 = 2.0 * (imu_w * imu_x + imu_y * imu_z);
        float t1 = 1.0 - 2.0 * (imu_x * imu_x + imu_y * imu_y);
        float roll = std::atan2(t0, t1);

        float t2 = 2.0 * (imu_w * imu_y - imu_z * imu_x);
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
        
        this->obs.imu_roll = wrap_to_pi(roll);
        this->obs.imu_pitch = wrap_to_pi(pitch);
        



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
            auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
            /*joint_mapping: [0, 6, 12, 1, 7, 13, 2, 8, 14, 
                3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 
                18, 25, 19, 26, 20, 27, 21, 28]*/
            for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
            {
                int pinocchio_index = base_offset + joint_mapping[i];
                if (pinocchio_index < q_pin.size())
                {
                    q_pin[pinocchio_index] = this->robot_state.motor_state.q[i];
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

            // here is the name of each joint
            /******
            [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
            Index: 0 | Name: universe
            Index: 1 | Name: root_joint
            Index: 2 | Name: left_hip_pitch_joint
            Index: 3 | Name: left_hip_roll_joint
            Index: 4 | Name: left_hip_yaw_joint
            Index: 5 | Name: left_knee_joint
            Index: 6 | Name: left_ankle_pitch_joint
            Index: 7 | Name: left_ankle_roll_joint
            Index: 8 | Name: right_hip_pitch_joint
            Index: 9 | Name: right_hip_roll_joint
            Index: 10 | Name: right_hip_yaw_joint
            Index: 11 | Name: right_knee_joint
            Index: 12 | Name: right_ankle_pitch_joint
            Index: 13 | Name: right_ankle_roll_joint
            Index: 14 | Name: waist_yaw_joint
            Index: 15 | Name: waist_roll_joint
            Index: 16 | Name: waist_pitch_joint
            Index: 17 | Name: left_shoulder_pitch_joint
            Index: 18 | Name: left_shoulder_roll_joint
            Index: 19 | Name: left_shoulder_yaw_joint
            Index: 20 | Name: left_elbow_joint
            Index: 21 | Name: left_wrist_roll_joint
            Index: 22 | Name: left_wrist_pitch_joint
            Index: 23 | Name: left_wrist_yaw_joint
            Index: 24 | Name: right_shoulder_pitch_joint
            Index: 25 | Name: right_shoulder_roll_joint
            Index: 26 | Name: right_shoulder_yaw_joint
            Index: 27 | Name: right_elbow_joint
            Index: 28 | Name: right_wrist_roll_joint
            Index: 29 | Name: right_wrist_pitch_joint
            Index: 30 | Name: right_wrist_yaw_joint
            */

            pinocchio::SE3 world_M_root = data_pin.oMi[15];
            for (pinocchio::JointIndex joint_id = 2; joint_id < model_pin.joints.size(); ++joint_id)
            {
                const auto & world_transform = data_pin.oMi[joint_id];

                pinocchio::SE3 local_joint_transform = world_M_root.inverse() * world_transform;

                Eigen::Vector3d world_translation = world_transform.translation();
                Eigen::Vector3d local_translation = local_joint_transform.translation();

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
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
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
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }
    this->obs.last_raw_actions = actions;

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}

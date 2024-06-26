#include "controller.h"

using namespace control_interface;


void ruckig_controller::ruckig_state_manage()
{
    if (ruckig_activate) {
        ruckig::Result ruckig_result = ruckig_ptr->update(ruckig_input, ruckig_output);
        if (ruckig_result != ruckig::Result::Finished) {
            auto& p = ruckig_output.new_velocity;
            tm_msgs::SetVelocity vel_srv;
            std::vector < double > cmd_vel(NUM_DOF);
            for (size_t i = 0; i < NUM_DOF; ++i) {
                cmd_vel.at(i) = p.at(i);
                current_accelerations.at(i) = std::clamp(
                    ruckig_output.new_acceleration.at(i), -ruckig_input.max_acceleration.at(i), ruckig_input.max_acceleration.at(i));
            }
            vel_srv.request.motion_type = 1;
            vel_srv.request.velocity = cmd_vel;
            vel_client.call(vel_srv);
            ruckig_output.pass_to_input(ruckig_input);
        } else {
            ROS_INFO("Target Reached!!");
            ruckig_activate = false;
            tm_msgs::SetVelocity vel_srv;
            std::vector < double > cmd_vel(NUM_DOF, 0.0);
            vel_srv.request.motion_type = 1;
            vel_srv.request.velocity = cmd_vel;
            vel_client.call(vel_srv);
        }
    }
}


// This callback function is for setting current_positions, current_velocities and current_accelerations from 
// topic: /joint_states


void ruckig_controller::joint_callback(const sensor_msgs::JointState& joint_msg)
{
    for (size_t i = 0; i < NUM_DOF; ++i) {
        current_velocities.at(i) =
            std::clamp(joint_msg.velocity.at(i), -ruckig_input.max_velocity.at(i), ruckig_input.max_velocity.at(i));
        current_positions.at(i) = joint_msg.position.at(i);
    }
    if (!ruckig_activate) {
        current_accelerations.fill(0.0);
    }
}


// Assign ruckig_ptr a pointer by using std::make_unique to make a object which belong to class ruckig::Ruckig, this object
// is initialized by num_dof and ruckig_timestep) 


void ruckig_controller::target_joint_callback(const sensor_msgs::JointState& target_joint_msg)
{
    std::vector < double > joint_values(NUM_DOF);

    ruckig_ptr = std::make_unique < ruckig::Ruckig < ruckig::DynamicDOFs >> (NUM_DOF, ruckig_timestep);
    std::copy_n(target_joint_msg.position.begin(), NUM_DOF, joint_values.begin());

    getNextRuckigInput(joint_values, 0.3);
}


// This callback function will take xyz and quarternion as input and then move the arm to that popse if valid


void ruckig_controller::target_callback(const geometry_msgs::Pose& target_msg)
{
    std::vector < double > candidate_ik(NUM_DOF, 0.0);
    std::vector < double > joint_values(NUM_DOF, 0.0);

    auto element_wise_dist =
        [] (auto& source_vec, auto& target_vec)
    {
        double dist = 0.0;
        for (size_t i = 0; i < NUM_DOF; ++i) {
            dist += abs(source_vec.at(i) - target_vec.at(i));
        }
        return dist;
    };

    double min_dist = 999.0;

    for (int i = 0; i < ik_attemp; i++) {
        std::copy_n(current_positions.begin(), NUM_DOF, joint_values.begin());
        kinematic_state->setJointGroupPositions(joint_model_group, joint_values);

        bool found_ik = kinematic_state->setFromIK(joint_model_group, target_msg, ik_timeout);

        if (found_ik) {
            kinematic_state->copyJointGroupPositions(joint_model_group, joint_values);
            double current_dist = element_wise_dist(current_positions, joint_values);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                std::copy_n(joint_values.begin(), NUM_DOF, candidate_ik.begin());
            }
        }
    }

    if (std::count(candidate_ik.begin(), candidate_ik.end(), 0.0) != NUM_DOF) {
        // For debug: print ik value
        // for (std::size_t i = 0; i < candidate_ik.size(); ++i) {
        //     ROS_INFO("Joint %ld: %f", i + 1, candidate_ik[i]);
        // }

        ruckig_ptr = std::make_unique < ruckig::Ruckig < ruckig::DynamicDOFs >> (NUM_DOF, ruckig_timestep);
        getNextRuckigInput(joint_values, 0.);
    } else {
        ROS_WARN("Did not find IK solution");
    }
}

void ruckig_controller::ruckig_stop()
{
    ruckig_input.control_interface = ruckig::ControlInterface::Velocity;
    ruckig_input.synchronization = ruckig::Synchronization::None;
    ruckig_input.target_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    ruckig_input.target_acceleration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

void ruckig_controller::getNextRuckigInput(std::vector < double > joint_input, double max_vel)
{
    kinematic_state->setJointGroupPositions(joint_model_group, joint_input);

    if (kinematic_state->satisfiesBounds()) {
        for (size_t joint = 0; joint < NUM_DOF; ++joint) {
            // Target state is the next waypoint
            ruckig_input.target_position.at(joint) = joint_input.at(joint);
            ruckig_input.target_velocity.at(joint) = 0.0;
            ruckig_input.target_acceleration.at(joint) = 0.0;
        }
        initializeRuckigState(max_vel);
        ROS_INFO("Recieve New Target Point!!");
        ruckig_activate = true;
    } else {
        ROS_WARN("Target position over joint limit");
    }
}

void ruckig_controller::initializeRuckigState(double max_vel)
{
    ruckig_input.control_interface = ruckig::ControlInterface::Position;
    std::copy_n(current_positions.begin(), NUM_DOF, ruckig_input.current_position.begin());
    std::copy_n(current_velocities.begin(), NUM_DOF, ruckig_input.current_velocity.begin());
    std::copy_n(current_accelerations.begin(), NUM_DOF, ruckig_input.current_acceleration.begin());
    if (max_vel) {
        auto tmp_vel = std::vector(NUM_DOF, max_vel);
        std::copy_n(tmp_vel.begin(), NUM_DOF, ruckig_input.max_velocity.begin());
    } else {
        std::copy_n(max_velocities.begin(), NUM_DOF, ruckig_input.max_velocity.begin());
    }
    // std::copy_n(max_velocities.begin(), NUM_DOF, ruckig_input.max_velocity.begin());
    std::copy_n(max_accelerations.begin(), NUM_DOF, ruckig_input.max_acceleration.begin());
    std::copy_n(max_jerks.begin(), NUM_DOF, ruckig_input.max_jerk.begin());
    // Initialize output data struct
    ruckig_output.new_position = ruckig_input.current_position;
    ruckig_output.new_velocity = ruckig_input.current_velocity;
    ruckig_output.new_acceleration = ruckig_input.current_acceleration;
}

void ruckig_controller::script_call()
{
    tm_msgs::SendScript script_srv;

    script_srv.request.id = "Vstart";
    script_srv.request.script = "ContinueVJog()";
    script_client.call(script_srv);
    ROS_INFO("Sending script service request to tm_driver...");
}

void ruckig_controller::run()
{
    ruckig_state_manage();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "controller_node");
    ros::NodeHandle n;
    double ros_rate = 25.0;
    auto controller_node = std::make_shared < control_interface::ruckig_controller > (n, ros_rate);
    ros::Rate rate(ros_rate);
    controller_node->script_call();
    ROS_INFO("Controller Node Startup");
    while (ros::ok()) {
        controller_node->run();
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}

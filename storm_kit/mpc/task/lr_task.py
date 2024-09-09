#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import copy
import torch
import yaml
import numpy as np

from ...util_file import get_mpc_configs_path as mpc_configs_path
from ...mpc.rollout.arm_reacher import ArmReacher
from ...mpc.rollout.lr_rollout import LimbRepoRollout
from ...mpc.control import MPPI
from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess
from ...mpc.utils.lr_mpc_process_wrapper import LimbRepoControlProcess
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from .arm_task import ArmTask
from ...util_limb_repo import LRState, ABState


class LimbRepoTask(ArmTask):
    '''
    Initializes all MPC things (MPPI, rollout, control process) 
    and provides a get_command function to get the next command to be executed by the robot.
    '''
    def __init__(self, task_file='ur10.yml', robot_file='ur10_reacher.yml', world_file='collision_env.yml', tensor_args={'device':"cpu", 'dtype':torch.float32}):
        
        self.tensor_args = tensor_args
        self.controller = self.init_mppi(task_file, robot_file, world_file)
        super().__init__(task_file=task_file, robot_file=robot_file,
                         world_file=world_file, tensor_args=tensor_args)
        self.control_process = LimbRepoControlProcess(self.controller)
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)
        self.passive_state_filter = JointStateFilter(filter_coeff=self.exp_params['state_filter_coeff'], dt=self.exp_params['control_dt'])
        
    def _state_to_tensor(self, state: ABState):
        state_tensor = np.concatenate((state.pos, state.vel, state.acc))

        state_tensor = torch.tensor(state_tensor)
        return state_tensor

    def get_rollout_fn(self, **kwargs):
        rollout_fn = LimbRepoRollout(**kwargs)
        return rollout_fn

    def get_command(self, t_step, curr_state: LRState, control_dt, WAIT=False) -> LRState:
        '''
        Predict forward from previous action and previous state.
        Returns LRState.
        '''

        curr_state = copy.deepcopy(curr_state)
        curr_state_dict = curr_state.active.to_dict()
        passive_curr_state_dict = curr_state.passive.to_dict()

        # set initial values to 0 to prevent certain errors relating to initializing as None
        if(self.state_filter.cmd_joint_state is None):
            curr_state_dict['velocity'] *= 0.0
            self.state_filter.prev_cmd_qd = curr_state_dict['velocity']

        if(self.passive_state_filter.cmd_joint_state is None):
            passive_curr_state_dict['velocity'] *= 0.0
            self.passive_state_filter.prev_cmd_qd = passive_curr_state_dict['velocity']


        # filter the state (a * current joint state + (1-a) * last commanded joint state)
        filt_state_dict = self.state_filter.filter_joint_state(curr_state_dict)
        filt_state = ABState.from_dict(filt_state_dict)

        passive_filt_state_dict = self.passive_state_filter.filter_joint_state(passive_curr_state_dict)
        passive_filt_state = ABState.from_dict(passive_filt_state_dict)

        filt_state = LRState(active=filt_state, passive=passive_filt_state)

        # convert to arrays because get_command does batched things that would be weird with objects
        # state_tensor = ABState.to_storm_format_np(filt_state, device=self.tensor_args['device'])


        if(WAIT):
            next_command, val, info, best_action = self.control_process.get_command_debug(t_step, filt_state, control_dt=control_dt)
        else:
            next_command, val, info, best_action = self.control_process.get_command(t_step, filt_state, control_dt=control_dt)

        
        print('control space in limb repo task', self.exp_params['control_space'])

        if self.exp_params['control_space'] == 'acc':
            cmd_des = self.state_filter.integrate_acc(next_command)
        elif self.exp_params['control_space'] == 'vel':
            cmd_des = self.state_filter.integrate_vel(next_command, curr_state.active.to_storm_format_np())

        cmd_des_active_obj = ABState.from_dict(cmd_des)
        cmd_des_obj = LRState(active=cmd_des_active_obj, passive=None)

        assert type(cmd_des_obj) == LRState

        return cmd_des_obj
    
    def get_velocity_command(self, t_step:int, current_state:LRState, control_dt:float, WAIT=False) -> LRState:
        '''Get the next velocity command to be executed by the robot.'''
        

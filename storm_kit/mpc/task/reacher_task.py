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
import torch
import yaml
import numpy as np

from ...util_file import get_mpc_configs_path as mpc_configs_path
from ...mpc.rollout.arm_reacher import ArmReacher
from ...mpc.control import MPPI
from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from .arm_task import ArmTask


class ReacherTask(ArmTask):
    """
    .. inheritance-diagram:: ReacherTask
       :parts: 1

    """
    def __init__(self, task_file='ur10.yml', robot_file='ur10_reacher.yml', world_file='collision_env.yml', tensor_args={'device':"cpu", 'dtype':torch.float32}):
        
        super().__init__(task_file=task_file, robot_file=robot_file,
                         world_file=world_file, tensor_args=tensor_args)

    def get_rollout_fn(self, **kwargs):
        rollout_fn = ArmReacher(**kwargs)
        return rollout_fn
    
    def get_command(self, t_step, curr_state, control_dt, WAIT=False):

        # predict forward from previous action and previous state:
        #self.state_filter.predict_internal_state(self.prev_qdd_des)

        if(self.state_filter.cmd_joint_state is None):
            curr_state['velocity'] *= 0.0
        filt_state = self.state_filter.filter_joint_state(curr_state)
        state_tensor = self._state_to_tensor(filt_state)

        if(WAIT):
            next_command, val, info, best_action = self.control_process.get_command_debug(t_step, state_tensor.numpy(), control_dt=control_dt)
        else:
            next_command, val, info, best_action = self.control_process.get_command(t_step, state_tensor.numpy(), control_dt=control_dt)

        qdd_des = next_command
        self.prev_qdd_des = qdd_des
        cmd_des = self.state_filter.integrate_acc(qdd_des)

        return cmd_des
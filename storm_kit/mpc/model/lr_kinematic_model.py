from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.autograd.profiler as profiler

from ...differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel
from urdfpy import URDF
from .urdf_kinematic_model import URDFKinematicModel
from .integration_utils import build_int_matrix, build_fd_matrix, tensor_step_acc, tensor_step_vel, tensor_step_pos, tensor_step_jerk
from ...util_limb_repo import LRState, ABState

class LimbRepoKinematicModel(URDFKinematicModel):
    def __init__(self, urdf_path, dt, batch_size=1000, horizon=5,
                 tensor_args={'device':'cpu','dtype':torch.float32}, ee_link_name='ee_link', link_names=[], dt_traj_params=None, vel_scale=0.5, control_space='vel'):
        super(LimbRepoKinematicModel, self).__init__(urdf_path, dt, batch_size, horizon, tensor_args, ee_link_name, link_names, dt_traj_params, vel_scale, control_space)

        self.robot_model = DifferentiableRobotModel(urdf_path, None, tensor_args=tensor_args)

        #self.robot_model.half()
        self.n_dofs = self.robot_model._n_dofs
        self.urdfpy_robot = URDF.load(urdf_path) #only for visualization
        
        print('self n dofs', self.n_dofs)

        self.d_state = 3 * self.n_dofs + 1
        self.d_action = self.n_dofs

    def get_next_state(self, curr_state: LRState, act:torch.Tensor, dt) -> LRState:
        """ Does a single step from the current state
        Args:
        curr_state: current state
        act: action
        dt: time to integrate
        Returns:
        next_state
        """
        # curr_state.to_torch(device=act.device)
        print('act', act)

        if(self.control_space == 'acc'):
            curr_state.active.acc = act 
            curr_state.active.vel = curr_state.active.vel + curr_state.active.acc * dt
            curr_state.active.pos = curr_state.active.pos + curr_state.active.vel * dt
        
        elif(self.control_space == 'vel'):
            curr_state.active.acc = act * 0.0
            curr_state.active.vel = act
            curr_state.active.pos = curr_state.active.pos + curr_state.active.vel * dt

        elif(self.control_space == 'pos'):
            curr_state.active.acc = act * 0.0
            curr_state.active.vel = act * 0.0
            curr_state.active.pos = act

        print('lr kinematic model curr_state', curr_state)
        
        return curr_state
    
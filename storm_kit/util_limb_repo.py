import numpy as np
import torch
import os
import storm_kit.util_file as futil
from scipy.spatial.transform import Rotation as R
from typing import Optional, List

OptArray = Optional[np.ndarray] 

class ABState:
    '''
    To represent an articulated body's state (robot or human).
    '''
    def __init__(self, pos=None, vel=None, acc=None, torque=None, base_pos=None, base_orn=None, body_name:str=None):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.acc = np.array(acc)
        self.torque = np.array(torque)
        self.base_pos = np.array(base_pos)
        self.base_orn = np.array(base_orn)
        self.body_name = body_name
    
    def to_dict(self) -> dict:
        return {'position': self.pos, 'velocity': self.vel, 'acceleration': self.acc, 
                'torque': self.torque, 'base_pos': self.base_pos, 'base_orn': self.base_orn}
    
    @classmethod
    def from_dict(self, joint_state:dict, body_name_in_dict_key:bool=False) -> 'ABState':
        if not body_name_in_dict_key:
            return ABState(pos=joint_state['position'], vel=joint_state['velocity'], acc=joint_state['acceleration'], 
                        torque=joint_state['torque'], base_pos=joint_state['base_pos'], base_orn=joint_state['base_orn'])
        else:
            return ABState(pos=joint_state[f'{self.body_name}_position'], vel=joint_state[f'{self.body_name}_velocity'],
                           acc=joint_state[f'{self.body_name}_acceleration'], torque=joint_state[f'{self.body_name}_torque'],
                           base_pos=joint_state[f'{self.body_name}_base_pos'], base_orn=joint_state[f'{self.body_name}_base_orn'])

    def to_storm_format(self, time=None, device=None) -> torch.tensor:
        '''Gives np array [pos, vel, acc, time] if time is not None, else [pos, vel, acc].'''
        for thing in [self.pos, self.vel, self.acc]:
            assert thing.all() != None, "Articulated body state not fully defined"

        self.to_torch(device=device)
        if time is None:
            print('in util self pos', torch.ravel(torch.stack([self.pos, self.vel, self.acc])))
            out = torch.ravel(torch.stack([self.pos, self.vel, self.acc]))
            self.to_np()
            return out
        
        out = torch.ravel(torch.stack([self.pos, self.vel, self.acc, torch.tensor([time])]))
        self.to_np()
        return out
    
    def to_storm_format_np(self, time=None) -> np.array:
        '''Gives np array [pos, vel, acc, time] if time is not None, else [pos, vel, acc].'''
        for thing in [self.pos, self.vel, self.acc]:
            print('thing', thing)
            assert thing.all() != None, "Articulated body state not fully defined"

        self.to_np()
        if time is None:
            print('in util self pos', np.ravel(np.stack([self.pos, self.vel, self.acc])))
            out = np.ravel(np.stack([self.pos, self.vel, self.acc]))
            self.to_np()
            return out
        
        out = np.ravel(np.stack([self.pos, self.vel, self.acc, np.tensor([time])]))
        return out

    def from_storm_format(self, storm_state, n_dof:int) -> 'ABState':
        '''Converts storm state to ABState object.'''
        pos = storm_state[:n_dof]
        vel = storm_state[n_dof:2*n_dof]
        acc = storm_state[2*n_dof:3*n_dof]
        return ABState(pos=pos, vel=vel, acc=acc)
    
    @property
    def n_dofs(self) -> int:
        self.to_np()
        valid_states = [arr for arr in [self.robot_pos, self.robot_vel, self.robot_acc, self.robot_torque] if arr.all() != np.array(None)]
        assert len(valid_states) > 0, "No valid articulated body states found"
        
        valid_states_dofs = [arr.shape[0] for arr in valid_states]

        assert len(set(valid_states_dofs)) == 1, "Articulated body state dimension mismatch"
        return valid_states_dofs[0].item()
        
    def to_np(self) -> None:
        for i in [self.pos, self.vel, self.acc, self.torque, self.base_pos, self.base_orn]:
            if i is not None and type(i) == torch.Tensor:
                i = i.cpu().numpy()

    def to_torch(self, device) -> None:
        self.pos = torch.tensor(self.pos, device=device)
        self.vel = torch.tensor(self.vel, device=device)
        self.acc = torch.tensor(self.acc, device=device)
        self.torque = torch.tensor(self.torque, device=device) if self.torque != None else None
        self.base_pos = torch.tensor(self.base_pos, device=device) if self.base_pos != None else None
        self.base_orn = torch.tensor(self.base_orn, device=device) if self.base_orn != None else None

    def __getitem__(self, key):
        if key == 'position':
            return self.pos
        elif key == 'velocity':
            return self.vel
        elif key == 'acceleration':
            return self.acc
        else:
            raise KeyError(f"Key '{key}' not found in ABState")

class LRState:
    '''
    To represent a limb repo state. Two articulated bodies are represented: one active and one passive.
    '''
    active: ABState
    passive: ABState
    def __init__(self, active:ABState=None, passive:ABState=None):
        self.active = active
        self.passive = passive
        return
    
    def to_storm_format_np(self, time=None) -> dict:
        '''Returns {'active': active, 'passive': passive}, where active and passive are in storm array format'''
        active = self.active.to_storm_format_np(time)
        passive = self.passive.to_storm_format_np(time)
        return {'active': active, 'passive': passive}
    
    def to_torch(self, device) -> None:
        self.active.to_torch(device)
        self.passive.to_torch(device)
        return


class LRUtils:
    def __init__(self, use4DoFhuman=True):
        self.robot_n_dofs = 6
        self.human_n_dofs = 4 if use4DoFhuman else 6
        self.robot_path = os.path.join(futil.get_root_path(), 'content/assets/urdf/franka6/panda.urdf')

        if self.human_n_dofs == 4:
            self.robot_init_config = np.array([-0.54193711, -1.07197495, -2.81591873, -1.6951869,   2.48184051, -1.43600207])
            self.human_init_config = np.array([-2.9077931,  -0.64935838, -1.09709132,  0.17183818])
            self.human_path = os.path.join(futil.get_root_path(), 'content/assets/urdf/human_arm/arm.urdf')
        elif self.human_n_dofs == 6:
            self.robot_init_config = np.array([0.94578431, -0.89487842, -1.67534487, -0.34826698, 1.73607292, 0.14233887])
            self.human_init_config = np.array([1.43252278, -0.81111486, -0.42373363, 0.49931369, -1.17420521, 0.37122887])
            self.human_path = os.path.join(futil.get_root_path(), 'content/assets/urdf/human_arm/arm_6dof_continuous.urdf')


        # got rid of third joint because we lock it
        self.robot_config_min = np.array([-2.90, -1.76, -3.07, -2.90, -0.02, -2.90]) # including joint 3: [-2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90]
        self.robot_config_max = np.array([2.90, 1.76, -0.07, 2.90, 3.75, 2.90]) # including joint 3: [2.90, 1.76, 2.90, -0.07, 2.90, 3.75, 2.90]
        self.robot_vel_min = 2 * np.ones((self.robot_n_dofs,))
        self.robot_vel_max = -2 * np.ones((self.robot_n_dofs,))
        self.robot_torque_max = 5 * np.ones((self.robot_n_dofs,))
        self.robot_torque_min = -5 * np.ones((self.robot_n_dofs,))

        self.human_vel_min = 2 * np.ones((self.human_n_dofs,))
        self.human_vel_max = -2 * np.ones((self.human_n_dofs,))

        self.grasp_orn_mat = np.array([[0, 0, -1], 
                                       [0, -1, 0], 
                                       [-1, 0, 0]])

        return
    
    def get_init_configs_dict(self):
        return {'initial_robot_position': self.robot_init_config, 'initial_human_position': self.human_init_config}
    
    def get_init_configs(self):
        return self.robot_init_config, self.human_init_config

    def get_urdf_paths(self):
        return self.robot_path, self.human_path
    
    def sample_robot_torques(self):
        return np.array([np.random.uniform(i, j) for i, j in zip(self.robot_torque_min, self.robot_torque_max)])
    
    def sample_robot_velocities(self):
        return np.array([np.random.uniform(i, j) for i, j in zip(self.robot_vel_min, self.robot_vel_max)])
    
    def sample_robot_config(self):
        return np.array([np.random.uniform(i, j) for i, j in zip(self.robot_config_min, self.robot_config_max)])
    
    def sample_human_velocities(self):
        return np.array([np.random.uniform(i, j) for i, j in zip(self.human_vel_min, self.human_vel_max)])
    
    def solve_human_grasp(self, robot_ee_pos, robot_ee_orn):
        '''
        robot_ee_pos: np.array([x, y, z])
        robot_ee_orn: np.array([x, y, z, w])
        output: arm_goal_pos, arm_goal_rot
        '''
        robot_ee_orn = R.from_quat(robot_ee_orn).as_matrix()

        robot_pose = np.eye(4)
        robot_pose[:3, :3] = robot_ee_orn
        robot_pose[:3, 3] = robot_ee_pos

        # Transformation between robot ee and human ee
        transformation = np.eye(4)
        transformation[:3, :3] = self.grasp_orn_mat
        transformation[:3, 3] = np.array([0, 0, 0])

        # Get desired human_ee pose
        arm_goal_pose = robot_pose @ transformation
        arm_goal_pos = arm_goal_pose[:3, 3]
        arm_goal_rot = R.as_quat(R.from_matrix(arm_goal_pose[:3, :3]))

        return arm_goal_pos, arm_goal_rot
    
    def valid_grasp(self, robot_ee_pos, robot_ee_vel, robot_ee_orn, human_ee_pos, human_ee_vel, human_ee_orn, debug=False):

        position_check = np.allclose(robot_ee_pos, human_ee_pos, atol=0.01)
        velocity_check = np.allclose(robot_ee_vel, human_ee_vel, atol=0.1)
        orientaion_check = np.allclose((np.linalg.inv(robot_ee_orn) @ human_ee_orn), self.grasp_orn_mat, atol=0.01)

        if debug:
            print(f"robot and human ee positions don't match\nrobot ee pos: {robot_ee_pos}\nhuman ee pos: {human_ee_pos}") if not position_check else 0
            print(f"robot and human ee velocities don't match\nrobot ee vel: {robot_ee_vel}\nhuman ee vel: {human_ee_vel}") if not velocity_check else 0
            print(f"robot and human ee orientations don't match\nrobot ee orn: {robot_ee_orn}\nhuman ee orn: {human_ee_orn}") if not orientaion_check else 0

        return position_check and velocity_check and orientaion_check

    def get_model_weights_path(self, model):
        return os.path.join(futil.get_root_path(), f'storm_kit/mpc/model/nn/weights/{model}')
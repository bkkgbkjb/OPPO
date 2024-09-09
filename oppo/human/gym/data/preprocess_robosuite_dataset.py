import gym
import numpy as np
import h5py
import collections
import pickle
from tqdm import tqdm
import json
import d4rl

datasets = []

for env_name in ['Lift', 'Can']:
    for dataset_type in ['mh', 'ph']:
        f = h5py.File(f'../robosuite_dataset/{env_name.lower()}/{dataset_type}/low_dim.hdf5', 'r')

        # N = dataset['rewards'].shape[0]
        demos = list(f['data'].keys())
        N = len(demos)
        # done_ = []
        # traj_idx_ = []
        # seg_idx_ = []

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        # if 'timeouts' in dataset:
        #     use_timeouts = True

        episode_step = 0
        paths = []
        # obs_keys = kwargs.get("obs_key", ["object", "robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"])
        obs_keys = ["object", "robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"]

        for ep in tqdm(demos, desc="load robosuite demonstrations"):
            ep_grp = f[f"data/{ep}"]
            traj_len = ep_grp["actions"].shape[0]
            obs_ = []
            action_ = []
            reward_ = []
            for i in range(traj_len):
                total_obs = ep_grp["obs"]
                obs = np.concatenate([total_obs[key][i].tolist() for key in obs_keys], axis=0)
                # new_obs = np.concatenate([total_obs[key][i + 1].tolist() for key in obs_keys], axis=0)
                action = ep_grp["actions"][i]
                reward = ep_grp["rewards"][i]
                # done_bool = bool(ep_grp["dones"][i])

                obs_.append(obs)
                # next_obs_.append(new_obs)
                action_.append(action)
                reward_.append(reward)
                # done_.append(done_bool)
                # traj_idx_.append(int(ep[5:]))
                # seg_idx_.append(i)
            paths.append({'observations': np.array(obs_), 'actions': np.array(action_), 'rewards': np.array(reward_)})

        f.close()

        with open(f'{env_name}_{dataset_type}.pkl', 'wb') as f:
          pickle.dump(paths, f)
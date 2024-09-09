import numpy as np
import json
import h5py
from tqdm import tqdm
import collections
import robomimic.utils.env_utils as EnvUtils
from robosuite.wrappers import GymWrapper
import wrappers

def qlearning_robosuite_dataset(dataset_path, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    f = h5py.File(dataset_path, 'r')

    # N = dataset['rewards'].shape[0]
    demos = list(f['data'].keys())
    N = len(demos)
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    traj_idx_ = []
    seg_idx_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    # if 'timeouts' in dataset:
    #     use_timeouts = True

    episode_step = 0
    obs_keys = kwargs.get("obs_key", ["object", "robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"])
    # for ep in tqdm(demos, desc="load robosuite demonstrations"):
    #     ep_grp = f[f"data/{ep}"]
    #     traj_len = ep_grp["actions"].shape[0]
    #     for i in range(traj_len - 1):
    #         total_obs = ep_grp["obs"]
    #         obs = np.concatenate([total_obs[key][i].tolist() for key in obs_keys], axis=0)
    #         new_obs = np.concatenate([total_obs[key][i + 1].tolist() for key in obs_keys], axis=0)
    #         action = ep_grp["actions"][i]
    #         reward = ep_grp["rewards"][i]
    #         done_bool = bool(ep_grp["dones"][i])

    #         obs_.append(obs)
    #         next_obs_.append(new_obs)
    #         action_.append(action)
    #         reward_.append(reward)
    #         done_.append(done_bool)
    #         traj_idx_.append(int(ep[5:]))
    #         seg_idx_.append(i)

    return {
        # 'observations': np.array(obs_),
        # 'actions': np.array(action_),
        # 'next_observations': np.array(next_obs_),
        # 'rewards': np.array(reward_),
        # 'terminals': np.array(done_),
        'env_meta': json.loads(f["data"].attrs["env_args"]),
        # 'traj_indices': np.array(traj_idx_),
        # 'seg_indices': np.array(seg_idx_),
    }


def make_robosuite_env_and_dataset(env_name: str,
                         seed: int,
                         dataset_path: str,
                         max_episode_steps: int = 500):


    ds = qlearning_robosuite_dataset(dataset_path)
    # dataset = RelabeledDataset(ds['observations'], ds['actions'], ds['rewards'], ds['terminals'], ds['next_observations'])

    ds['env_meta']['env_kwargs']['horizon'] = max_episode_steps
    env = EnvUtils.create_env_from_metadata(
        env_meta=ds['env_meta'],
        render=False,            # no on-screen rendering
        render_offscreen=False,   # off-screen rendering to support rendering video frames
    ).env
    env.ignore_done = False

    env._max_episode_steps = env.horizon
    env = GymWrapper(env)
    env = wrappers.RobosuiteWrapper(env)
    env = wrappers.EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env, None

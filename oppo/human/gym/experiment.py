from datetime import datetime
import setup
from setup import seed
import gym
import numpy as np
import torch
import wandb
import json
from args import args

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, evaluate_episode_phi
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.encoder_transformer import EncoderTransformer
from decision_transformer.models.preference_decision_transformer import PreferenceDecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.pdt_trainer import PDTTrainer
from reporter import get_reporter

from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE
import os
import d4rl
from utils import make_robosuite_env_and_dataset
from tqdm import tqdm


def discount_cumsum(x, gamma):
    discount_cumsum = torch.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    seed(variant['seed'])
    exp_name = json.dumps(variant, indent=4, sort_keys=True)
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    dataset_path = os.path.join(variant['robosuite_dataset_path'], variant['env'].lower(), variant['dataset'], "low_dim.hdf5")

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif env_name =='antmaze-medium':
        env = gym.make(f'antmaze-medium-{dataset}-v2')
        max_ep_len=1001
        scale=1.
        env_targets=[1.]
    elif env_name == 'antmaze-large':
        env = gym.make(f'antmaze-large-{dataset}-v2')
        max_ep_len=1001
        env_targets=[1.]
        scale=1.
    elif env_name in ['Lift', 'Can']:
        env, _ = make_robosuite_env_and_dataset(env_name, variant['seed'], dataset_path, max_episode_steps=500)
        max_ep_len = 500
        if env_name=='Can' and dataset == 'mh':
            max_ep_len = 1050
        env_targets = [1.]
        scale=1.
    else:
        raise NotImplementedError
    env.seed(variant['seed'])

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dir_path = variant.get('dirpath', '.')
    dataset_path = f'{dir_path}/data/{env_name}_{dataset}.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    state_mean_tensor = torch.as_tensor(state_mean, dtype=torch.float32, device=device)
    state_std_tensor = torch.as_tensor(state_std, dtype=torch.float32, device=device)

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    z_dim = variant['z_dim']
    print(f'z_dim is: {z_dim}')
    print(f"reward foresee is: {variant['foresee']}")

    expert_score = 1.
    random_score = 0.
    print(f"max score is: {expert_score}, min score is {random_score}")

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])


    trajectories = [
        {
            "observations": torch.tensor(
                t["observations"], dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(t["actions"], dtype=torch.float32, device=device),
            "rewards": torch.tensor(t["rewards"], dtype=torch.float32, device=device),
            # "terminals": torch.tensor(t["terminals"], dtype=torch.bool, device=device),
        }
        for t in trajectories
    ]


    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )

        # s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        s, a, timesteps, mask = (
            torch.empty(
                (batch_size, max_len, state_dim), dtype=torch.float32, device=device
            ),
            torch.empty(
                (batch_size, max_len, act_dim), dtype=torch.float32, device=device
            ),
            torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype=torch.long,
                device=device,
            ),
            torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype=torch.float32,
                device=device,
            ),
        )

        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            tlen = traj["observations"][si : si + max_len].shape[0]  # .reshape(1, -1, state_dim)
            s[i] = torch.cat(
                (
                    torch.zeros(
                        (max_len - tlen, state_dim),
                        device=device,
                        dtype=torch.float32,
                    ),
                    traj["observations"][si : si + max_len],
                )
            )
            s[i] = (s[i] - state_mean_tensor) / state_std_tensor
            a[i] = torch.cat(
                (
                    torch.ones(
                        (max_len - tlen, act_dim), device=device, dtype=torch.float32
                    )
                    * -10.0,
                    traj["actions"][si : si + max_len],
                )
            )
            timesteps[i] = torch.concat(
                (
                    torch.zeros((max_len-tlen, ),device=device, dtype=torch.long), 
                    torch.arange(si, si + tlen, device=device, dtype=torch.long)
                )
            )
            mask[i] = torch.concat(
                (
                    torch.zeros((max_len - tlen,), dtype=torch.float32, device=device),
                    torch.ones((tlen,), device=device, dtype=torch.float32),
                )
            )
            assert not (timesteps[i] >= max_ep_len).any()


        return s, a, None, None, None, timesteps, mask

    # deal with human_labeled data
    datadir = f'./data/dataset/human_label/{env_name}_{dataset}/data.pkl'
    with open(datadir, 'rb') as f:
        ds = pickle.load(f)
    s1 = torch.as_tensor(ds['observations'], dtype=torch.float32, device=device)
    s1 = (s1 - torch.as_tensor(state_mean, dtype=torch.float32, device=device)) / torch.as_tensor(state_std, dtype=torch.float32, device=device)
    s2 = torch.as_tensor(ds['observations_2'], dtype=torch.float32, device=device)
    s2 = (s2 - torch.as_tensor(state_mean, dtype=torch.float32, device=device)) / torch.as_tensor(state_std, dtype=torch.float32, device=device)
    a1 = torch.as_tensor(ds['actions'], dtype=torch.float32, device=device)
    a2 = torch.as_tensor(ds['actions_2'], dtype=torch.float32, device=device)

    t1 = torch.as_tensor(ds['fixed_timestep_1'], dtype=torch.long, device=device)
    t2 = torch.as_tensor(ds['fixed_timestep_2'], dtype=torch.long, device=device)

    pref = torch.as_tensor(ds['labels'], dtype=torch.float32, device=device)
    # s1 = s1[pref[:, 0] != 0.5]
    # s2 = s2[pref[:, 0] != 0.5]
    # a1 = a1[pref[:, 0] != 0.5]
    # a2 = a2[pref[:, 0] != 0.5]
    # t1 = t1[pref[:, 0] != 0.5]
    # t2 = t2[pref[:, 0] != 0.5]
    # pref = pref[pref[:, 0] != 0.5]


    def get_batch2(batch_size=256, max_len=100, K=20):
        nonlocal s1, s2, a1, a2, t1, t2, pref
        sample_idx = torch.randint(0, s1.size(0), size=(batch_size,))
        # sample_seq_start_point1 = torch.randint(0, s1.size(1) - K + 1, size=(batch_size, ))
        # sample_seq_start_point2 = torch.randint(0, s1.size(1) - K + 1, size=(batch_size, ))
        # sampling_idx1 = torch.arange(K).view(1, K) + sample_seq_start_point1.view(batch_size, 1)
        # sampling_idx1 = sampling_idx1.to(device=device).unsqueeze(-1)
        # sampling_idx2 = torch.arange(K).view(1, K) + sample_seq_start_point2.view(batch_size, 1)
        # sampling_idx2 = sampling_idx2.to(device=device).unsqueeze(-1)

        s1_seq = s1[sample_idx]
        s2_seq = s2[sample_idx]
        a1_seq = a1[sample_idx]
        a2_seq = a2[sample_idx]
        t1_seq = t1[sample_idx]
        t2_seq = t2[sample_idx]
        pref_seq = pref[sample_idx]

        # s1_seq = torch.gather(s1[sample_idx], dim=1, index=sampling_idx1.repeat(1, 1, s1.size(-1)))
        # s2_seq = torch.gather(s2[sample_idx], dim=1, index=sampling_idx2.repeat(1, 1, s2.size(-1)))
        # a1_seq = torch.gather(a1[sample_idx], dim=1, index=sampling_idx1.repeat(1, 1, a1.size(-1)))
        # a2_seq = torch.gather(a2[sample_idx], dim=1, index=sampling_idx2.repeat(1, 1, a2.size(-1)))
        # t1_seq = torch.gather(t1[sample_idx], dim=1, index=sampling_idx1.squeeze(-1))
        # t2_seq = torch.gather(t2[sample_idx], dim=1, index=sampling_idx2.squeeze(-1))
        # pref_seq = pref[sample_idx]

        # s1_seq = torch.empty((batch_size, K, s1.size(-1)), dtype=torch.float32, device=device)
        # s2_seq = torch.empty((batch_size, K, s2.size(-1)), dtype=torch.float32, device= device)
        # a1_seq = torch.empty((batch_size, K, a1.size(-1)), dtype=torch.float32, device= device)
        # a2_seq = torch.empty((batch_size, K, a2.size(-1)), dtype=torch.float32, device= device)
        # t1_seq = torch.empty((batch_size, K ), dtype=torch.long, device= device)
        # t2_seq = torch.empty((batch_size, K ), dtype=torch.long, device= device)
        # pref_seq = torch.empty((batch_size, 2), dtype=torch.float32, device=device)

        # for i in range(batch_size):
        #     row_idx = sample_idx[i]
        #     sp = sample_seq_start_point[i]
        #     s1_seq[i] = s1[row_idx, sp:sp+K]
        #     s2_seq[i] = s2[row_idx, sp:sp+K]
        #     a1_seq[i] = a1[row_idx, sp:sp+K]
        #     a2_seq[i] = a2[row_idx, sp:sp+K]
        #     t1_seq[i] = t1[row_idx, sp:sp+K]
        #     t2_seq[i] = t2[row_idx, sp:sp+K]
        #     pref_seq[i] = pref[row_idx]


        # return (s1[sample_idx], s2[sample_idx]), (a1[sample_idx], a2[sample_idx]), (t1[sample_idx], t2[sample_idx]), pref[sample_idx]
        return (s1_seq, s2_seq), (a1_seq, a2_seq), (t1_seq, t2_seq), pref_seq

    def eval_episodes(target_rew):
        largest_norm_return_mean = -1e7
        def fn(model):
            nonlocal largest_norm_return_mean
            returns, norm_returns, lengths = [], [], []
            for _ in tqdm(range(num_eval_episodes)):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=500,
                            scale=scale,
                            target_return=target_rew/scale if not variant['subepisode'] else target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            eval_no_change=variant['eval_no_change']
                        )
                    elif model_type == 'pdt':
                        ret, length = evaluate_episode_phi(
                            env,
                            state_dim,
                            act_dim,
                            model[0],
                            max_ep_len=500,
                            scale=scale,
                            phi=(model[1]).unsqueeze(0),
                            # phi=(model[1] / torch.linalg.vector_norm(model[1])).unsqueeze(0),
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=500,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                norm_ret = (ret - random_score) / (expert_score - random_score) * 100
                returns.append(ret)
                norm_returns.append(norm_ret)
                lengths.append(length)
            if variant.get("in_tune", False):
                from ray import tune
                # tune.report(**{"eval/return": np.mean(norm_returns)})
                mean_norm_return = np.mean(norm_returns)
                if mean_norm_return > largest_norm_return_mean:
                    largest_norm_return_mean = mean_norm_return
                tune.report(**{"eval/return": largest_norm_return_mean})

            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_norm_return_mean': np.mean(norm_returns),
                f'target_{target_rew}_norm_return_std': np.std(norm_returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    elif model_type == 'pdt':
        model = PreferenceDecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            phi_size=z_dim,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'pdt':
        en_model = EncoderTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=variant['embed_dim'],
            output_size=z_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            num_hidden_layers=3,
            num_attention_heads=2,
            intermediate_size=4*variant['embed_dim'],
            max_position_embeddings=1024,
            hidden_act=variant['activation_function'],
            hidden_dropout_prob=variant['dropout'],
            attention_probs_dropout_prob=variant['dropout'],
        )
        en_model = en_model.to(device=device)
        et_optimizer = torch.optim.AdamW(
            en_model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

        w = torch.nn.parameter.Parameter(torch.empty(z_dim, requires_grad=True, device=device) * 2)
        torch.nn.init.normal_(w)
        w_optimizer = torch.optim.AdamW(
            [w],
            lr=variant["w_lr"],
            weight_decay=1e-4
        )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'pdt':
        trainer = PDTTrainer(
            en_model=en_model,
            de_model=model,
            optimizer=optimizer,
            et_optimizer=et_optimizer,
            w=w,
            w_optimizer=w_optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            get_batch2=get_batch2,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((s_hat - s)**2) + torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            device=device,
            pref_loss_ratio=variant["pref_loss_ratio"],
            phi_norm_loss_ratio=variant["phi_norm_loss_ratio"]
        )

    name = f"new_phi_norm_{variant['env']}-{variant['dataset']}-{variant['model_type']}-{variant['seed']}-{datetime.now().strftime('%m-%d-%H-%M-%S-%f')}"
    if log_to_wandb:
        # wandb.init(
        #     name=exp_prefix,
        #     group=group_name,
        #     project='decision-transformer',
        #     config=variant
        # )

        reporter = get_reporter(name, exp_name)
        # wandb.watch(model)  # wandb has some bug

    max_perf = -1e7
    last_saved_idx = -1
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        norm_return_mean = outputs[f'evaluation/target_{env_targets[0]}_norm_return_mean']
        # if (variant['force_save_model'] or variant.get("in_tune", False)) and iter >= 30 and iter % 2 == 0:
        if (variant['force_save_model'] or variant.get("in_tune", False)) and iter >= 20 and (norm_return_mean > max_perf or (iter - last_saved_idx >= 20)):
            folder = f"./model_weight_{name}"
            if not os.path.exists(folder):
                os.mkdir(folder)
            torch.save((en_model.state_dict(), w, model.state_dict()), f"./{folder}/params_{iter}.pt")
            max_perf = norm_return_mean
            last_saved_idx = iter
        if log_to_wandb:
            # wandb.log(outputs)
            reporter(outputs)


if __name__ == '__main__':

    experiment('gym-experiment', variant=vars(args))

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


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
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
    else:
        raise NotImplementedError
    env.seed(variant['seed'])

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dir_path = variant.get('dirpath', '.')
    dataset_path = f'{dir_path}/data/{env_name}-{dataset}-v2.pkl'
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

    expert_score = REF_MAX_SCORE[f"{variant['env']}-{variant['dataset']}-v2"]
    random_score = REF_MIN_SCORE[f"{variant['env']}-{variant['dataset']}-v2"]
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

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset

            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            assert not (timesteps[-1] >= max_ep_len).any()
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

            if variant['train_no_change']:
                if not variant['subepisode']:
                    rtg.append(discount_cumsum(traj['rewards'][0:], gamma=1.)[0].reshape(1, 1, 1).repeat(s[-1].shape[1] + 1, axis=1))
                else:
                    rtg.append(discount_cumsum(traj['rewards'][si:si+variant['foresee']], gamma=1.)[0].reshape(1, 1, 1).repeat(s[-1].shape[1] + 1, axis=1))
                    # rtg.append((np.sum(traj['rewards'][si:si + s[-1].shape[1]])).reshape(1,1,1).repeat(s[-1].shape[1] + 1, axis=1))

            else:
                rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            # print(f"rtg is: {rtg[-1]}")

            if rtg[-1].shape[1] <= s[-1].shape[1]:
                assert False
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        largest_norm_return_mean = -1e7
        def fn(model):
            nonlocal largest_norm_return_mean
            returns, norm_returns, lengths = [], [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
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
                            max_ep_len=max_ep_len,
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
                            max_ep_len=max_ep_len,
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

        w = (torch.randn(z_dim) * 2).to(device=device)
        w.requires_grad = True
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
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            device=device,
            pref_loss_ratio=variant["pref_loss_ratio"],
            phi_norm_loss_ratio=variant["phi_norm_loss_ratio"]
        )

    name = f"{variant['env']}-{variant['dataset']}-{variant['model_type']}-{variant['seed']}-{datetime.now().strftime('%f')}"
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
        if (variant['force_save_model'] or variant.get("in_tune", False)) and iter >= 10 and (norm_return_mean > max_perf or (iter - last_saved_idx > 20)):
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

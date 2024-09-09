import numpy as np
import torch
from tqdm import tqdm
from torch import nn

import time

from decision_transformer.training.trainer import Trainer
device = "cuda" if torch.cuda.is_available() else "cpu"


class PDTTrainer(Trainer):
    def __init__(
        self,
        en_model,
        de_model,
        optimizer,
        et_optimizer,
        w,
        w_optimizer,
        batch_size,
        get_batch,
        get_batch2,
        loss_fn,
        device,
        scheduler=None,
        eval_fns=None,
        pref_loss_ratio=0.1,
        phi_norm_loss_ratio=0.1
    ):
        super().__init__(None, optimizer, batch_size, get_batch, loss_fn, scheduler, eval_fns)
        self.en_model = en_model
        self.de_model = de_model
        self.et_optimizer = et_optimizer
        self.w = w
        self.w_optimizer = w_optimizer

        self.device = device

        self.regress_loss = nn.MSELoss()
        self.phi_loss = nn.ReLU()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='sum')
        self.triplet_loss2 = nn.TripletMarginLoss(margin=0, p=2, reduction='sum')

        self.pref_loss_ratio = pref_loss_ratio
        self.phi_norm_loss_ratio = phi_norm_loss_ratio
        self.get_batch2 = get_batch2
        self.total_data = 0
        self.used_data = 0

        self.cont = 0

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        # train_losses = []
        regress_losses = []
        recon_losses = []
        phi_norm_losses = []
        phi_recon_losses = []
        phi_pref_losses = []

        logs = dict()

        train_start = time.time()

        self.en_model.train()
        self.de_model.train()
        for i in tqdm(range(num_steps)):
            regress_loss, recon_loss, phi_norm_loss, phi_recon_loss, phi_pref_loss = self.train_step()
            regress_losses.append(regress_loss)
            recon_losses.append(recon_loss)
            phi_norm_losses.append(phi_norm_loss)
            phi_recon_losses.append(phi_recon_loss)
            phi_pref_losses.append(phi_pref_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.en_model.eval()
        self.de_model.eval()
        for eval_fn in self.eval_fns:
            # print(self.w)
            outputs = eval_fn((self.de_model,self.w.detach()))
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        # logs['training/train_loss_mean'] = np.mean(train_losses)
        # logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/pref_loss_mean'] = np.mean(regress_losses)
        logs['training/pref_loss_std'] = np.std(regress_losses)
        logs['training/train_loss_mean'] = np.mean(recon_losses)
        logs['training/train_loss_std'] = np.std(recon_losses)
        logs['training/phi_recon_loss_mean'] = np.mean(phi_recon_losses)
        logs['training/phi_recon_loss_std'] = np.std(phi_recon_losses)
        logs['training/phi_norm_loss_mean'] = np.mean(phi_norm_losses)
        logs['training/phi_norm_loss_std'] = np.std(phi_norm_losses)
        logs['training/phi_pref_loss_mean'] = np.mean(phi_pref_losses)
        logs['training/phi_pref_loss_std'] = np.std(phi_pref_losses)
        logs['training/used_data_perc'] = self.used_data / self.total_data * 100
        print(self.w)
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        recon_loss = torch.tensor(0.0, device=device)
        if self.cont % 2 == 0:
            self.cont = 0
            states_1, actions_1, rewards_1, dones_1, rtg_1, timesteps_1, attention_mask_1 = self.get_batch(self.batch_size)
            # states_2, actions_2, rewards_2, dones_2, rtg_2, timesteps_2, attention_mask_2 = self.get_batch(self.batch_size)

            action_target_1 = torch.clone(actions_1)
            # action_target_2 = torch.clone(actions_2)
            state_target_1 = torch.clone(states_1)
            # state_target_2 = torch.clone(states_2)

            phi_1 = self.en_model.forward(states_1, actions_1, timesteps_1, attention_mask_1)
            # phi_2 = self.en_model.forward(states_2, actions_2, timesteps_2, attention_mask_2)

            phi_1 = phi_1.expand(states_1.shape[1], -1, -1).permute(1, 0, 2)
            states_1 = states_1 + torch.randn_like(states_1) * 0.1
            # actions_1 = actions_1 * 0
            state_preds_1, action_preds_1, reward_preds_1 = self.de_model.forward(
                states_1, actions_1, None, phi_1, timesteps_1, attention_mask=attention_mask_1,
            )
            # phi_2 = phi_2.expand(states_2.shape[1], -1, -1).permute(1, 0, 2)
            # states_2 = states_2 + torch.randn_like(states_2) * 0.1
            # # actions_2 = actions_2 * 0
            # state_preds_2, action_preds_2, reward_preds_2 = self.de_model.forward(
            #     states_2, actions_2, None, phi_2, timesteps_2, attention_mask=attention_mask_2,
            # )

            act_dim = action_preds_1.shape[2]
            action_preds_1 = action_preds_1.reshape(-1, act_dim)[attention_mask_1.reshape(-1) > 0]
            action_target_1 = action_target_1.reshape(-1, act_dim)[attention_mask_1.reshape(-1) > 0]

            # act_dim = action_preds_2.shape[2]
            # action_preds_2 = action_preds_2.reshape(-1, act_dim)[attention_mask_2.reshape(-1) > 0]
            # action_target_2 = action_target_2.reshape(-1, act_dim)[attention_mask_2.reshape(-1) > 0]

            recon_loss = (self.loss_fn(
                state_preds_1, action_preds_1, None,
                state_target_1, action_target_1, None,))
            # )
            # + self.loss_fn(
            #     state_preds_2, action_preds_2, None,
            #     state_target_2, action_target_2, None,
            # ))

            self.optimizer.zero_grad()
            self.et_optimizer.zero_grad()
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.de_model.parameters(), .25)
            torch.nn.utils.clip_grad_norm_(self.en_model.parameters(), .25)
            self.optimizer.step()
            self.et_optimizer.step()
        self.cont += 1

        (states_1, states_2), (actions_1, actions_2), (timesteps_1, timesteps_2), prefs = self.get_batch2(self.batch_size)

        _bs, _seq_len = states_1.size(0), states_1.size(1)
        attention_mask_1 = torch.ones((_bs, _seq_len),device=device, dtype=torch.long)
        attention_mask_2 = torch.ones((_bs, _seq_len),device=device, dtype=torch.long)

        action_target_1 = torch.clone(actions_1)
        action_target_2 = torch.clone(actions_2)
        state_target_1 = torch.clone(states_1)
        state_target_2 = torch.clone(states_2)

        lb = prefs[:, 0] == 1.0
        rb = prefs[:, 1] == 1.0
        eb = prefs[:, 0] == 0.5

        # pre = (rtg_1[:,0,0]>rtg_2[:,0,0]).to(dtype=torch.float32)
        # margin = 0

        phi_1 = self.en_model.forward(states_1, actions_1, timesteps_1, attention_mask_1)
        phi_2 = self.en_model.forward(states_2, actions_2, timesteps_2, attention_mask_2)
        # if self.phi_norm == "hard":
        #     phi = _phi / torch.linalg.vector_norm(_phi.detach(), dim=1).unsqueeze(1)
        # else:

        # phi = _phi
        phi_norm_loss = (self.phi_loss(torch.norm(phi_1, dim=1) - torch.ones(self.batch_size).to(self.device)).mean()
                    + self.phi_loss(torch.norm(phi_2, dim=1) - torch.ones(self.batch_size).to(self.device)).mean())
        # phi_norm_loss = torch.norm(phi_1, dim=1).sum() + torch.norm(phi_2, dim=1).sum()

        positive = torch.cat((phi_1[lb], phi_2[rb]), 0)
        negative = torch.cat((phi_2[lb], phi_1[rb]), 0)
        anchor = self.w.expand(positive.shape[0], -1).detach()
        positive2 = phi_1[eb]
        negative2 = phi_2[eb]
        anchor2 = self.w.expand(positive2.shape[0], -1).detach()
        phi_pref_loss = (self.triplet_loss(anchor, positive, negative)
                         + self.triplet_loss2(anchor2, positive2, negative2)) / self.batch_size

        self.total_data = self.total_data + self.batch_size
        self.used_data = self.used_data + positive.shape[0]

        # pred_returns = torch.inner(phi, self.w.detach())
        # returns_loss = -(
        #     pred_returns
        #     / torch.linalg.vector_norm(self.w)
        #     / torch.linalg.vector_norm(phi)
        # ).mean()
        # regress_loss = self.regress_loss(pred_returns, rtg[:,-1].unsqueeze(1))

        for param in self.de_model.parameters():
            param.requires_grad = False

        phi_1 = phi_1.expand(states_1.shape[1], -1, -1).permute(1, 0, 2)
        state_preds_1, action_preds_1, reward_preds_1 = self.de_model.forward(
            states_1, actions_1, None, phi_1, timesteps_1, attention_mask=attention_mask_1,
        )
        phi_2 = phi_2.expand(states_2.shape[1], -1, -1).permute(1, 0, 2)
        state_preds_2, action_preds_2, reward_preds_2 = self.de_model.forward(
            states_2, actions_2, None, phi_2, timesteps_2, attention_mask=attention_mask_2,
        )

        act_dim = action_preds_1.shape[2]
        action_preds_1 = action_preds_1.reshape(-1, act_dim)[attention_mask_1.reshape(-1) > 0]
        action_target_1 = action_target_1.reshape(-1, act_dim)[attention_mask_1.reshape(-1) > 0]

        act_dim = action_preds_2.shape[2]
        action_preds_2 = action_preds_2.reshape(-1, act_dim)[attention_mask_2.reshape(-1) > 0]
        action_target_2 = action_target_2.reshape(-1, act_dim)[attention_mask_2.reshape(-1) > 0]

        phi_recon_loss = (self.loss_fn(
            state_preds_1, action_preds_1, None,
            state_target_1, action_target_1, None,
        )
        + self.loss_fn(
            state_preds_2, action_preds_2, None,
            state_target_2, action_target_2, None,
        ))

        for param in self.de_model.parameters():
            param.requires_grad = True

        en_model_loss = (phi_recon_loss
                        + self.pref_loss_ratio * phi_pref_loss
                        + self.phi_norm_loss_ratio * phi_norm_loss)
                        # + 10 * returns_loss
                        # + (phi_norm_loss if self.phi_norm == "soft" else 0)

        self.et_optimizer.zero_grad()
        en_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.en_model.parameters(), .25)
        self.et_optimizer.step()

        # for i in range(5):
        # phi = self.en_model.forward(states, actions, timesteps, attention_mask).detach()
        # pred_returns = torch.inner(phi, self.w)
        # regress_loss = self.regress_loss(pred_returns, rtg[:,-1])
        phi_1 = self.en_model.forward(states_1, actions_1, timesteps_1, attention_mask_1).detach()
        phi_2 = self.en_model.forward(states_2, actions_2, timesteps_2, attention_mask_2).detach()
        positive = torch.cat((phi_1[lb] , phi_2[rb]),0)
        negative = torch.cat((phi_2[lb] , phi_1[rb]),0)
        anchor = self.w.expand(positive.shape[0], -1)
        positive2 = phi_1[eb]
        negative2 = phi_2[eb]
        anchor2 = self.w.expand(positive2.shape[0], -1)
        pref_loss = (self.triplet_loss(anchor, positive, negative)
                        + self.triplet_loss2(anchor2, positive2, negative2)) / self.batch_size
        self.w_optimizer.zero_grad()
        pref_loss.backward()
        self.w_optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds_1-action_target_1)**2).detach().cpu().item()

        return (pref_loss.detach().cpu().item(),
                recon_loss.detach().cpu().item(),
                phi_norm_loss.detach().cpu().item(),
                phi_recon_loss.detach().cpu(),
                phi_pref_loss.detach().cpu())

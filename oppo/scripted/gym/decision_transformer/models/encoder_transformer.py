import numpy as np
import torch
import torch.nn as nn


# from utils.nets import NeuralNetworks

import transformers
from args import args

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.bert import BertModel
# from decision_transformer.models.trajectory_gpt2 import GPT2Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EncoderTransformer(TrajectoryModel):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        output_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.BertConfig(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            hidden_size=hidden_size,
            **kwargs
        )
        self.output_size = output_size

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = BertModel(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        # self.to_mean = nn.Linear(2 * 8 * self.hidden_size, self.hidden_size)
        # self.to_std = nn.Linear(2 * 8 * self.hidden_size, self.hidden_size)
        self.to_phi = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self,
        states,
        actions,
        timesteps,
        attention_mask=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(
                DEVICE
            )

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.hidden_size)
        )

        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(
            batch_size,
            seq_length,
            2,
            self.hidden_size,
        ).permute(0, 2, 1, 3)

        x = x.sum(dim=2).sum(dim=1)

        return self.to_phi(x)

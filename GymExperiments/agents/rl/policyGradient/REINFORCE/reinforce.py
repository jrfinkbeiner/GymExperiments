from typing import Union, Optional, List, Callable
import numpy as np

import torch
import torch.nn as nn

from GymExperiments.agents.rl.rlbase import DiscreteRL, ContinousRL
from GymExperiments.util import torch_util
from GymExperiments.util import rl_util
from GymExperiments.trainers.vae.train_vae import vae_loss_fn
from GymExperiments.architectures.vae import reparameterize


class DiscreteReinforce(DiscreteRL):

    def train_step(self, *args, **kwargs):
        return self.train_on_session(*args, **kwargs)

    def train_on_session(self, states: List, actions: List, rewards: List, expl: float = 1.0, repre: float = 1.0):
        """
        Takes a sequence of states, actions and rewards produced by generate_session.
        Updates agent's weights by following the policy gradient above.
        """

        self.optimizer.zero_grad()

        # cast everything into torch tensors
        states = torch.tensor(states[:-1], dtype=torch.float32)
        if self.from_pixel:
            states = states.permute(0,3,1,2)/255
        actions = self.convert_from_action_space(torch.tensor(actions, dtype=torch.float32))
        cumulative_returns = np.array(rl_util.get_cumulative_rewards(rewards, self.gamma))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32).unsqueeze(dim=1)

        # predict logits, probas and log-probas using an agent.
        out = self.model(states)

        if isinstance(out, torch.Tensor):
            logits = out
        else: 
            logits = out[0]
        
        log_probs = nn.functional.log_softmax(logits, -1)
        log_probs_for_actions = torch.sum(log_probs * torch_util.to_one_hot(actions, log_probs.size()[-1]), dim=1)
        
        # # print(actions)
        # # print(log_probs)
        # # print(distr)
        # print("mean", torch.min(out[0]).item(), torch.mean(out[0]).item(), torch.max(out[0]).item(), torch.var(out[0]).item())
        # print("var", torch.min(out[1]).item(), torch.mean(out[1]).item(), torch.max(out[1]).item(), torch.var(out[1]).item())
        # print("log_probs: ", torch.min(log_probs_for_actions).item(), torch.mean(log_probs_for_actions).item(), torch.max(log_probs_for_actions).item(), torch.var(log_probs_for_actions).item())
        # print("cum_ret: ", torch.min(cumulative_returns).item(), torch.mean(cumulative_returns).item(), torch.max(cumulative_returns).item(), torch.var(cumulative_returns).item())
        # print("actions: ", torch.min(actions, dim=0)[0], torch.mean(actions, dim=0), torch.max(actions, dim=0)[0], torch.var(actions, dim=0))

        loss_policy = -torch.sum(log_probs_for_actions * cumulative_returns)
        # loss_entropy = torch.tensor([0.0], dtype=torch.float32) # -torch.sum(distr.entropy()) # TODO implement!
        loss_entropy = -torch.mean(torch.exp(log_probs)*log_probs) * log_probs.size()[0]


        if self.decoder is not None: # TODO really keep this here ? 
            representation, repr_var = out[2], out[3]
            repres = reparameterize(representation, repr_var)
            recon_states = self.decoder(repres)
            loss_repre, BCE, KLD = vae_loss_fn(recon_states, states, representation, repr_var)
        else:
            loss_repre = torch.tensor([0.0], dtype=torch.float32)

        loss = loss_policy + expl*loss_entropy + repre*loss_repre
        loss.backward()
        self.optimizer.step()

        return  loss.item(), loss_policy.item(), loss_entropy.item(), loss_repre.item()



class ContinuousReinforce(ContinousRL):

    def train_step(self, *args, **kwargs):
        return self.train_on_session(*args, **kwargs)

    def train_on_session(self, states: List, actions: List, rewards: List, expl: float = 1.0, repre: float = 1.0):
        """
        Takes a sequence of states, actions and rewards produced by generate_session.
        Updates agent's weights by following the policy gradient above.
        """

        self.optimizer.zero_grad()

        # cast everything into torch tensors
        states = torch.tensor(states[:-1], dtype=torch.float32)
        if self.from_pixel:
            states = states.permute(0,3,1,2)/255
        actions = self.convert_from_action_space(torch.tensor(actions, dtype=torch.float32))
        cumulative_returns = np.array(rl_util.get_cumulative_rewards(rewards, self.gamma))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32).unsqueeze(dim=1)

        # predict logits, probas and log-probas using an agent.
        out = self.model(states)

        distr = self.distribution(out[0], out[1])
        log_probs = distr.log_prob(actions)

        # print(actions)
        # print(log_probs)
        # print(distr)
        print("mean", torch.min(out[0]).item(), torch.mean(out[0]).item(), torch.max(out[0]).item(), torch.var(out[0]).item())
        print("var", torch.min(out[1]).item(), torch.mean(out[1]).item(), torch.max(out[1]).item(), torch.var(out[1]).item())
        print("log_probs: ", torch.min(log_probs).item(), torch.mean(log_probs).item(), torch.max(log_probs).item(), torch.var(log_probs).item())
        print("cum_ret: ", torch.min(cumulative_returns).item(), torch.mean(cumulative_returns).item(), torch.max(cumulative_returns).item(), torch.var(cumulative_returns).item())
        print("actions: ", torch.min(actions, dim=0)[0], torch.mean(actions, dim=0), torch.max(actions, dim=0)[0], torch.var(actions, dim=0))
        loss_policy = -torch.sum(distr.log_prob(actions) * cumulative_returns)
        loss_entropy = -torch.sum(distr.entropy())

        if self.decoder is not None:
            representation, repr_var = out[2], out[3]
            repres = reparameterize(representation, repr_var)
            recon_states = self.decoder(repres)
            loss_repre, BCE, KLD = vae_loss_fn(recon_states, states, representation, repr_var)
        else:
            loss_repre = torch.tensor([0.0], dtype=torch.float32)

        loss = loss_policy + expl*loss_entropy + repre*loss_repre
        loss.backward()
        self.optimizer.step()

        return  loss.item(), loss_policy.item(), loss_entropy.item(), loss_repre.item()
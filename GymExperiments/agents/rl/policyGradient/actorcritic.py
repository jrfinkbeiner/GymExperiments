from typing import Union, Optional, List, Callable
import numpy as np

import torch
import torch.nn as nn

from GymExperiments.agents.rl.rlbase import DiscreteRL, ContinousRL
from GymExperiments.util import torch_util
from GymExperiments.util import rl_util
from GymExperiments.trainers.vae.train_vae import vae_loss_fn
from GymExperiments.architectures.vae import reparameterize


# TODO mean or sum in loss ? 

class DiscreteActorCritic(DiscreteRL):

    def train_step(self, *args, **kwargs):
        
        self.optimizer.zero_grad()
        
        return self.train_on_session(*args, **kwargs)

    # TODO identical to Continous... class structure should be different...
    def train_on_session(self, obs: List, actions: List, rewards: List, expl: float = 1.0, repre: float = 1.0):
        """
        Takes a sequence of states, actions and rewards produced by generate_session.
        Updates agent's weights by following the policy gradient above.
        """

        # cast everything into torch tensors
        all_obs = torch.tensor(obs, dtype=torch.float32)
        if self.from_pixel:
            states = states.permute(0,3,1,2)/255
        actions = self.convert_from_action_space(torch.tensor(actions, dtype=torch.float32))
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # TODO recalculation for critic could be omitted...
        return self.train_on_tensors(all_obs[:-1], actions, rewards, all_obs[1:], expl, repre)

    def train_on_tensors(self, obs: List, actions: List, rewards: List, next_obs: torch.Tensor, expl: float = 1.0, repre: float = 1.0):

        # predict logits, probas and log-probas using an agent.
        out = self.model(obs)

        if isinstance(out, dict):
            logits = out["action_logits"]
            vvalues = out["vvalues"]
        else:
            logits = out[0]
            vvalues = out[1]
            
        with torch.no_grad(): # TODO correct ?
            vvalues_prime = self.model.critic(next_obs)

        log_probs = nn.functional.log_softmax(logits, -1)
        log_probs_for_actions = torch.sum(log_probs * torch_util.to_one_hot(actions, log_probs.size()[-1]), dim=1)
        
        loss_vval = torch.sum((vvalues - (rewards + self.gamma * vvalues_prime))**2)

        with torch.no_grad(): # TODO correct ? 
            advantage = rewards + self.gamma*vvalues_prime - vvalues

        # # print(actions)
        # # print(log_probs)
        # # print(distr)
        # print("mean", torch.min(out[0]).item(), torch.mean(out[0]).item(), torch.max(out[0]).item(), torch.var(out[0]).item())
        # print("var", torch.min(out[1]).item(), torch.mean(out[1]).item(), torch.max(out[1]).item(), torch.var(out[1]).item())
        # print("log_probs: ", torch.min(log_probs).item(), torch.mean(log_probs).item(), torch.max(log_probs).item(), torch.var(log_probs).item())
        # print("cum_ret: ", torch.min(cumulative_returns).item(), torch.mean(cumulative_returns).item(), torch.max(cumulative_returns).item(), torch.var(cumulative_returns).item())
        # print("actions: ", torch.min(actions, dim=0)[0], torch.mean(actions, dim=0), torch.max(actions, dim=0)[0], torch.var(actions, dim=0))
        loss_policy = -torch.sum(log_probs_for_actions * advantage)
        loss_entropy = -torch.sum(torch.exp(log_probs)*log_probs) * log_probs.size()[0]

        print(f"loss_vval = {loss_vval}")
        print(f"loss_policy = {loss_policy}")
        print(f"loss_entropy = {loss_entropy}")

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



class ContinuousActorCritic(ContinousRL):

    def train_step(self, *args, **kwargs):
        
        self.optimizer.zero_grad()
        
        return self.train_on_session(*args, **kwargs)

    def train_on_session(self, obs: List, actions: List, rewards: List, expl: float = 1.0, repre: float = 1.0):
        """
        Takes a sequence of states, actions and rewards produced by generate_session.
        Updates agent's weights by following the policy gradient above.
        """

        # cast everything into torch tensors
        all_obs = torch.tensor(obs, dtype=torch.float32)
        if self.from_pixel:
            states = states.permute(0,3,1,2)/255
        actions = self.convert_from_action_space(torch.tensor(actions, dtype=torch.float32))
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # TODO recalculation for critic could be omitted...
        return self.train_on_tensors(all_obs[:-1], actions, rewards, all_obs[1:], expl, repre)

    def train_on_tensors(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_obs: torch.Tensor, expl: float = 1.0, repre: float = 1.0):

        # predict logits, probas and log-probas using an agent.
        out = self.model(obs)

        if isinstance(out, dict):
            dirtr_args = out["action_distributions"]
            vvalues = out["vvalues"]
        else:
            dirtr_args = out[0]
            vvalues = out[1]

        with torch.no_grad: # TODO correct ?
            vvalues_prime = self.model.critic(next_obs)

        distr = self.distribution(*dirtr_args)
        log_probs = distr.log_prob(actions)

        loss_vval = vvalues - (rewards + self.gamma * vvalues_prime )

        with torch.no_grad: # TODO correct ? 
            advantage = rewards + self.gamma*vvalues_prime - vvalues

        # # print(actions)
        # # print(log_probs)
        # # print(distr)
        # print("mean", torch.min(out[0]).item(), torch.mean(out[0]).item(), torch.max(out[0]).item(), torch.var(out[0]).item())
        # print("var", torch.min(out[1]).item(), torch.mean(out[1]).item(), torch.max(out[1]).item(), torch.var(out[1]).item())
        # print("log_probs: ", torch.min(log_probs).item(), torch.mean(log_probs).item(), torch.max(log_probs).item(), torch.var(log_probs).item())
        # print("cum_ret: ", torch.min(cumulative_returns).item(), torch.mean(cumulative_returns).item(), torch.max(cumulative_returns).item(), torch.var(cumulative_returns).item())
        # print("actions: ", torch.min(actions, dim=0)[0], torch.mean(actions, dim=0), torch.max(actions, dim=0)[0], torch.var(actions, dim=0))
        loss_policy = -torch.sum(log_probs * advantage)
        loss_entropy = -torch.sum(distr.entropy())

        print(f"loss_vval = {loss_vval}")
        print(f"loss_policy = {loss_policy}")
        print(f"loss_entropy = {loss_entropy}")

        if self.decoder is not None:
            representation, repr_var = out[2], out[3]
            repres = reparameterize(representation, repr_var)
            recon_obs = self.decoder(repres)
            loss_repre, BCE, KLD = vae_loss_fn(recon_obs, obs, representation, repr_var)
        else:
            loss_repre = torch.tensor([0.0], dtype=torch.float32)

        loss = loss_policy + loss_vval + expl*loss_entropy + repre*loss_repre # TODO correct to update critic only here?
        loss.backward()
        self.optimizer.step()

        return  loss.item(), loss_policy.item(), loss_entropy.item(), loss_repre.item()
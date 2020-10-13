from abc import ABC, abstractmethod
from typing import Union, Optional, List
import numpy as np

import torch

from GymExperiments.util import torch_util
from GymExperiments.util import rl_util
from GymExperiments.trainers.vae.train_vae import vae_loss_fn
from GymExperiments.architectures.vae import reparameterize


class Reinforce(ABC):    
    def __init__(
            self,
            env,
            model,
            optimizer,
            gamma: float, 
            decoder: Optional[torch.nn.Module] = None,
            latent_represantation: bool = True,
        ):
        super().__init__()
        # TODO check in setters for consistency between env, model and latent_represantation
        self.env = env
        
        if decoder is not None:
            assert latent_represantation
        self.model = (model, latent_represantation)
        self.decoder = decoder
        self.optimizer = optimizer
        self.gamma = gamma

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_tuple):
        if isinstance(model_tuple, (list, tuple)):
            if len(model_tuple) == 1:
                self._model = model_tuple[0]
                self._latent_represantation = True
            elif len(model_tuple) == 2:
                self._model = model_tuple[0]
                self._latent_represantation = model_tuple[1]
        else:
            self._model = model_tuple

    @property
    def latent_represantation(self):
        return self._latent_represantation

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        if hasattr(self, "model"):
            pass # TODO check that model fits to env..?
        self._env = env

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError

    def generate_session(self, max_len: int, state: Optional=None):
        if state is None:
            state = self.env.reset()
        
        states = [state.transpose(2,0,1)]
        rewards = []
        actions = []
        for _ in range(max_len):
            state = states[-1]
            action, _, _ = self.get_action(torch.tensor(state.copy(), dtype=torch.float32).unsqueeze(dim=0)/255.) # TODO copy is shit!
            # print()
            # print(action)
            # print(type(action))
            # print(action.dtype)
            # print()
            state, reward, done, _ = self.env.step(action)
            
            states.append(state.transpose(2,0,1)) 
            rewards.append(reward)
            actions.append(action)

            if done:
                break
        
        return states, rewards, actions


class ContinousReinforce(Reinforce):    

    def __init__(
            self,
            env,
            model,
            optimizer,
            gamma: float, 
            distribution: torch.distributions.distribution.Distribution,
            decoder: Optional[torch.nn.Module] = None,
            latent_represantation: Optional[bool] = True,
        ):
        super().__init__(env, model, optimizer, gamma, decoder, latent_represantation)

        self.distribution = distribution

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution: str):
        # assert isinstance(distribution, torch.distributions.distribution.Distribution)

        if distribution == "Normal":
        # if isinstance(distribution, torch.distributions.normal.Normal):
            self.get_mode = torch_util.get_mode_normal
            self._distribution = torch.distributions.normal.Normal
            # self.log_distr = torch.distributions.log_normal.LogNormal
        elif distribution == "LogNormal":
        # elif isinstance(torch.distributions.log_normal.LogNormal):
            self.get_mode = torch_util.get_mode_normal
            self._distribution = torch.distributions.normal.LogNormal
            # self.log_distr = None
        elif distribution == "Beta":
        # elif isinstance(distribution, torch.distributions.beta.Beta):
            self._distribution = torch.distributions.beta.Beta
            self.get_mode = torch_util.get_mode_beta
            # self.log_distr = None
        else:
            raise ValueError("Currently it is only possible to use ``torch.distributions.normal.Normal``, ``torch.distributions.log_normal.LogNormal`` or ``torch.distributions.beta.Beta``")

        # support = self.distribution.support
        # print(support)
        # sys.exit() # TODO remove

    def convert_to_action_space(self, action_distr: np.ndarray):
        # print()
        # import sys
        # print(self.distribution.support)
        # # print(min(self.distribution.support))
        # # print(max(self.distribution.support))
        # print()
        # print(self.env.action_space.low)
        # print(self.env.action_space.high)

        # print(self.distribution.support.check(self.env.action_space.low))
        # print(self.distribution.support.check(self.env.action_space.high))

        # # TODO assert isinstance(support, Interval) the apply transformation
        # # depending on action space also half open interval might be ok

        # sys.exit()
        action = action_distr.copy()
        action[0] = action[0]*2-1
        return action

    def convert_from_action_space(self, action: torch.Tensor):
        
        action_distr = action
        action_distr[:,0] = action_distr[:,0]*0.5+0.5
        return action_distr

    def get_action(self, state: torch.Tensor, mode: bool = False):
        with torch.no_grad():
            out0, out1, representation, repr_var = self.model(state)
            
            distribution = self.distribution(out0, out1)

            if mode:
                action_distr = self.get_mode(distribution)
            else:
                action_distr = distribution.sample()
            action_distr = action_distr.cpu().flatten().numpy()

        action = self.convert_to_action_space(action_distr)

        return action, action_distr, (out0, out1, representation, repr_var)

    def train_on_session(self, states: List, actions: List, rewards: List, expl: float = 1.0, repre: float = 1.0):
        """
        Takes a sequence of states, actions and rewards produced by generate_session.
        Updates agent's weights by following the policy gradient above.
        """

        self.optimizer.zero_grad()

        # cast everything into torch tensors
        states = torch.tensor(states[:-1], dtype=torch.float32)/255
        actions = self.convert_from_action_space(torch.tensor(actions, dtype=torch.float32))
        cumulative_returns = np.array(rl_util.get_cumulative_rewards(rewards, self.gamma))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32).unsqueeze(dim=1)

        # predict logits, probas and log-probas using an agent.
        out0, out1, representation, repr_var = self.model(states)

        # if self.log_distr is not None: # TODO unnecessary: remove
        #     log_distr = self.log_distr(out0, out1)
        # else:
        #     log_distr = torch.log(self.distribution(out0, out1))

        distr = self.distribution(out0, out1)

        loss_policy = -torch.sum(distr.log_prob(actions) * cumulative_returns)
        loss_entropy = -torch.sum(distr.entropy())

        if self.decoder is not None:
            repres = reparameterize(representation, repr_var)
            recon_states = self.decoder(repres)
            loss_repre, BCE, KLD = vae_loss_fn(recon_states, states, representation, repr_var)
        else:
            loss_repre = 0.0

        loss = loss_policy + expl*loss_entropy + repre*loss_repre
        loss.backward()

        return  loss.item(), loss_policy.item(), loss_entropy.item(), loss_repre.item()
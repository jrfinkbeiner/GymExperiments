from abc import ABC, abstractmethod
from typing import Union, Optional, List, Callable
import numpy as np

import torch
import torch.nn as nn

import gym
import gym.wrappers

from GymExperiments.util import torch_util
from GymExperiments.util import rl_util
from GymExperiments.trainers.vae.train_vae import vae_loss_fn
from GymExperiments.architectures.vae import reparameterize
from GymExperiments.util.torch_util import to_one_hot

class Reinforce(ABC):    
    def __init__(
            self,
            env,
            model,
            optimizer,
            gamma: float, 
            decoder: Optional[torch.nn.Module] = None,
            convert_to_action_space: Callable = None, 
            convert_from_action_space: Callable = None,
            from_pixel: bool = False,
        ):
        super().__init__()
        # TODO check in setters for consistency between env, model and latent_represantation
        self.env = env
        
        self.model = model
        self.decoder = decoder
        self.optimizer = optimizer
        self.gamma = gamma
        self.from_pixel = from_pixel

        if convert_to_action_space is None:
            self.convert_to_action_space = lambda x: x
        else:
            self.convert_to_action_space = convert_to_action_space

        if convert_from_action_space is None:
            self.convert_from_action_space = lambda x: x
        else:
            self.convert_from_action_space = convert_from_action_space

    # def convert_to_action_space(self, x):
    #     self._convert_to_action_space(x)

    # def convert_from_action_space(self, x):
    #     self._convert_from_action_space(x)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_tuple):
        # if isinstance(model_tuple, (list, tuple)):
        #     if len(model_tuple) == 1:
        #         self._model = model_tuple[0]
        #         self._latent_represantation = True
        #     elif len(model_tuple) == 2:
        #         self._model = model_tuple[0]
        #         self._latent_represantation = model_tuple[1]
        # else:
        self._model = model_tuple

    # @property
    # def latent_represantation(self):
    #     return self._latent_represantation

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

    def generate_session(self, max_len: int, state: Optional = None, **kwargs):


        if kwargs != {}:
            assert state is None
            return self._generate_monitor_session(max_len, **kwargs)
        else:
            return self._generate_session(self.env, max_len, state)

    def _generate_monitor_session(self, max_len: int, **kwargs):
        with gym.wrappers.Monitor(self.env, **kwargs) as env_monitor:
            return self._generate_session(env_monitor, max_len)        

    def _generate_session(self, env: gym.Env, max_len: int, state: Optional=None):
        if state is None:
            state = env.reset()
        
        states = [state]
        rewards = []
        actions = []
        for _ in range(max_len):
            state = states[-1]
            
            torch_state = torch.tensor(state.copy(), dtype=torch.float32).unsqueeze(dim=0)
            if self.from_pixel:
                torch_state =  torch_state.permute(0,3,1,2)/255.

            action, _ = self.get_action(torch_state) # TODO copy is shit!
            # print()
            # print(action)
            # print(type(action))
            # print(action.dtype)
            # print()
            state, reward, done, _ = env.step(action)
            
            states.append(state) 
            rewards.append(reward)
            actions.append(action)

            if done:
                break

        return states, rewards, actions


def generate_epsilon_greed_discrete_action_sample(epsilon):

    def epsilon_greed_discrete_action_sample(action_probs: torch.Tensor):
        num_actions = len(action_probs)
        if np.random.random() < self._epsilon:
            action = np.random.randint(num_actions)
        else:
            action = int(torch.argmax(self._successes / (self._successes + self._failures)).flatten()[0])
        return action

    return epsilon_greed_discrete_action_sample


def proportional_discrete_action_sample(action_probs: torch.Tensor):
    return np.random.choice(np.arange(len(action_probs)), p=action_probs.numpy())


class DiscreteReinforce(Reinforce):    

    def __init__(self, exploration: Optional[str] = None, epsilon: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
       
        self.set_exploration(exploration, epsilon=epsilon)
        self.epsilon = epsilon

    @property
    def exploration(self):
        return self._exploration

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon
        self.set_exploration(self.exploration, epsilon=epsilon)

    def set_exploration(self, exploration, **kwargs):
        if exploration is None or exploration == 'none':
            self._sample_action = lambda x: torch.argmax(x).item() # TODO breaks for identical values?
        elif exploration == 'e-greedy':
            assert "epsilon" in kwargs
            self._epsilon = kwargs["epsilon"]
            self._sample_action = generate_epsilon_greed_discrete_action_sample(kwargs["epsilon"])
        elif exploration == 'proportional':
            self._sample_action = proportional_discrete_action_sample
        else:
            raise ValueError(f"Given exploration-mode '{exploration}' is not implemented.")
        self._exploration = exploration

    def get_action(self, state: torch.Tensor, mode: bool = False):
        with torch.no_grad():
            out = self.model(state)

            if isinstance(out, torch.Tensor):
                logits = out
            else: 
                logits = out[0]

            if mode:
                action = int(torch.argmax(logits.squeeze(dim=0)).item()) # TODO breaks for identical values?
            else:
                actions_probs = torch.softmax(logits.squeeze(dim=0), dim=0)
                action = self._sample_action(actions_probs)
        return action, out


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
        log_probs_for_actions = torch.sum(log_probs * to_one_hot(actions, log_probs.size()[-1]), dim=1)
        
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




class ContinousReinforce(Reinforce):    

    def __init__(
            self,
            env,
            model,
            optimizer,
            gamma: float, 
            distribution: torch.distributions.distribution.Distribution,
            decoder: Optional[torch.nn.Module] = None,
            **kwargs
        ):
        super().__init__(env, model, optimizer, gamma, decoder, **kwargs)

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

    def get_action(self, state: torch.Tensor, mode: bool = False):
        with torch.no_grad():
            out = self.model(state)
            distribution = self.distribution(out[0], out[1])

            if mode:
                action_distr = self.get_mode(distribution)
            else:
                action_distr = distribution.sample()
            action_distr = action_distr.cpu().flatten().numpy()

        action = self.convert_to_action_space(action_distr)

        return action, out

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
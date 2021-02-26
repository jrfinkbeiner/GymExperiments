import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import gym

from GymExperiments.agents.rl.policyGradient.actorcritic import ContinuousActorCritic
from GymExperiments.trainers.train_sessions import train_sessions
from GymExperiments.architectures.multihead import Dualhead, ReprDualhead
from GymExperiments.architectures.blocks import MLP
from GymExperiments.util.gym_util import create_video_callable

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        base = MLP([nn.ReLU(), nn.ReLU()], [3, 64, 64])
        head0 = MLP([nn.ReLU(), nn.Tanh()], [64, 32, 1])
        head1 = MLP([nn.ReLU(), nn.Softplus()], [64, 32, 1])

        self.dualhead = Dualhead(base, head0, head1)
        # self.scale = 1.0

    def forward(self, inp):
        x0, x1 = self.dualhead(inp)
        return 2*x0, 0.3*x1


class PendulumModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor = Actor()
        self.critic =  MLP([nn.ReLU(), nn.ReLU(), None], [3, 64, 64, 1])

    def forward(self, inp):
        action_distr = self.actor(inp)
        vvals = self.critic(inp)
        return action_distr, vvals


def main():

    env = gym.make('Pendulum-v0')

    # print("\nstate")
    # state = env.reset()
    # print(state.shape)
    # print("\action")
    # print(env.action_space)
    # print(env.action_space.high)
    # print(env.action_space.low)
    # sys.exit()

    save_ith_epoch = 500
    dir_name = "./model_saves/actorcritic/try5k/"


    model = torch.load(os.path.join(dir_name, "best_model"))
    # model = PendulumModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    kwargs = {"directory": os.path.join(dir_name, "monitor"), "resume": True, "force": True, "video_callable": create_video_callable(save_ith_epoch)}
    with gym.wrappers.Monitor(env, **kwargs) as env_monitor:
    
        agent = ContinuousActorCritic(
            env=env_monitor,
            model=model,
            optimizer=optimizer,
            gamma=0.98, 
            distribution="Normal",
            decoder=None,
        )

        rewards = train_sessions(
            num_epochs=5000,
            agent=agent,
            dir_name=dir_name,
            save_ith_epoch=save_ith_epoch,
            monitor=False,
            init_expl=100,
            start_epoch=10000,
        )

    env.close()

    print(rewards)
    plt.figure()
    plt.plot(rewards)
    plt.plot(np.convolve(rewards, np.ones((50,))/50, mode='same'))
    plt.show()

    



if __name__ == "__main__":
    main()



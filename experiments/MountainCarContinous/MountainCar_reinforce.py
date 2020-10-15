import os
import sys  
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import gym

from GymExperiments.agents.rl.policyGradient.REINFORCE.reinforce import ContinousReinforce
from GymExperiments.trainers.reinforce.train_reinforce import train_reinforce
from GymExperiments.architectures.multihead import Dualhead, ReprDualhead
from GymExperiments.architectures.blocks import MLP
from GymExperiments.util.gym_util import create_video_callable

class MouintainCarContinousModel(nn.Module):
    def __init__(self):
        super().__init__()

        base = MLP([nn.ReLU(), nn.ReLU()], [2, 16, 16])
        head0 = MLP([nn.ReLU(), nn.Tanh()], [16, 8, 1])
        head1 = MLP([nn.ReLU(), nn.Softplus()], [16, 8, 1])

        self.dualhead = Dualhead(base, head0, head1)
        # self.scale = 1.0

    def forward(self, inp):
        x0, x1 = self.dualhead(inp)
        return x0, 0.5*x1
        # return self.scale*x0, x1


def main():

    env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('CarRacing-v0')

    # print("\nstate")
    # state = env.reset()
    # print(state.shape)
    # print("\action")
    # print(env.action_space)
    # print(env.action_space.high)
    # print(env.action_space.low)
    # sys.exit()


    save_ith_epoch = 100
    dir_name = "./model_saves/reinforce/try5/"

    kwargs = {"directory": os.path.join(dir_name, "monitor"), "resume": False, "force": True, "video_callable": create_video_callable(save_ith_epoch)}
    
    with gym.wrappers.Monitor(env, **kwargs) as env_monitor:
    
        model = MouintainCarContinousModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        reinforce = ContinousReinforce(
            env=env_monitor,
            model=model,
            optimizer=optimizer,
            gamma=0.98, 
            distribution="Normal",
            decoder=None,
        )

        rewards = train_reinforce(
            num_epochs=10000,
            reinforce=reinforce,
            dir_name=dir_name,
            save_ith_epoch=save_ith_epoch,
            monitor=False,
        )

    env.close()

    print(rewards)
    plt.figure()
    plt.plot(rewards)
    plt.plot(np.convolve(rewards, np.ones((50,))/50, mode='same'))
    plt.show()

    



if __name__ == "__main__":
    main()



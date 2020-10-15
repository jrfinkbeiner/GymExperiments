import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import gym

from GymExperiments.agents.rl.policyGradient.REINFORCE.reinforce import DiscreteReinforce
from GymExperiments.trainers.reinforce.train_reinforce import train_reinforce
from GymExperiments.architectures.blocks import MLP
from GymExperiments.util.gym_util import create_video_callable

class MountainCar(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = MLP([nn.ReLU(), nn.ReLU()], [2, 64, 64])
        self.linear = nn.Linear(64, 3)
        
    def forward(self, inp):
        x = self.mlp(inp)
        x = self.linear(x)
        return x


def main():

    env = gym.make('MountainCar-v0')

    # print("\nstate")
    # state = env.reset()
    # print(state.shape)
    # print("\action")
    # print(env.action_space)
    # # print(env.action_space.high)
    # # print(env.action_space.low)
    # sys.exit()


    save_ith_epoch = 100
    dir_name = "./model_saves/reinforce/try0/"




    kwargs = {"directory": os.path.join(dir_name, "monitor"), "resume": True, "force": True, "video_callable": create_video_callable(save_ith_epoch)}
    
    with gym.wrappers.Monitor(env, **kwargs) as env_monitor:
    
        model = MountainCar()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        reinforce = DiscreteReinforce(
            env=env_monitor,
            model=model,
            optimizer=optimizer,
            gamma=0.98, 
            exploration='proportional'
        )

        rewards = train_reinforce(
            num_epochs=1000,
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



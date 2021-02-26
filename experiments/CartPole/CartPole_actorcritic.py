import os
import sys  
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import gym

from GymExperiments.agents.rl.policyGradient.actorcritic import DiscreteActorCritic
from GymExperiments.trainers.train_sessions import train_sessions
from GymExperiments.architectures.blocks import MLP


class CartPoleACModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor = MLP([nn.ReLU(), nn.ReLU(), None], [4, 64, 64, 2])
        self.critic = MLP([nn.ReLU(), nn.ReLU(), None], [4, 64, 64, 1])
        
    def forward(self, inp):
        action_logits =  self.actor(inp)
        vvalues = self.critic(inp)
        return action_logits, vvalues


def main():

    env = gym.make('CartPole-v1')

    # print("\nstate")
    # state = env.reset()
    # print(state.shape)
    # print("\action")
    # print(env.action_space)
    # print(env.action_space.high)
    # print(env.action_space.low)
    # sys.exit()


    save_ith_epoch = 500
    dir_name = "./model_saves/actorcritic/try5k_expl/"

    def create_video_callable(save_ith_episode):
        def video_callable(episode):
            return True if ((episode+1) % save_ith_episode == 0) else False
        return video_callable 

    kwargs = {"directory": os.path.join(dir_name, "monitor"), "resume": True, "force": True, "video_callable": create_video_callable(save_ith_epoch)}
    
    with gym.wrappers.Monitor(env, **kwargs) as env_monitor:
    
        model = CartPoleACModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        agent = DiscreteActorCritic(
            env=env_monitor,
            model=model,
            optimizer=optimizer,
            gamma=0.98, 
            exploration='proportional'
        )

        rewards = train_sessions(
            num_epochs=5000,
            agent=agent,
            dir_name=dir_name,
            save_ith_epoch=save_ith_epoch,
            monitor=False,
        )

    env.close()

    print(rewards)
    plt.figure()
    plt.plot(rewards)
    plt.plot(np.convolve(rewards, np.ones((100,))/100, mode='same'))
    plt.show()

    



if __name__ == "__main__":
    main()



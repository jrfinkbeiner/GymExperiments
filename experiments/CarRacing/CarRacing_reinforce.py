import os
import sys  
import numpy as np
import matplotlib.pyplot as plt

import torch

import gym

from GymExperiments.agents.rl.policyGradient.reinforce import ContinousReinforce
from GymExperiments.util.carracing_util import load_observations, transform_observations_torch
from GymExperiments.architectures import instaniate_SimpleCNNVAE
from GymExperiments.architectures.combined.from_pixels import set_up_repr_dualhead_from_pixels
from GymExperiments.trainers.vae.train_vae import train_vae, vae_loss_fn
from GymExperiments.trainers.train_sessions import train_sessions
from GymExperiments.util.gym_util import create_video_callable


def convert_to_action_space(action_distr: np.ndarray):
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
    action[0] = (action[0]-0.5)*2
    return action

def convert_from_action_space(action: torch.Tensor):
    
    action_distr = action
    action_distr[:,0] = action_distr[:,0]*0.5+0.5
    return action_distr



def main():
    num_epochs = 20
    save_ith_epoch = 2
    batch_size = 20
    num_workers = None

    representation_dim = 16
    module_dir_vae = "./model_saves/simpleCNNVAE_startGas_nroll100_lroll100"
    # module_vae = instaniate_SimpleCNNVAE(representation_dim, image_channels=3)
    # criterion = vae_loss_fn
    # optimizer_vae = torch.optim.Adam(module_vae.parameters(), lr=1e-3)

    # train_observations = load_observations("./observations/startGas/observations_startGas_nroll100_lroll100_000000.npy")[:, ::10, :, :, :]
    # train_observations = transform_observations_torch(train_observations)

    # val_observations = load_observations("./observations/startGas/observations_startGas_nroll100_lroll100_000001.npy")[:, ::20, :, :, :]
    # val_observations = transform_observations_torch(val_observations)

    # train_vae(
    #     num_epochs = num_epochs,
    #     module = module_vae,
    #     module_dir = module_dir_vae,
    #     train_observations = train_observations,
    #     val_observations = val_observations,
    #     criterion = criterion,
    #     optimizer = optimizer_vae,
    #     save_ith_epoch = save_ith_epoch,
    #     batch_size = batch_size,
    #     num_workers = num_workers,
    # )

    module_vae = torch.load(os.path.join(module_dir_vae, "best_module"))

    env = gym.make('CarRacing-v0')

    # env.reset()
    # env.reset()

    # print(env.action_space.high)
    # print(env.action_space.low)
    # sys.exit()

    encoder = module_vae.vae_encoder
    decoder = module_vae.decoder

    model = set_up_repr_dualhead_from_pixels(encoder, encoder_out_dim=representation_dim, out_dim=3)
    optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": decoder.parameters()}], lr=1e-3)

    save_ith_epoch = 2
    dir_name = "./model_saves/reinforce/try0/"

    kwargs = {"directory": os.path.join(dir_name, "monitor"), "resume": True, "force": True, "video_callable": create_video_callable(save_ith_epoch)}
    
    with gym.wrappers.Monitor(env, **kwargs) as env_monitor:

        agent = ContinousReinforce(
            env=env_monitor,
            model=model,
            optimizer=optimizer,
            gamma=0.98, 
            distribution="Normal",
            decoder=decoder,
            from_pixel=True,
            convert_to_action_space=convert_to_action_space,
            convert_from_action_space=convert_from_action_space,
        )
            
        rewards = train_sessions(
            num_epochs=20,
            agent=agent,
            dir_name=dir_name,
            save_ith_epoch=1,
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



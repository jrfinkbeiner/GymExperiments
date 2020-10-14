import os
import sys  
import matplotlib.pyplot as plt

import torch

import gym

from GymExperiments.agents.rl.policyGradient.REINFORCE.reinforce import ContinousReinforce
from GymExperiments.util.carracing_util import load_observations, transform_observations_torch
from GymExperiments.architectures import instaniate_SimpleCNNVAE
from GymExperiments.architectures.combined.from_pixels import set_up_repr_dualhead_from_pixels
from GymExperiments.trainers.vae.train_vae import train_vae, vae_loss_fn
from GymExperiments.trainers.reinforce.train_reinforce import train_reinforce



def main():
    num_epochs = 20
    save_ith_epoch = 2
    batch_size = 20
    num_workers = None

    representation_dim = 16
    module_dir_vae = "./module_saves/simpleCNNVAE_startGas_nroll100_lroll100"
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
    optimizer = torch.optim.Adam(module_vae.parameters(), lr=1e-4)

    reinforce = ContinousReinforce(
        env=env,
        model=model,
        optimizer=optimizer,
        gamma=0.98, 
        distribution="Normal",
        decoder=decoder,
        latent_represantation=True,
    )
        
    rewards = train_reinforce(
        num_epochs=10000,
        reinforce=reinforce,
        dir_name="./module_saves/carracing/reinforce/Normal10k/",
        save_ith_epoch=10,
        save_videos=True,        
    )

    print(rewards)
    plt.figure()
    plt.plot(rewards)
    plt.show()







if __name__ == "__main__":
    main()



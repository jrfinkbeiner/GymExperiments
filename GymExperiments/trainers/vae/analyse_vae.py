import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from carracing_util import load_observations, transform_observations_torch


def main():

    module_dir = "/home/jan/Documents/MachineLearning/GymExperiments/fromPixels/CarRacing/module_saves/simpleCNNVAE_startGas_nroll100_lroll100"
    # indices = np.arange(10,1000,100)

    # val_obs = load_observations("./observations/observations_nroll10_lroll100.npy")
    # val_obs = transform_observations_torch(val_obs)[indices]

    val_obs = load_observations("./observations/startGas/observations_startGas_nroll100_lroll100_000001.npy")[::10, 5::50, :, :, :]
    val_obs = transform_observations_torch(val_obs)

    with torch.no_grad():
        module = torch.load(os.path.join(module_dir, "best_module"))
        module.eval()
        recon_obs, mu, logvar = module(val_obs)

        val_obs = val_obs.detach().cpu().numpy().transpose(0,2,3,1)
        recon_obs = recon_obs.detach().cpu().numpy().transpose(0,2,3,1)



    print(len(val_obs))
    for i in range(len(val_obs)):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(val_obs[i])
        ax[1].imshow(recon_obs[i])

    plt.show()




if __name__ == "__main__":
    main()
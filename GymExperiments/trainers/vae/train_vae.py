import os
import numpy as np
from multiprocessing import cpu_count
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from GymExperiments.architectures import instaniate_SimpleCNNVAE


def vae_loss_fn(recon_x, x, mu, logvar):

    BCE = F.binary_cross_entropy(recon_x, x[:,:, :recon_x.size()[2], :recon_x.size()[3]], size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD



def train_vae(
        num_epochs,
        module,
        module_dir,
        train_observations,
        val_observations,
        criterion,
        optimizer,
        lr_scheduler = None,
        save_ith_epoch = 1,
        batch_size = 1,
        num_workers = None,
    ):

    num_cores = cpu_count()
    if not num_workers:
        num_workers = num_cores
        print(f'Number of cores used = {num_cores}')
    else:
        if num_workers > num_cores:
            num_workers = num_cores
            print(f'Number of cores used reduced to {num_cores}, due to avaiable cores constraint.')
        if num_workers > batch_size:
            print(f'Number of cores used reduced to {num_cores}, due to batch size constraint.')

    checkpoint_dir = os.path.join(module_dir, "checkpoints")
    tensorboard_dir = os.path.join(module_dir, "tensorboard")

    if not os.path.exists(module_dir):
        os.makedirs(module_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # os.makedirs(tensorboard_dir, exist_ok=True)

    
    writer = SummaryWriter(log_dir=tensorboard_dir, comment=module_dir)



    trainloader = DataLoader(train_observations, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_observations, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    smallest_loss = 1e32
    for epoch in tqdm.tqdm(range(num_epochs)):
        running_loss = 0
        running_BCE = 0
        running_KLD = 0
        for obs_batch in trainloader:
            optimizer.zero_grad()
            recon_obs, mu, logvar = module(obs_batch)
            loss, BCE, KLD = criterion(recon_obs, obs_batch, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_BCE += BCE.item()
            running_KLD += KLD.item()


        # running_loss *= batch_size / len(torch_observations)

        if lr_scheduler:
            lr_scheduler.setp()

        if (epoch+1) % save_ith_epoch == 0:
            with torch.no_grad():
                running_loss_val = 0
                running_BCE_val = 0
                running_KLD_val = 0
                for obs_batch in valloader:
                    recon_obs, mu, logvar = module(obs_batch)
                    loss, BCE, KLD = criterion(recon_obs, obs_batch, mu, logvar)
                    running_loss_val += loss
                    running_BCE_val += BCE
                    running_KLD_val += KLD

            save_dict = {
                'epoch': epoch+1,
                'model_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss-train': running_loss,
                'loss-val': running_loss_val,
            }
            if lr_scheduler:
                save_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            torch.save(save_dict, os.path.join(checkpoint_dir, f"checkpoint_{str(epoch+1).zfill(6)}.tar"))
            
            if smallest_loss > running_loss_val:
                torch.save(module, os.path.join(module_dir, "best_module"))

            writer.add_scalar('loss-val', running_loss_val, epoch)
            writer.add_scalar('loss-val-BCE', running_BCE_val, epoch)
            writer.add_scalar('loss-val-KLD', running_KLD_val, epoch)

        writer.add_scalar('loss-train', running_loss, epoch)
        writer.add_scalar('loss-train-BCE', running_BCE, epoch)
        writer.add_scalar('loss-train-KLD', running_KLD, epoch)


def main():

    num_epochs = 20
    save_ith_epoch = 2
    batch_size = 20
    num_workers = 4

    representation_dim = 16
    module_dir = "./module_saves/simpleCNNVAE_startGas_nroll100_lroll100"
    module = instaniate_SimpleCNNVAE(representation_dim, image_channels=3)
    criterion = vae_loss_fn
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

    train_observations = load_observations("./observations/startGas/observations_startGas_nroll100_lroll100_000000.npy")[:, ::10, :, :, :]
    train_observations = transform_observations_torch(train_observations)

    val_observations = load_observations("./observations/startGas/observations_startGas_nroll100_lroll100_000001.npy")[:, ::20, :, :, :]
    val_observations = transform_observations_torch(val_observations)
    

    train_vae(
        num_epochs = num_epochs,
        module = module,
        module_dir = module_dir,
        train_observations = train_observations,
        val_observations = val_observations,
        criterion = criterion,
        optimizer = optimizer,
        save_ith_epoch = save_ith_epoch,
        batch_size = batch_size,
        num_workers = num_workers,
    )


if __name__ == "__main__":
    
    main()

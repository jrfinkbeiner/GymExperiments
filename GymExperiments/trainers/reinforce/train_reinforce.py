import os
import numpy as np
from multiprocessing import cpu_count
import tqdm

import torch

from GymExperiments.agents.rl.policyGradient.REINFORCE.reinforce import Reinforce 

def train_reinforce(
        num_epochs: int,
        reinforce: Reinforce,
        dir_name: str,
        save_ith_epoch = 1,
    ):

    checkpoint_dir = os.path.join(dir_name, "checkpoints")

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # os.makedirs(tensorboard_dir, exist_ok=True)

    total_rewards = np.empty(num_epochs)

    smallest_loss = 1e32
    for epoch in tqdm.tqdm(range(num_epochs)):

        max_len = 300+10*epoch # TODO
        expl = np.exp(-0.005*epoch) # TODO
        repre = np.exp(-0.01*epoch) # TODO

        states, rewards, actions = reinforce.generate_session(max_len=max_len)
        loss, loss_policy, loss_entropy, loss_repre = reinforce.train_on_session(states, actions, rewards, expl=expl, repre=repre)

        total_rewards[epoch] = sum(rewards)

        if (epoch+1) % save_ith_epoch == 0:

            save_dict = {
                'epoch': epoch+1,
                'model_state_dict': reinforce.model.state_dict(),
                'optimizer_state_dict': reinforce.optimizer.state_dict(),
                'reward': total_rewards[epoch],
            }
            if hasattr(reinforce, "lr_scheduler"):
                save_dict['lr_scheduler_state_dict'] = reinforce.lr_scheduler.state_dict()
            torch.save(save_dict, os.path.join(checkpoint_dir, f"checkpoint_{str(epoch+1).zfill(6)}.tar"))
            
            if smallest_loss > loss:
                torch.save(reinforce.model, os.path.join(dir_name, "best_model"))

        #     writer.add_scalar('loss-val', running_loss_val, epoch)
        #     writer.add_scalar('loss-val-BCE', running_BCE_val, epoch)
        #     writer.add_scalar('loss-val-KLD', running_KLD_val, epoch)

        # writer.add_scalar('loss-train', running_loss, epoch)
        # writer.add_scalar('loss-train-BCE', running_BCE, epoch)
        # writer.add_scalar('loss-train-KLD', running_KLD, epoch)

    return total_rewards
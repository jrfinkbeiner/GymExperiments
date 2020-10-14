import os
import numpy as np
from multiprocessing import cpu_count, Pool
import tqdm

import torch

from GymExperiments.agents.rl.policyGradient.REINFORCE.reinforce import Reinforce 
from GymExperiments.util.carracing_util import create_video

def train_reinforce(
        num_epochs: int,
        reinforce: Reinforce,
        dir_name: str,
        save_ith_epoch: int = 1,
        save_videos: bool = True,
    ):

    checkpoint_dir = os.path.join(dir_name, "checkpoints")

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if save_videos:
        video_dir = os.path.join(dir_name, "obs_vids")
        os.makedirs(video_dir, exist_ok=True)
    # os.makedirs(tensorboard_dir, exist_ok=True)

    total_rewards = np.empty(num_epochs)

    smallest_loss = 1e32
    for epoch in tqdm.tqdm(range(num_epochs)):

        max_len = 3000+5*epoch # TODO
        expl = 0.5*np.exp(-0.001*epoch) # TODO
        repre = 0.0001*np.exp(-0.001*epoch) # TODO

        # with Pool(processes=1) as p:
        #     session = p.map(reinforce.generate_session, [max_len])
        states, rewards, actions = reinforce.generate_session(max_len=max_len)

        loss, loss_policy, loss_entropy, loss_repre = reinforce.train_on_session(states, actions, rewards, expl=expl, repre=repre)

        total_rewards[epoch] = sum(rewards)

        if (epoch+1) % save_ith_epoch == 0:

            save_dict = {
                'epoch': epoch+1,
                'model_state_dict': reinforce.model.state_dict(),
                'optimizer_state_dict': reinforce.optimizer.state_dict(),
                'reward': total_rewards[epoch],
                'loss': loss,
                'loss_policy': loss_policy,
                'loss_entropy': loss_entropy,
                'loss_repre': loss_repre,
            }
            if hasattr(reinforce, "lr_scheduler"):
                save_dict['lr_scheduler_state_dict'] = reinforce.lr_scheduler.state_dict()
            torch.save(save_dict, os.path.join(checkpoint_dir, f"checkpoint_{str(epoch+1).zfill(6)}.tar"))
            
            if smallest_loss > loss:
                torch.save(reinforce.model, os.path.join(dir_name, "best_model"))

            if save_videos:
                print("create video")
                create_video(os.path.join(video_dir, f"obs_{str(epoch+1).zfill(6)}.avi"), states, fps=120)

        #     writer.add_scalar('loss-val', running_loss_val, epoch)
        #     writer.add_scalar('loss-val-BCE', running_BCE_val, epoch)
        #     writer.add_scalar('loss-val-KLD', running_KLD_val, epoch)

        # writer.add_scalar('loss-train', running_loss, epoch)
        # writer.add_scalar('loss-train-BCE', running_BCE, epoch)
        # writer.add_scalar('loss-train-KLD', running_KLD, epoch)
        print(total_rewards[epoch], loss, loss_policy, loss_entropy, loss_repre)

    return total_rewards
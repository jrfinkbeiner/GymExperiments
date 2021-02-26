import os
from typing import Optional
import numpy as np
from multiprocessing import cpu_count, Pool
import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from GymExperiments.agents.rl.rlbase import RLBase
from GymExperiments.util.carracing_util import create_video

def train_sessions(
        num_epochs: int,
        agent: RLBase,
        dir_name: str,
        save_ith_epoch: int = 1,
        monitor: bool = True,
        init_expl: float = 1e-1,
        start_epoch: Optional[int] = 0,
    ):

    checkpoint_dir = os.path.join(dir_name, "checkpoints")

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if monitor:
        monitor_dir = os.path.join(dir_name, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
    tensorboard_dir = os.path.join(dir_name, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir, comment=dir_name)

    total_rewards = np.empty(num_epochs)

    smallest_loss = 1e32
    for epoch in tqdm.tqdm(range(start_epoch, start_epoch+num_epochs)):

        
        save = True if (epoch) % save_ith_epoch == 0 else False
        
        max_len = 1000 #3000+5*epoch # TODO
        expl = init_expl*np.exp(-0.00001*epoch) + 1e-4# TODO
        repre = 0.0001*np.exp(-0.001*epoch) + 1e-5 # TODO

        # with Pool(processes=1) as p:
        #     session = p.map(reinforce.generate_session, [max_len])

        # if save:
        #     # kwargs = {"monitor_dir": os.path.join(monitor_dir, f"epoch{str(epoch).zfill(6)}")}
        #     kwargs = {"directory": monitor_dir, "resume": False, "uid": epoch}
        # else:
        kwargs = {}
        states, rewards, actions = agent.generate_session(max_len=max_len, **kwargs)

        loss, loss_policy, loss_entropy, loss_repre = agent.train_step(states, actions, rewards, expl=expl, repre=repre)

        total_rewards[epoch-start_epoch] = sum(rewards)

        if save:
            save_dict = {
                'epoch': epoch+1,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': total_rewards[epoch-start_epoch],
                'loss': loss,
                'loss_policy': loss_policy,
                'loss_entropy': loss_entropy,
                'loss_repre': loss_repre,
            }
            if hasattr(agent, "lr_scheduler"):
                save_dict['lr_scheduler_state_dict'] = agent.lr_scheduler.state_dict()
            torch.save(save_dict, os.path.join(checkpoint_dir, f"checkpoint_{str(epoch+1).zfill(6)}.tar"))
            
            if smallest_loss > loss:
                torch.save(agent.model, os.path.join(dir_name, "best_model"))

            # if save_videos:
            #     print("create video")
            #     create_video(os.path.join(video_dir, f"obs_{str(epoch+1).zfill(6)}.avi"), states, fps=120)

        #     writer.add_scalar('loss-val', running_loss_val, epoch)
        #     writer.add_scalar('loss-val-BCE', running_BCE_val, epoch)
        #     writer.add_scalar('loss-val-KLD', running_KLD_val, epoch)

        # writer.add_scalar('loss-train', running_loss, epoch)
        # writer.add_scalar('loss-train-BCE', running_BCE, epoch)
        # writer.add_scalar('loss-train-KLD', running_KLD, epoch)
        writer.add_scalar('session-reward', total_rewards[epoch-start_epoch], epoch)
        print(total_rewards[epoch-start_epoch], loss, loss_policy, loss_entropy, loss_repre)

    return total_rewards
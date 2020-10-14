import os
import numpy as np
import torch
from cv2 import VideoWriter, VideoWriter_fourcc


def create_video(filename, observations, fps: int = 24):

    width = 96
    height = 96

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(filename, fourcc, float(fps), (width, height))

    for obervation in observations:
        frame = obervation.astype(np.uint8)
        video.write(frame)
    video.release()

    return video


def load_observations(observations_file):
    return np.load(observations_file)


def transform_observations_torch(observations):
    concat_observations = np.concatenate(observations, axis=0)
    concat_observations = concat_observations.transpose(0,3,1,2)
    return torch.tensor(concat_observations/ 255., dtype=torch.float32)
import os
import numpy as np
import gym
import matplotlib.pyplot as plt

from GymExperiments.architectures import instaniate_SimpleCNNVAE
from carracing_util import load_observations, create_video



def perform_rollout(env, rollout_length):
    env.reset()
    observations = np.empty((rollout_length, 96, 96, 3))
    for i in range(60):
        _, _, _, _ = env.step([np.random.random()-0.5, 1, 0])
    for iroll in range(rollout_length):
        observations[iroll], _, _, _ = env.step(env.action_space.sample())
    return observations

def gernerate_random_observations(env, num_rollouts, rollout_length):

    observations = []
    for i in range(num_rollouts):
        observations.append(perform_rollout(env, rollout_length))

    observations = np.array(observations)


    observations_dir = "./observations/startGas"
    os.makedirs(observations_dir, exist_ok=True)
    np.save(os.path.join(observations_dir, f"observations_startGas_nroll{num_rollouts}_lroll{rollout_length}_000001"), observations)

    print(observations.shape)



env = gym.make('CarRacing-v0')

print(env.action_space)
print(type(env.action_space))

observation = env.reset()
print(observation)
print(type(observation))
print(observation.shape)

gernerate_random_observations(env, num_rollouts=100, rollout_length=100)
# gernerate_random_observations(num_rollouts=100, rollout_length=100)

# observations = load_observations("./observations/startGas/observations_startGas_nroll100_lroll100_000001.npy")
# for i in range(len(observations)):
#     create_video(f'./observation_videos/startGas/observation_startGas_000001_{i}.avi', observations[i])


#     plt.figure()
#     plt.imshow(observations[i,0].astype(int))
# plt.show()

# for i in range(100):
    
#     action = env.action_space.sample()
#     print(action)

#     observation, _, _, _ = env.step(action) # take a random action

#     env.render()
#     if i > 50:
#         plt.figure()
#         plt.imshow(observation)
#         plt.show()

env.close()
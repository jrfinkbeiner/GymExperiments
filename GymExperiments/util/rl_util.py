import numpy as np
from typing import Callable

import torch

def generate_session(env, get_action: Callable, t_max=1000):
    """ 
    Play a full session with agent.
    Returns sequences of states, actions, and rewards.
    """
    # arrays to record session
    states, actions, rewards = [], [], []
    s = env.reset()

    for t in range(t_max):
        action = get_action(state)
        new_state, reward, done, _ = env.step(action)

        # record session history to train later
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state
        if done:
            break

    return states, actions, rewards


def get_cumulative_rewards(rewards, gamma=0.99):
    """
    Take a list of immediate rewards r(s,a) for the whole session 
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).
    
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    A simple way to compute cumulative rewards is to iterate from the last
    to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    num_rewards = len(rewards)
    gammas = np.cumprod(np.full((num_rewards-1), gamma))[::-1]
    cum_rewards = np.array(rewards, dtype=np.float64)

    for irew, reward in enumerate(rewards[1:][::-1]):
        cum_rewards[:-(irew+1)] += gammas[irew:]*reward

    return cum_rewards



def train_on_session(module, optimizer, states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """

    optimizer.zero_grad()

    # cast everything into torch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    # predict logits, probas and log-probas using an agent.
    logits = module(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    assert all(isinstance(v, torch.Tensor) for v in [logits, probs, log_probs]), \
        "please use compute using torch tensors and don't use predict_probs function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    log_probs_for_actions = torch.sum(
        log_probs * to_one_hot(actions, env.action_space.n), dim=1)
   
    # Compute loss here. Don't forgen entropy regularization with `entropy_coef` 
    # entropy = <YOUR CODE>
    # loss = <YOUR CODE>

    # print()
    entropy = -torch.mean(torch.exp(log_probs)*log_probs) * states.size()[0]
    # print(entropy)
    # print(log_probs_for_actions)
    # print(cumulative_returns)
    loss = -torch.mean(log_probs_for_actions*cumulative_returns) + entropy_coef*entropy
    # print(loss)

    # Gradient descent step
    loss.backward()
    optimizer.step()

    # technical: return session rewards to print them later
    return np.sum(rewards)
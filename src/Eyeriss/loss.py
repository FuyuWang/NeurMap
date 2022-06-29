import numpy as np
import torch

gamma = 0.9
EPISIOLON = 2**(-12)


def compute_policy_loss(rewards, log_probs):
    rewards = np.array(rewards)
    # rewards = rewards - rewards.min()
    # rewards = (rewards - rewards.mean()) / (rewards.std() + EPISIOLON)

    dis_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        dis_rewards.insert(0, R)
    dis_rewards = np.array(dis_rewards)
    # dis_rewards = (dis_rewards - dis_rewards.mean()) / (dis_rewards.std() + EPISIOLON)
    # print(dis_rewards)
    policy_loss = []
    for log_prob, r in zip(log_probs, dis_rewards):
        if len(log_prob) == 2:
            policy_loss.append(-log_prob[0] * r)
            policy_loss.append(-log_prob[1] * r)
        else:
            policy_loss.append(-log_prob * r)
    policy_loss = torch.stack(policy_loss).sum()

    return policy_loss

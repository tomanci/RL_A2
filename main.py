import gymnasium as gym
import numpy as np
import torch
from copy import deepcopy
from model import DQN
from e_greedy import epsilon_greedy

environment = gym.make("CartPole-v1")

MAX_EPOCHS = 1000
EPSILON_DECAY = 0.99

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train_qlearning(env, model, epochs=1000, epsilon=0.3, epsilon_decay=0.99, gamma=0.90):
    epoch_total_rewards = []
    for epoch in range(epochs):
        terminated = False
        s, info = env.reset()
        episode_reward = 0

        collected_batch_x, collected_batch_y = [], []
        batchsize = 2  # setting this to >500 will result in a whole episode being collected before the gradient update is done
        while not terminated:
            state_batch = torch.tensor(s)
            state_q_values = model.forward_pass_no_grad(state_batch).flatten().numpy()

            # epsilon-greedy action slection
            action = epsilon_greedy(state_q_values, epsilon)

            # do step in env
            sp, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # compute the target Q value we expect from the network
            target_q_values = state_q_values
            if terminated:
                target_q_values[action] = reward
            else:
                next_q_values = model.forward_pass_no_grad(sp)
                action_target_value = reward + gamma * torch.max(next_q_values)
                target_q_values[action] = action_target_value

            collected_batch_x.append(s)
            collected_batch_y.append(target_q_values)

            if len(collected_batch_x) == batchsize:
                collected_batch_x = torch.tensor(collected_batch_x)
                collected_batch_y = torch.tensor(collected_batch_y)
                train_model(collected_batch_x, collected_batch_y)
                collected_batch_x, collected_batch_y = [], []

            # internally advance to next state
            s = sp

            # truncated can be used to end the episode prematurely before a terminal state is reached. If true, the user needs to call env.reset.
            if truncated:
                break
        # apply on remaining part of batch
        if collected_batch_x:
            train_model(torch.tensor(collected_batch_x), torch.tensor(collected_batch_y))

        epsilon = max(epsilon * epsilon_decay, 0.01)
        # print("New Epsilon:", epsilon)
        epoch_total_rewards.append(episode_reward)

    print(f"Average reward: {np.mean(epoch_total_rewards)}")

def record_episode(env, q_net, epsilon, gamma=0.90):
    batch_size = 2
    epoch_rewards = []

    for epoch in range(MAX_EPOCHS):
        s, info = env.reset()
        terminated = False
        states = []
        targets = []
        step = 0
        rewards = 0

        while not terminated:
            q_sa = q_net.forward_pass_no_grad(s).numpy()
            a = epsilon_greedy(q_sa, epsilon)  # Select an action
            sp, reward, terminated, truncated, info = env.step(a)  # Take the step

            q_spa = q_net.forward_pass_no_grad(sp)

            if terminated:
                q_sa[a] = reward
            else:
                target = reward + gamma * torch.max(q_spa).item()
                q_sa[a] = target

            states.append(s)
            targets.append(q_sa)
            rewards += reward

            s = sp
            step += 1

            if truncated or terminated:
                break

            if step % batch_size == 0:
                model_input = torch.tensor(states)
                model_targets = torch.tensor(targets)
                loss = train_model(model_input, model_targets)
                #  print(f"Loss: {loss}")
                states = []
                targets = []

        if len(states) != 0:
            model_input = torch.tensor(states)
            model_targets = torch.tensor(targets)
            loss = train_model(model_input, model_targets)
            #  print(f"Loss: {loss}")

        epsilon = max(0.01, epsilon * EPSILON_DECAY)
        epoch_rewards.append(rewards)

    print(f"Average reward: {np.mean(epoch_rewards)}")
    return states, targets


def train_model(model_input, targets):
    q_net.optimizer.zero_grad()
    batch_output = q_net.forward_pass(model_input)

    loss = q_net.compute_loss(batch_output, targets)
    loss.backward()
    q_net.optimizer.step()

    return loss.item()


def train_qlearn(env, q_net, alpha=0.001, gamma=1.0, epsilon=0.3):
    max_length = 0
    for epoch in range(MAX_EPOCHS):
        states, targets = record_episode(env, q_net, epsilon, gamma)
        model_input = torch.tensor(states[:-1])
        targets = torch.tensor(targets)
        loss = train_model(model_input, targets)
        episode_length = len(states) - 1

        max_length = max(max_length, episode_length)

        epsilon = max(epsilon * EPSILON_DECAY, 0.01)

        if epoch % 100 == 99:
            print(f"Max length: {loss}")

    print(f"{max_length}")

    # Train a network

    return q_net


q_net = DQN()

q_net2 = deepcopy(q_net)

# record_episode(environment, q_net, epsilon=0.3)
train_qlearning(environment, q_net, epochs=MAX_EPOCHS)

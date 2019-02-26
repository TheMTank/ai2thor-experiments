"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

Contains the train code run by each A3C process on either Atari or AI2ThorEnv.
For initialisation, we set up the environment, seeds, shared model and optimizer.
In the main training loop, we always ensure the weights of the current model are equal to the
shared model. Then the algorithm interacts with the environment args.num_steps at a time,
i.e it sends an action to the env for each state and stores predicted values, rewards, log probs
and entropies to be used for loss calculation and backpropagation.
After args.num_steps has passed, we calculate advantages, value losses and policy losses using
Generalized Advantage Estimation (GAE) with the entropy loss added onto policy loss to encourage
exploration. Once these losses have been calculated, we add them all together, backprop to find all
gradients and then optimise with Adam and we go back to the start of the main training loop.

if natural_language is set to True, environment returns sentence instruction with image as state.
A3C_LSTM_GA model is used instead.
"""

import time
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from algorithms.a3c.env_atari import create_atari_env
from algorithms.a3c.model import ActorCritic, A3C_LSTM_GA
from gym_ai2thor.task_utils import unpack_state


def save_checkpoint(save_object, checkpoint_path, filename, is_best=False):
    fp = os.path.join(checkpoint_path, filename)
    torch.save(save_object, fp)
    print('Saved model to path: {}'.format(fp))
    if is_best:
        shutil.copyfile(fp, os.path.join(checkpoint_path, 'model_best.pth.tar'))

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, writer, optimizer=None):
    """
    Main A3C or A3C_LSTM_GA train loop and initialisation
    """
    train_start_time = time.time()
    torch.manual_seed(args.seed + rank)

    if args.atari:
        env = create_atari_env(args.atari_env_name)
    elif args.vizdoom:
        # many more dependencies required for VizDoom
        from algorithms.a3c.env_vizdoom import GroundingEnv

        env = GroundingEnv(args)
        env.game_init()
    else:
        env = AI2ThorEnv(config_file=args.config_file_path, config_dict=args.config_dict)
    env.seed(args.seed + rank)

    if env.task.task_has_language_instructions:
        model = A3C_LSTM_GA(env.observation_space.shape[0], env.action_space.n,
                            args.resolution, len(env.task.word_to_idx), args.max_episode_length)
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.resolution)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    # instruction_indices is None if task doesn't require language instructions
    state = env.reset()
    image_state, instruction_indices = unpack_state(state, env)

    done = True

    # monitoring
    avg_over_num_episodes = 10
    avg_episode_returns = []
    avg_episode_return, best_avg_episode_return = -np.inf, -np.inf
    episode_total_rewards_list = []
    all_rewards_in_episode = []
    p_losses = []
    v_losses = []

    total_length = args.total_length
    episode_number = args.episode_number
    episode_length = 0
    num_backprops = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        interaction_start_time = time.time()
        for step in range(args.num_steps):
            # save model every args.checkpoint_freq
            if rank == 0 and total_length > 0 and total_length % (args.checkpoint_freq //
                                                                  args.num_processes) == 0:
                fn = 'checkpoint_total_length_{}.pth.tar'.format(total_length)
                checkpoint_dict = {
                    'total_length': total_length,
                    'episode_number': episode_number,
                    'counter': counter.value,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                best_so_far = False
                if avg_episode_return > best_avg_episode_return:
                    best_so_far = True
                save_checkpoint(checkpoint_dict, args.checkpoint_path, fn, best_so_far)

            if not env.task.task_has_language_instructions:
                value, logit, (hx, cx) = model((image_state.unsqueeze(0).float(), (hx, cx)))
            else:
                tx = torch.from_numpy(np.array([episode_length])).long()
                value, logit, (hx, cx) = model((image_state.unsqueeze(0).float(),
                                                instruction_indices.long(),
                                                (tx, hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_int = action.numpy()[0][0].item()
            state, reward, done, _ = env.step(action_int)

            image_state, instruction_indices = unpack_state(state, env)
            episode_length += 1
            total_length += 1
            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
                # logging, benchmarking and saving stats
                total_reward_for_episode = sum(all_rewards_in_episode)
                episode_total_rewards_list.append(total_reward_for_episode)
                if len(episode_total_rewards_list) > avg_over_num_episodes:
                    avg_episode_return = sum(episode_total_rewards_list[-avg_over_num_episodes:]) / \
                                    len(episode_total_rewards_list[-avg_over_num_episodes:])
                    avg_episode_returns.append(avg_episode_return)
                    writer.add_scalar('avg_episode_returns', avg_episode_return, episode_number)
                all_rewards_in_episode = []

                print('Rank: {}. Episode {} Over. Total Length: {}. Total reward for episode: {:.4f}. '
                      .format(rank, episode_number, total_length, total_reward_for_episode))
                print('Rank: {}. Step no: {}. total length: {}'.format(rank, episode_length,
                                                                       total_length))
                print('Rank: {}. Total Length: {}. Counter across all processes: {}. '
                      'Total reward for episode: {}'.format(rank, total_length, counter.value,
                                                            total_reward_for_episode))
                if rank == 0:
                    writer.add_scalar('episode_lengths', episode_length, episode_number)
                    writer.add_scalar('episode_total_rewards', total_reward_for_episode, episode_number)
                    writer.add_image('Image', image_state, episode_number)

                # instruction_indices is None if task doesn't require language instructions
                state = env.reset()
                image_state, instruction_indices = unpack_state(state, env)

                episode_number += 1
                episode_length = 0

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            all_rewards_in_episode.append(reward)

            if done:
                break

        # No interaction with environment below
        # Backprop and optimisation
        if not done:  # to change last return to predicted value
            if not env.task.task_has_language_instructions:
                value, _, _ = model((image_state.unsqueeze(0).float(), (hx, cx)))
            else:
                value, _, _ = model((image_state.unsqueeze(0).float(),
                                     instruction_indices.long(),
                                     (tx, hx, cx)))
            R = value.detach()
        else:
            # todo how did this ever work before when it didn't work before? was the value spike loss bug?
            R = 0.0

        # todo where is this set?
        values.append(R)  # if episode is terminal, 0 reward. Otherwise, predicted value
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                          args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        # benchmarking and general info
        num_backprops += 1
        writer.add_scalar('policy_loss', policy_loss.item(), num_backprops)
        writer.add_scalar('value_loss', value_loss.item(), num_backprops)

        p_losses.append(policy_loss.item())
        v_losses.append(value_loss.item())

        if len(p_losses) > 1000:  # 1000 * 20 (args.num_steps default) = every 20000 steps
            print(" ".join([
                "Training thread: {}".format(rank),
                "Num backprops: {}".format(num_backprops),
                "Avg policy loss: {}".format(np.mean(p_losses)),
                "Avg value loss: {}".format(np.mean(v_losses))]))
            p_losses = []
            v_losses = []

        if rank == 0 and args.verbose_num_steps:
            print('Step no: {}. total length: {}. Time elapsed: {}m'.format(
                episode_length,
                total_length,
                round((time.time() - train_start_time) / 60.0, 3)
                ))

            if num_backprops % 100 == 0:
                print('Time taken for args.steps ({}): {}'.format(args.num_steps,
                       round(time.time() - interaction_start_time, 3)))

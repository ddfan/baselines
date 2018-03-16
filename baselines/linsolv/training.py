import os
import time
from collections import deque
import pickle
from copy import copy
from progress.bar import Bar

from baselines.linsolv.linsolv import LINSOLV
import baselines.common.tf_util as U
from baselines.common.mpi_moments import mpi_moments

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, save_policies=False, policy_save_interval=10,
    action_process=None, use_linsolv=False, actorcritic=None, action_range=(-1., 1.)):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}')

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = LINSOLV(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, use_linsolv=use_linsolv, actorcritic=actorcritic)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver(max_to_keep=9999999)
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()
        best_avg_eval_return=-float("inf")
        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        if use_linsolv:
            action=np.zeros(env.action_space.shape[-1])
            epoch_actions_hat = [] 
            if eval_env is not None:
                eval_action_process=copy(action_process)

        for epoch in range(nb_epochs):
            bar = Bar('Training', max=nb_epoch_cycles*nb_rollout_steps)
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    bar.next()
                    # Predict next action.
                    if use_linsolv:
                        #we have to feed in size (batch_size,obssize), (batch_size, action_size) because the tensor sizes are fixed.  bad tensorflow.
                        obs_rep=np.tile(obs,(batch_size,1))
                        action_rep=np.tile(action,(batch_size,1))
                        action_hat_rep, q_rep = agent.pi(obs_rep, action_rep, apply_noise=True, compute_Q=True)
                        action_hat=action_hat_rep[0,:]
                        q=q_rep[0,:]
                        action1 = action_process(action_hat, apply_noise=False)
                        action1 = np.clip(action1, action_range[0],action_range[1])
                    else:
                        action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    if use_linsolv:
                        epoch_actions_hat.append(action_hat)
                        agent.store_transition(obs, action, r, new_obs, done, action1)
                        action=action1
                    else:
                        agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()
                        if use_linsolv:
                            action_process.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if use_linsolv:
                    eval_action=np.zeros(env.action_space.shape[-1])
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        if use_linsolv:
                            eval_obs_rep=np.tile(eval_obs,(batch_size,1))
                            eval_action_rep=np.tile(eval_action,(batch_size,1))
                            eval_action_hat_rep, q_rep = agent.pi(eval_obs_rep, eval_action_rep, apply_noise=False, compute_Q=True)
                            eval_action_hat=action_hat_rep[0,:]
                            eval_q=q_rep[0,:]
                            eval_action1 = action_process(action_hat, apply_noise=False)
                            eval_action1 = np.clip(action1, action_range[0],action_range[1])
                        else:
                            eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if use_linsolv:
                            eval_action=eval_action1
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.
                            if use_linsolv:
                                eval_action_process.reset()

            bar.finish()
            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            if use_linsolv:
                combined_stats['rollout/actions_hat_mean'] = np.mean(epoch_actions_hat)
                combined_stats['rollout/actions_hat_std'] = np.std(epoch_actions_hat)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = np.mean(eval_qs)
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

            # save the policy if it's better than the previous ones
            avg_eval_return = mpi_average(combined_stats['eval/return'])
            if rank == 0 and avg_eval_return >= best_avg_eval_return and save_policies:
                best_avg_eval_return = avg_eval_return
                logger.info('New best return: {}. Saving policy to {} ...'.format(best_avg_eval_return, best_policy_path))
                saver.save(sess,best_policy_path)
                saver.save(sess,latest_policy_path)
            if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
                policy_path = periodic_policy_path.format(epoch)
                logger.info('Saving periodic policy to {} ...'.format(policy_path))
                saver.save(sess,policy_path)


def play(env, nb_epochs, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_rollout_steps, batch_size, memory,
    tau=0.01, restore_path=None):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    agent = LINSOLV(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()

        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()
        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        #load weights
        assert restore_path is not None
        saver.restore(sess,restore_path)

        for epoch in range(nb_epochs):
            # Perform rollouts.
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
                assert action.shape == env.action_space.shape

                # Execute next action.
                if rank == 0 and render:
                    env.render()
                assert max_action.shape == action.shape
                new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_step += 1

                epoch_actions.append(action)
                epoch_qs.append(q)
                obs = new_obs

                if done:
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    episode_reward = 0.
                    episode_step = 0
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset()
                    obs = env.reset()

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            combined_stats={}
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                print(key,combined_stats[key])
            print("------------------------------------------")

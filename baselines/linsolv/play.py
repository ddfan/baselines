# from mujoco_py import load_model_from_path, MjSim, MjViewer
# from mujoco_py.modder import TextureModder
# import os
# model = load_model_from_path("/home/aeuser/Documents/mujoco-py/xmls/fetch/main.xml")
# sim = MjSim(model)
# viewer = MjViewer(sim)


import argparse
import time
import os
import logging
# from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.linsolv.training as training
from baselines.linsolv.models import Actor, Critic
from baselines.linsolv.memory import Memory
from baselines.linsolv.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI

def run(env_id, seed, noise_type, layer_norm, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()

    # Create envs.
    env = gym.make(env_id)
    eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e2), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    if rank == 0:
        start_time = time.time()
    training.play(env=env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Swimmer-v2')
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=True)
    boolean_flag(parser, 'normalize-returns', default=True)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=1)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=10)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-rollout-steps', type=int, default=1000)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='ou_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--restore-path', type=str, default=None)  # choices are adaptive-param_xx, ou_xx, normal_xx, none

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    run(**args)

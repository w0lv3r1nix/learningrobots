import argparse, pickle, os, random
import numpy as np
from collections import deque

# network can be passed as input
parser = argparse.ArgumentParser(description='Train deepRL agent for cam-shift toy-problem.')
parser.add_argument('network', metavar='agent', type=str, nargs=1,
                    help='network to train.' )
args = parser.parse_args()

# determine run nr and set seed as run_nr
log_dirs = os.listdir('./logs/%sAgent/'%(args.network[0]))
run_nr = len(log_dirs) + 1
os.mkdir('./logs/%sAgent/run_%i/'%(args.network[0],run_nr))
np.random.seed(run_nr)
random.seed(run_nr)

import tensorflow as tf
tf.set_random_seed(run_nr)
from environment import *
from agent import *
from target import *

# initialize settings for each run
n_epochs = 1500
max_steps = 100
sample_rate = 6
world_dims = [400,600]
agent_fov = [60,80,1]
action_dims = [3,3]
memory_size = 2048
epsilon = [1,.001,.1]
tau = 100

# after how many training episodes to evaluate how many times, how often to save logs/weights
eval_freq = 1
n_eval = 5
save_freq = 100

# initialize Target
target = Target(shape=render_target(),static=False)

# initialize Agent
if args.network[0] == 'DRQN':
    agent = DRQNAgent(agent_fov,action_dims,sample_rate=sample_rate,memory_size=memory_size,epsilon=epsilon)
elif args.network[0] == 'EffDRQN':
    agent = EffDRQNAgent(agent_fov,action_dims,sample_rate=sample_rate,memory_size=memory_size,epsilon=epsilon)

# set up Environment
env = Environment(world_dims, agent, target, max_steps=max_steps)

# buffers for storing the mos frequent states (and actions)
sample_buffer = deque()
action_buffer = deque()

def play_episode(evaluate=False):
    """
    'Plays' one episode of the simulation.
    The parameter 'evaluate' determines whether it is a training or evaluation episode.
    Only for training, the epsilon-greedy policy is applied, each experience is stored
    and the agent is trained at the endself.
    For evaluation, metrics are logged and returned at the end.
    """
    # reset everything
    env.reset()
    sample_buffer.clear()
    action_buffer.clear()
    done = False

    if evaluate:
        cumulative_reward = 0
        qs, hits, hit_last, ahead = [],[],[],[] # lists to store metrics

    # fill buffers with the same state at the beginning
    for s in range(sample_rate):
        sample_buffer.append(env.get_observation())
        action_buffer.append(np.array([0,0]))
    # reshape state and previous actions to be fed into network
    state = np.vstack(np.expand_dims(sample_buffer,0))
    action_state = np.vstack(np.expand_dims(action_buffer,0)) - 1 

    # i.e. until episode terminates
    while not done:
        if agent.ID == 'EffDRQNAgent':
            action,action_values = agent.choose_action([state,action_state],exploration=not evaluate)
        else:
            action,action_values = agent.choose_action(state,exploration=not evaluate)

        observation, reward, done, info = env.step(action)

        # update buffers
        sample_buffer.popleft()
        sample_buffer.append(observation)
        action_buffer.popleft()
        action_buffer.append(action)

        # reshape state and previous actions to be fed into network
        next_state = np.vstack(np.expand_dims(sample_buffer,0))
        next_action_state = np.vstack(np.expand_dims(action_buffer,0)) - 1

        if evaluate:
            # collect metrics
            cumulative_reward += reward
            qs.append(action_values)
            hits.append(info['hit_on_target'])
            hit_last.append(info['hit_last_target'])
            ahead.append(info['ahead'])
        else:
            # store tuple in experience replay memory
            if agent.ID == 'EffDRQNAgent':
                agent.memory.remember(([state,action_state],action,reward,[next_state,next_action_state],done))
            else:
                agent.memory.remember((state,action,reward,next_state,done))

        # update states
        state = next_state
        action_state = next_action_state

    if evaluate:
        return {
            'final_distance': env.get_distance(),
            'cumulative_reward': cumulative_reward,
            'n_steps': env.n_steps,
            'q_means': np.mean(qs),
            'q_stds': np.std(qs),
            'hits_on_target': np.sum(hits),
            'hits_on_last_target': np.sum(hit_last),
            'ahead': np.mean(ahead),
            'region_hits': np.sum(np.abs(ahead))/env.n_steps
        }
    else:
        # if in training mode, lower epsilon
        agent.decay_epsilon()
        if (episode%tau) == 0:
            agent.update_target_model()
        if agent.memory.current_size > 200:
            agent.replay(32)

    # function end

log_dict = {
    'final_distance': [], # distance between target and agent after the final steps
    'cumulative_reward': [], # cumulative_reward over episode
    'n_steps': [], # number of simulation steps (lower than 100 if target left field of view)
    'q_means': [], # mean q values
    'q_stds': [], # q value standard deviation
    'hits_on_target': [], # number of direct hits on target (distance == 0)
    'hits_on_last_target': [], # number of direct hits on last target position
    'ahead': [], # whether the agent was mostly ahead (positive) or behind (negative) the target, when in target radius [-1,1]
    'region_hits': [], # percentage of steps of agent inside target radius
    'run_nr': run_nr, # current run nr (identifier)
    'target_id': target.ID, # target ID, not really used, since always moving
    'agent_id': agent.ID, # agent ID, e.g. DRQNAgent
    'max_steps': max_steps, # maximum number of simulation steps
    'n_epochs': n_epochs, # number of episodes played in run
}

# main loop for run
for episode in range(1,n_epochs+1):
    play_episode() # training episode

    if (episode%eval_freq) == 0:
        # evaluate n_eval times
        dicts = []
        for i in range(n_eval):
            dicts.append(play_episode(evaluate=True))
        # average over evaluation episodes
        for m in dicts[-1].keys():
            metric = 0
            for e in range(n_eval):
                metric += dicts[e][m]
            log_dict[m].append(metric/n_eval)

        print('%s - episode %4i' % (agent.ID,episode),
              '| epsilon: %1.2f' % agent.epsilon,
              '| n_steps: %3i' % log_dict['n_steps'][-1],
              '| final_distance: %3i' % log_dict['final_distance'][-1].astype(int),
              '| total_reward: %3i' % log_dict['cumulative_reward'][-1],
              '| mean_q: %.2f [%.2f]' % (log_dict['q_means'][-1],log_dict['q_stds'][-1]))

    if (episode%save_freq) == 0:
        # save model, weights, logs, gifs of last episode (comment in)
        agent.model.save_weights('./logs/%s/run_%i/weights.h5' %(agent.ID,run_nr))
        j = agent.model.to_json()
        with open('./logs/%s/run_%i/model.json' %(agent.ID,run_nr),'w') as f:
            f.write(j)
        with open('./logs/%s/run_%i/log.pkl' % (agent.ID,run_nr),'wb') as f:
            pickle.dump(log_dict,f)
        # env.generate_world_gif('./logs/%s/run_%i/%i'%(agent.ID,run_nr,episode))
        # env.generate_state_gif('./logs/%s/run_%i/%i'%(agent.ID,run_nr,episode))

from argparse import ArgumentParser
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from super_mario.dqn_agent import DQNAgent
from super_mario.wrappers import wrap_environment
import time

import sys
sys.path.append('C:/Users/notfu/Desktop/super_mario')
ENV_NAME = 'SuperMarioBros-1-3-v0'
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 100000
GAMMA = 0.99
INITIAL_LEARNING = 10000
# INITIAL_LEARNING = 10
LEARNING_RATE = 1e-4
MEMORY_CAPACITY = 20000
NUM_EPISODES = 5000
TARGET_UPDATE_FREQUENCY = 1000


def train(env, args, agent):
    for episode_idx in range(args.num_episodes):
        episode_reward = 0.0
        state = env.reset()

        while True:
            action = agent.act(state, episode_idx)
            #if args.render:
            #env.render()
            #time.sleep(0.1)
            #print('\n')
            next_state, reward, done, stats = env.step(action)
            
            agent.process(episode_idx, state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                print(f'{episode_idx}: {episode_reward}')
                # plot_rewards(episode_reward)
                if stats["flag_get"] == True:
                    print('Succeed!!!')
                    break
                    return
                break
                
class Range:
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def __eq__(self, input_num):
        return self._start <= input_num <= self._end


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--buffer-capacity', type=int, default=MEMORY_CAPACITY)
    parser.add_argument('--env-name', type=str, default=ENV_NAME)
    parser.add_argument('--epsilon-start', type=float, choices=[Range(0.0, 1.0)], default=EPSILON_START)
    parser.add_argument('--epsilon-final', type=float, choices=[Range(0.0, 1.0)], default=EPSILON_FINAL)
    parser.add_argument('--epsilon-decay', type=int, default=EPSILON_DECAY)
    parser.add_argument('--gamma', type=float, choices=[Range(0.0, 1.0)], default=GAMMA)
    parser.add_argument('--initial-learning', type=int, default=INITIAL_LEARNING)
    parser.add_argument('--learning-rate', type=int, default=LEARNING_RATE)
    parser.add_argument('--num-episodes', type=int, default=NUM_EPISODES)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--target-update-frequency', type=int, default=TARGET_UPDATE_FREQUENCY)

    return parser.parse_args()


def main():
    args = parse_args()
    env = wrap_environment(args.env_name, SIMPLE_MOVEMENT)
    agent = DQNAgent(env,
                     buffer_capacity=args.buffer_capacity,
                     epsilon_start=args.epsilon_start,
                     epsilon_final=args.epsilon_final,
                     epsilon_decay=args.epsilon_decay,
                     lr=args.learning_rate,
                     initial_learning=args.initial_learning,
                     gamma=args.gamma,
                     target_update_frequency=args.target_update_frequency,
                     batch_size = args.batch_size)
    
    train(env, args, agent)
    env.close()


if __name__ == '__main__':
    main()

from typing import Optional, Iterable
import cv2

from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
from keras.engine.saving import save_model


class AgentConf:
    def __init__(self):
        self.n_neurons = [32, 32]
        self.activations = ['relu', 'relu', 'linear']
        self.episodes = 2000
        self.epsilon_stop_episode = 1500
        self.mem_size = 25000
        self.discount = 0.95
        self.replay_start_size = 5000
        self.batch_size = 1024
        self.epochs = 1
        self.render_every = None
        self.train_every = 1
        self.log_every = 10
        self.max_steps: Optional[int] = 10000


# Run dqn with Tetris
def dqn(ac: AgentConf):
    env = Tetris()

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=ac.n_neurons, activations=ac.activations,
                     epsilon_stop_episode=ac.epsilon_stop_episode, mem_size=ac.mem_size,
                     discount=ac.discount, replay_start_size=ac.replay_start_size)

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'logs/tetris-{timestamp_str}-nn={str(ac.n_neurons)}-mem={ac.mem_size}' \
        f'-bs={ac.batch_size}-e={ac.epochs}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    episodes_wrapped: Iterable[int] = tqdm(range(ac.episodes))
    for episode in episodes_wrapped:
        current_state = env.reset()
        done = False
        steps = 0

        # update render flag
        render = True if ac.render_every and episode % ac.render_every == 0 else False

        # game
        while not done and (not ac.max_steps or steps < ac.max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            # find the action, that corresponds to the best state
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.hard_drop([best_action[0], 0], best_action[1], render=render)

            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        # just return score
        scores.append(env.get_game_score())

        # train
        if episode % ac.train_every == 0:
            # n = len(agent.memory)
            # print(f" agent.memory.len: {n}")
            agent.train(batch_size=ac.batch_size, epochs=ac.epochs)

        # logs
        if ac.log_every and episode and episode % ac.log_every == 0:
            avg_score = mean(scores[-ac.log_every:])
            min_score = min(scores[-ac.log_every:])
            max_score = max(scores[-ac.log_every:])
            log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)
    # save_model
    save_model(agent.model, f'{log_dir}/model.hdf', overwrite=True, include_optimizer=True)


def enumerate_dqn():
    for bs in [256, 512, 1024]:
        for ms in [5000, 10_000, 15_000, 20_000, 25_000]:
            agent_conf = AgentConf()
            agent_conf.batch_size = bs
            agent_conf.mem_size = ms
            dqn(agent_conf)


if __name__ == "__main__":
    enumerate_dqn()
    exit(0)

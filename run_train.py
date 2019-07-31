from typing import Optional, Iterable

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
        self.batch_size = 512
        self.activations = ['relu', 'relu', 'linear']
        self.episodes = 2000
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_stop_episode = 1600
        self.mem_size = 25000
        self.discount = 0.95
        self.replay_start_size = 2000
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
                     epsilon=ac.epsilon, epsilon_min=ac.epsilon_min, epsilon_stop_episode=ac.epsilon_stop_episode,
                     mem_size=ac.mem_size, discount=ac.discount, replay_start_size=ac.replay_start_size)

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    # conf.mem_size = mem_size
    # conf.epochs = epochs
    # conf.epsilon_stop_episode = epsilon_stop_episode
    # conf.discount = discount
    log_dir = f'logs/tetris-{timestamp_str}-ms{ac.mem_size}-e{ac.epochs}-ese{ac.epsilon_stop_episode}-d{ac.discount}'
    log = CustomTensorBoard(log_dir=log_dir)

    print(f"AGENT_CONF = {log_dir}")

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
    for mem_size in [10_000, 15_000, 20_000, 25_000]:
        for epochs in [1, 2, 3]:
            for epsilon_stop_episode in [1600, 1800, 2000]:
                for discount in [0.95, 0.97, 0.99]:
                    conf = AgentConf()
                    conf.mem_size = mem_size
                    conf.epochs = epochs
                    conf.epsilon_stop_episode = epsilon_stop_episode
                    conf.discount = discount
                    dqn(conf)


if __name__ == "__main__":
    enumerate_dqn()
    exit(0)

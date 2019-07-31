import types
from typing import Optional, Iterable

import cv2

from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
from keras.engine.saving import save_model

agent_conf = types.SimpleNamespace()
agent_conf.n_neurons = [32, 32]
agent_conf.activations = ['relu', 'relu', 'linear']
agent_conf.episodes = 2000
agent_conf.epsilon_stop_episode = 1500
agent_conf.mem_size = 15000
agent_conf.discount = 0.95
agent_conf.replay_start_size = 2500
agent_conf.batch_size = 512
agent_conf.epochs = 1
agent_conf.render_every = None
agent_conf.train_every = 1
agent_conf.log_every = 10
agent_conf.max_steps: Optional[int] = 1000


# Run dqn with Tetris
def dqn():
    env = Tetris()

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=agent_conf.n_neurons, activations=agent_conf.activations,
                     epsilon_stop_episode=agent_conf.epsilon_stop_episode, mem_size=agent_conf.mem_size,
                     discount=agent_conf.discount, replay_start_size=agent_conf.replay_start_size)

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'logs/tetris-{timestamp_str}-nn={str(agent_conf.n_neurons)}-mem={agent_conf.mem_size}' \
        f'-bs={agent_conf.batch_size}-e={agent_conf.epochs}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    episodes_wrapped: Iterable[int] = tqdm(range(agent_conf.episodes))
    for episode in episodes_wrapped:
        current_state = env.reset()
        done = False
        steps = 0

        # update render flag
        render = True if agent_conf.render_every and episode % agent_conf.render_every == 0 else False

        # game
        while not done and (not agent_conf.max_steps or steps < agent_conf.max_steps):
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
        if episode % agent_conf.train_every == 0:
            # n = len(agent.memory)
            # print(f" agent.memory.len: {n}")
            agent.train(batch_size=agent_conf.batch_size, epochs=agent_conf.epochs)

        # logs
        if agent_conf.log_every and episode and episode % agent_conf.log_every == 0:
            avg_score = mean(scores[-agent_conf.log_every:])
            min_score = min(scores[-agent_conf.log_every:])
            max_score = max(scores[-agent_conf.log_every:])
            log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)
    # save_model
    save_model(agent.model, f'{log_dir}/model.hdf', overwrite=True, include_optimizer=True)


if __name__ == "__main__":
    dqn()
    cv2.destroyAllWindows()
    exit(0)

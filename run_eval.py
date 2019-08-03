from typing import List

import cv2
from dqn_agent import DQNAgent
from tetris import Tetris
from run_train import AgentConf
from keras.engine.saving import load_model


def run_eval(dir_name: str, episodes: int = 100) -> List[int]:
    agent_conf = AgentConf()
    agent_conf.render_every = None
    render = True if agent_conf.render_every is not None else False
    env = Tetris()
    agent = DQNAgent(env.get_state_size(),
                     n_neurons=agent_conf.n_neurons, activations=agent_conf.activations,
                     epsilon_stop_episode=agent_conf.epsilon_stop_episode, mem_size=agent_conf.mem_size,
                     discount=agent_conf.discount, replay_start_size=agent_conf.replay_start_size)

    # timestamp_str = "20190730-165821"
    # log_dir = f'logs/tetris-nn={str(agent_conf.n_neurons)}-mem={agent_conf.mem_size}' \
    #     f'-bs={agent_conf.batch_size}-e={agent_conf.epochs}-{timestamp_str}'

    # tetris-20190731-221411-nn=[32, 32]-mem=25000-bs=512-e=1 good

    log_dir = 'logs/' + dir_name

    # load_model
    agent.model = load_model(f'{log_dir}/model.hdf')
    agent.epsilon = 0
    scores = []
    for episode in range(episodes):
        env.reset()
        done = False

        while not done:
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            # find the action, that corresponds to the best state
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break
            _, done = env.hard_drop([best_action[0], 0], best_action[1], render=render)
        scores.append(env.score)
        # print results at the end of the episode
        print(f'episode {episode} => {env.score}')
    return scores


def enumerate_run_eval():
    dirs = [
        'tetris-20190731-221411-nn=[32, 32]-mem=25000-bs=512-e=1',
        'tetris-20190801-030955-ms10000-e1-ese1600-d0.95',
        'tetris-20190801-032349-ms10000-e1-ese1600-d0.97',
        'tetris-20190801-034325-ms10000-e1-ese1600-d0.99',
        'tetris-20190801-040623-ms10000-e1-ese1800-d0.95',
        'tetris-20190801-042029-ms10000-e1-ese1800-d0.97',
        'tetris-20190801-043905-ms10000-e1-ese1800-d0.99',
        'tetris-20190801-045427-ms10000-e1-ese2000-d0.95',
        'tetris-20190801-050554-ms10000-e1-ese2000-d0.97',
        'tetris-20190801-051743-ms10000-e1-ese2000-d0.99',
        'tetris-20190801-053018-ms10000-e2-ese1600-d0.95',
        'tetris-20190801-054659-ms10000-e2-ese1600-d0.97',
        'tetris-20190801-060613-ms10000-e2-ese1600-d0.99',
        'tetris-20190801-064129-ms10000-e2-ese1800-d0.95',
        'tetris-20190801-065640-ms10000-e2-ese1800-d0.97',
        'tetris-20190801-071415-ms10000-e2-ese1800-d0.99',
        'tetris-20190801-073944-ms10000-e2-ese2000-d0.95',
        'tetris-20190801-075245-ms10000-e2-ese2000-d0.97',
        'tetris-20190801-080634-ms10000-e2-ese2000-d0.99',
        'tetris-20190801-082115-ms10000-e3-ese1600-d0.95',
        'tetris-20190801-084132-ms10000-e3-ese1600-d0.97',
        'tetris-20190801-090258-ms10000-e3-ese1600-d0.99',
        'tetris-20190801-094559-ms10000-e3-ese1800-d0.95',
        'tetris-20190801-100216-ms10000-e3-ese1800-d0.97',
        'tetris-20190801-102048-ms10000-e3-ese1800-d0.99',
        'tetris-20190801-104347-ms10000-e3-ese2000-d0.95',
        'tetris-20190801-105832-ms10000-e3-ese2000-d0.97',
        'tetris-20190801-111315-ms10000-e3-ese2000-d0.99',
        'tetris-20190801-112943-ms15000-e1-ese1600-d0.95',
        'tetris-20190801-115322-ms15000-e1-ese1600-d0.97',
        'tetris-20190801-121703-ms15000-e1-ese1600-d0.99',
        'tetris-20190801-124412-ms15000-e1-ese1800-d0.95',
        'tetris-20190801-130404-ms15000-e1-ese1800-d0.97',
        'tetris-20190801-132402-ms15000-e1-ese1800-d0.99',
        'tetris-20190801-135213-ms15000-e1-ese2000-d0.95',
        'tetris-20190801-140930-ms15000-e1-ese2000-d0.97',
        'tetris-20190801-142650-ms15000-e1-ese2000-d0.99',
        'tetris-20190801-144630-ms15000-e2-ese1600-d0.95',
        'tetris-20190801-151414-ms15000-e2-ese1600-d0.97',
        'tetris-20190801-154508-ms15000-e2-ese1600-d0.99',
        'tetris-20190801-162448-ms15000-e2-ese1800-d0.95',
        'tetris-20190801-164759-ms15000-e2-ese1800-d0.97',
        'tetris-20190801-171737-ms15000-e2-ese1800-d0.99',
        'tetris-20190801-175405-ms15000-e2-ese2000-d0.95',
        'tetris-20190801-181401-ms15000-e2-ese2000-d0.97',
        'tetris-20190801-183221-ms15000-e2-ese2000-d0.99',
        'tetris-20190801-185516-ms15000-e3-ese1600-d0.95',
        'tetris-20190801-192041-ms15000-e3-ese1600-d0.97',
        'tetris-20190801-195254-ms15000-e3-ese1600-d0.99',
        'tetris-20190801-203902-ms15000-e3-ese1800-d0.95',
        'tetris-20190801-210422-ms15000-e3-ese1800-d0.97',
        'tetris-20190801-213112-ms15000-e3-ese1800-d0.99',
        'tetris-20190801-220527-ms15000-e3-ese2000-d0.95',
        'tetris-20190801-222633-ms15000-e3-ese2000-d0.97',
        'tetris-20190801-224741-ms15000-e3-ese2000-d0.99',
        'tetris-20190801-231020-ms20000-e1-ese1600-d0.95',
        'tetris-20190801-234308-ms20000-e1-ese1600-d0.97',
        'tetris-20190802-002631-ms20000-e1-ese1600-d0.99',
        'tetris-20190802-015901-ms20000-e1-ese1800-d0.95',
        'tetris-20190802-022640-ms20000-e1-ese1800-d0.97',
        'tetris-20190802-025651-ms20000-e1-ese1800-d0.99',
        'tetris-20190802-033219-ms20000-e1-ese2000-d0.95',
        'tetris-20190802-035323-ms20000-e1-ese2000-d0.97',
        'tetris-20190802-041643-ms20000-e1-ese2000-d0.99',
        'tetris-20190802-043948-ms20000-e2-ese1600-d0.95',
        'tetris-20190802-051535-ms20000-e2-ese1600-d0.97',
        'tetris-20190802-055943-ms20000-e2-ese1600-d0.99',
        'tetris-20190802-072159-ms20000-e2-ese1800-d0.95',
        'tetris-20190802-074955-ms20000-e2-ese1800-d0.97',
        'tetris-20190802-082058-ms20000-e2-ese1800-d0.99',
        'tetris-20190802-085812-ms20000-e2-ese2000-d0.95',
        'tetris-20190802-092212-ms20000-e2-ese2000-d0.97',
        'tetris-20190802-094806-ms20000-e2-ese2000-d0.99',
        'tetris-20190802-101511-ms20000-e3-ese1600-d0.95',
        'tetris-20190802-105110-ms20000-e3-ese1600-d0.97',
        'tetris-20190802-113144-ms20000-e3-ese1600-d0.99',
        'tetris-20190802-122834-ms20000-e3-ese1800-d0.95',
        'tetris-20190802-125828-ms20000-e3-ese1800-d0.97',
        'tetris-20190802-133625-ms20000-e3-ese1800-d0.99',
        'tetris-20190802-142056-ms20000-e3-ese2000-d0.95',
        'tetris-20190802-144737-ms20000-e3-ese2000-d0.97',
        'tetris-20190802-151546-ms20000-e3-ese2000-d0.99',
        'tetris-20190802-154434-ms25000-e1-ese1600-d0.95',
        'tetris-20190802-161557-ms25000-e1-ese1600-d0.97',
        'tetris-20190802-171741-ms25000-e1-ese1600-d0.99',
        'tetris-20190802-183634-ms25000-e1-ese1800-d0.95',
        'tetris-20190802-191501-ms25000-e1-ese1800-d0.97',
        'tetris-20190802-200423-ms25000-e1-ese1800-d0.99',
        'tetris-20190802-205724-ms25000-e1-ese2000-d0.95',
        'tetris-20190802-213757-ms25000-e1-ese2000-d0.97',
        'tetris-20190802-221032-ms25000-e1-ese2000-d0.99',
        'tetris-20190802-223912-ms25000-e2-ese1600-d0.95',
        'tetris-20190802-232124-ms25000-e2-ese1600-d0.97',
        'tetris-20190803-001316-ms25000-e2-ese1600-d0.99',
        'tetris-20190803-013408-ms25000-e2-ese1800-d0.95',
        'tetris-20190803-021108-ms25000-e2-ese1800-d0.97',
        'tetris-20190803-024251-ms25000-e2-ese1800-d0.99',
        'tetris-20190803-032347-ms25000-e2-ese2000-d0.95',
        'tetris-20190803-035111-ms25000-e2-ese2000-d0.97',
        'tetris-20190803-042134-ms25000-e2-ese2000-d0.99',
        'tetris-20190803-045249-ms25000-e3-ese1600-d0.95',
        'tetris-20190803-053820-ms25000-e3-ese1600-d0.97',
        'tetris-20190803-062855-ms25000-e3-ese1600-d0.99',
        'tetris-20190803-074816-ms25000-e3-ese1800-d0.95',
        'tetris-20190803-082758-ms25000-e3-ese1800-d0.97',
        'tetris-20190803-090936-ms25000-e3-ese1800-d0.99',
        'tetris-20190803-100105-ms25000-e3-ese2000-d0.95',
        'tetris-20190803-103310-ms25000-e3-ese2000-d0.97',
        'tetris-20190803-110555-ms25000-e3-ese2000-d0.99',
    ]
    max_scores = []
    for d in dirs:
        print(f"Evaluating dir '{d}'")
        scores = run_eval(d, episodes=128)
        max_scores.append((d, max(scores)))

    max_scores.sort(key=lambda t: t[1], reverse=True)
    for k, v in max_scores:
        print(f"{v}\t{k}")


if __name__ == "__main__":
    enumerate_run_eval()
    cv2.destroyAllWindows()
    exit(0)

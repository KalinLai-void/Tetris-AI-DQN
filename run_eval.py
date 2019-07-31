import cv2
from dqn_agent import DQNAgent
from tetris import Tetris
from run import AgentConf
from keras.engine.saving import load_model


def run_eval(dir_name):
    episodes = 100
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
        # print results at the end of the episode
        print(f'episode {episode} => {env.score}')


def enumerate_run_eval():
    dirs = [
        # 'tetris-20190731-172645-nn=[32, 32]-mem=15000-bs=512-e=1',
        # 'tetris-20190731-180835-nn=[32, 32]-mem=25000-bs=512-e=1',
        # 'tetris-20190731-184008-nn=[32, 32]-mem=25000-bs=64-e=1',
        # 'tetris-20190731-190022-nn=[32, 32]-mem=25000-bs=128-e=1',
        # 'tetris-20190731-200804-nn=[32, 32]-mem=5000-bs=256-e=1',
        # 'tetris-20190731-201826-nn=[32, 32]-mem=10000-bs=256-e=1',
        # 'tetris-20190731-203105-nn=[32, 32]-mem=15000-bs=256-e=1',
        # 'tetris-20190731-204235-nn=[32, 32]-mem=20000-bs=256-e=1',
        # 'tetris-20190731-205614-nn=[32, 32]-mem=25000-bs=256-e=1',
        # 'tetris-20190731-210946-nn=[32, 32]-mem=5000-bs=512-e=1',
        # 'tetris-20190731-212322-nn=[32, 32]-mem=10000-bs=512-e=1',
        # 'tetris-20190731-213758-nn=[32, 32]-mem=15000-bs=512-e=1',
        # 'tetris-20190731-215701-nn=[32, 32]-mem=20000-bs=512-e=1',
        'tetris-20190731-221411-nn=[32, 32]-mem=25000-bs=512-e=1',
        # 'tetris-20190731-223042-nn=[32, 32]-mem=5000-bs=1024-e=1',
        'tetris-20190731-224908-nn=[32, 32]-mem=10000-bs=1024-e=1',
        # 'tetris-20190731-230310-nn=[32, 32]-mem=15000-bs=1024-e=1',
        # 'tetris-20190731-232109-nn=[32, 32]-mem=20000-bs=1024-e=1',
        # 'tetris-20190731-234617-nn=[32, 32]-mem=25000-bs=1024-e=1',
    ]
    for d in dirs:
        print(f"Evaluating dir '{d}'")
        run_eval(d)


if __name__ == "__main__":
    enumerate_run_eval()
    cv2.destroyAllWindows()
    exit(0)

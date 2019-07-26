# Play mode, when we can play the Tetris normally, with the keyboard

import cv2

from tetris import Tetris

env = Tetris()
env.reset()
env.render()

rotation = 0

iteration = 0
while True:
    k = cv2.waitKey(100)  # 10 milliseconds
    env.render()
    if k == 27:
        print('ESC')
        break
    elif k == 81:  # left
        env.move([-1, 0], 0, True)
    elif k == 83:  # right
        env.move([+1, 0], 0, True)
    elif k == 84:  # down
        env.move([0, +1], 0, True)
    elif k == 82:  # up
        # clockwise rotation
        rotation += 1
        print('rotation=', rotation)
        env.move([0, 0], -90, True)
    elif k == 32:  # space
        _, done = env.hard_drop(env.current_pos, env.current_rotation, render=True, render_delay=0)
        if done:
            break

    if iteration >= 8:
        env.fall(render=True)
        if env.game_over:
            break
        iteration = 0
    else:
        iteration += 1

# do this after the break from the while loop
cv2.destroyAllWindows()
# to avoid python console running
exit(0)

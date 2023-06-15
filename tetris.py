import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import itertools
from PIL import Image, ImageFont, ImageDraw
import os

font = ImageFont.truetype(f"Assets/Font/Cubic_11_1.013_R.ttf", 18, encoding="utf-8")
font_title = ImageFont.truetype(f"Assets/Font/Cubic_11_1.013_R.ttf", 20, encoding="utf-8")

# Tetris game class
# noinspection PyMethodMayBeStatic
class Tetris:
    """Tetris game class"""

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: {  # I
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            90: [(1, 0), (1, 1), (1, 2), (1, 3)],
            180: [(3, 0), (2, 0), (1, 0), (0, 0)],
            270: [(1, 3), (1, 2), (1, 1), (1, 0)],
        },
        1: {  # T
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 1), (1, 2), (1, 1), (1, 0)],
            180: [(1, 2), (2, 1), (1, 1), (0, 1)],
            270: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # L
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 0)],
            180: [(1, 2), (1, 1), (1, 0), (0, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # J
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 2)],
            180: [(1, 2), (1, 1), (1, 0), (2, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # Z
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(0, 2), (0, 1), (1, 1), (1, 0)],
            180: [(2, 1), (1, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (0, 1), (0, 2)],
        },
        5: {  # S
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(0, 1), (1, 1), (1, 0), (2, 0)],
            270: [(1, 2), (1, 1), (0, 1), (0, 0)],
        },
        6: {  # O
            0: [(1, 0), (2, 0), (1, 1), (2, 1)],
            90: [(1, 0), (2, 0), (1, 1), (2, 1)],
            180: [(1, 0), (2, 0), (1, 1), (2, 1)],
            270: [(1, 0), (2, 0), (1, 1), (2, 1)],
        },
        7: None, # Empty
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99), # MAP_BLOCK
        2: (0, 167, 247), # MAP_PLAYER
    }

    ASSETS_PATH = f"Assets/"
    ASSETS = {
        "BG": os.path.join(ASSETS_PATH,"bg.PNG"),
        "board_BG": os.path.join(ASSETS_PATH, "board_bg.PNG"),
        "board_grid": os.path.join(ASSETS_PATH, "board_grid.PNG"),
        "board_border": os.path.join(ASSETS_PATH, "board_border.PNG"),
        "board_info": os.path.join(ASSETS_PATH, "board_info.PNG"),
        "decorate": os.path.join(ASSETS_PATH, "decorate.PNG")
    }

    BLOCK_ASSETS_PATH = os.path.join(ASSETS_PATH, "Blocks") 
    MAP_BLOCK_IMG = os.path.join(BLOCK_ASSETS_PATH, "blockMap_1.PNG")
    BLOCK_ASSETS = {
        0: os.path.join(BLOCK_ASSETS_PATH, "blockI_1.PNG"), # I
        1: os.path.join(BLOCK_ASSETS_PATH, "blockT_1.PNG"), # T
        2: os.path.join(BLOCK_ASSETS_PATH, "blockL_1.PNG"), # L
        3: os.path.join(BLOCK_ASSETS_PATH, "blockJ_1.PNG"), # J
        4: os.path.join(BLOCK_ASSETS_PATH, "blockZ_1.PNG"), # Z
        5: os.path.join(BLOCK_ASSETS_PATH, "blockS_1.PNG"), # S
        6: os.path.join(BLOCK_ASSETS_PATH, "blockO_1.PNG"), # O
        7: None, # Empty
    }
    COMPLETED_BLOCK_ASSETS = {
        0: os.path.join(BLOCK_ASSETS_PATH, "blockI.PNG"), # I
        1: os.path.join(BLOCK_ASSETS_PATH, "blockT.PNG"), # T
        2: os.path.join(BLOCK_ASSETS_PATH, "blockL.PNG"), # L
        3: os.path.join(BLOCK_ASSETS_PATH, "blockJ.PNG"), # J
        4: os.path.join(BLOCK_ASSETS_PATH, "blockZ.PNG"), # Z
        5: os.path.join(BLOCK_ASSETS_PATH, "blockS.PNG"), # S
        6: os.path.join(BLOCK_ASSETS_PATH, "blockO.PNG"), # O
        7: None, # Empty
    }

    def __init__(self):
        # to avoid warnings just mention the warnings
        self.game_over = False
        self.current_pos = [3, 0]
        self.current_rotation = 0
        self.board = []
        self.board_label = []
        self.bag = []
        self.next_piece = None
        self.score = 0
        self.lines = 0

        self.reset()

    def reset(self):
        """Resets the game, returning the current state"""
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.board_label = [[-1] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS) - 1))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round(piece_fall=False)
        self.score = 0
        return self._get_board_props(self.board, self.board_label)

    def _get_rotated_piece(self, rotation):
        """Returns the current piece, including rotation"""
        return Tetris.TETROMINOS[self.current_piece][rotation]

    def _get_complete_board(self):
        """Returns the complete board, including the current piece"""
        piece = self._get_rotated_piece(self.current_rotation)
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
            self.board_label[y][x] = list(Tetris.TETROMINOS.keys()).index(self.current_piece)
        return board

    def get_game_score(self):
        """Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        """
        return self.score

    def _new_round(self, piece_fall=False) -> int:
        """Starts a new round (new piece)"""
        score = 0
        if piece_fall:
            # Update board and calculate score
            piece = self._get_rotated_piece(self.current_rotation)
            self.board, self.board_label = self._add_piece_to_board(piece, self.current_pos)
            lines_cleared, self.board, self.board_label = self._clear_lines(self.board, self.board_label)
            score = 1 + ((2 ** (lines_cleared - 1)) * Tetris.BOARD_WIDTH if lines_cleared > 0 else 0)
            self.score += score
            self.lines += lines_cleared

        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        # Generate new bag with the pieces
        if len(self.bag) < 3:
            tmp = list(range(len(Tetris.TETROMINOS) - 1))
            random.shuffle(tmp)
            self.bag += tmp

        if not self.is_valid_position(self._get_rotated_piece(self.current_rotation), self.current_pos):
            self.game_over = True
            self.lines = 0
        return score

    def is_valid_position(self, piece, pos):
        """Check if there is a collision between the current piece and the board.
        :returns: True, if the piece position is _invalid_, False, otherwise
        """
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return False
        return True

    def _rotate(self, angle):
        """Change the current rotation"""
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def _add_piece_to_board(self, piece, pos):
        """Place a piece in the board, returning the resulting board"""
        board = [x[:] for x in self.board]
        board_label = [x[:] for x in self.board_label]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
            board_label[y + pos[1]][x + pos[0]] = list(Tetris.TETROMINOS.keys()).index(self.current_piece)
        return board, board_label

    def _clear_lines(self, board, board_label):
        """Clears completed lines in a board"""
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            board_label = [row for index, row in enumerate(board_label) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
                board_label.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board, board_label

    def _number_of_holes(self, board):
        """Number of holes in the board (empty square with at least one block above it)"""
        holes = 0

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            holes += len([x for x in tail if x == Tetris.MAP_EMPTY])

        return holes

    def _bumpiness(self, board):
        """Sum of the differences of heights between pair of columns"""
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            n = Tetris.BOARD_HEIGHT - len([x for x in tail])
            min_ys.append(n)

        for (y0, y1) in window(min_ys):
            bumpiness = abs(y0 - y1)
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += bumpiness

        return total_bumpiness, max_bumpiness

    def _height(self, board):
        """Sum and maximum height of the board"""
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            height = len([x for x in tail])

            sum_height += height
            max_height = max(height, max_height)
            min_height = min(height, min_height)

        return sum_height, max_height, min_height

    def _get_board_props(self, board, board_label) -> List[int]:
        """Get properties of the board"""
        lines, board, board_label = self._clear_lines(board, board_label)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def get_next_states(self) -> Dict[Tuple[int, int], List[int]]:
        """Get all possible next states"""
        states = {}
        piece_id = self.current_piece

        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while self.is_valid_position(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board, board_label = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board, board_label)

        return states

    def get_state_size(self):
        """Size of the state"""
        return 4

    def move(self, shift_m, shift_r) -> bool:
        pos = self.current_pos.copy()
        pos[0] += shift_m[0]
        pos[1] += shift_m[1]
        rotation = self.current_rotation
        rotation = (rotation + shift_r + 360) % 360
        piece = self._get_rotated_piece(rotation)
        if self.is_valid_position(piece, pos):
            self.current_pos = pos
            self.current_rotation = rotation
            return True
        return False

    def fall(self) -> bool:
        """:returns: True, if there was a fall move, False otherwise"""
        if not self.move([0, 1], 0):
            # cannot fall further
            # start new round
            self._new_round(piece_fall=True)
            if self.game_over:
                self.score -= 2
        return self.game_over

    def hard_drop(self, pos, rotation, render=False):
        """Makes a hard drop given a position and a rotation, returning the reward and if the game is over"""
        self.current_pos = pos
        self.current_rotation = rotation
        # drop piece
        piece = self._get_rotated_piece(self.current_rotation)
        while self.is_valid_position(piece, self.current_pos):
            if render:
                self.render(wait_key=True)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1
        # start new round
        score = self._new_round(piece_fall=True)
        if self.game_over:
            score -= 2
        if render:
            self.render(wait_key=True)
        return score, self.game_over

    def render(self, wait_key=False):
        """Renders the current board"""
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape((Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3)).astype(np.uint8)
        img = img[..., ::-1]  # Convert RRG to BGR (used by cv2)

        nn_v_img = Image.fromarray(img, "RGB")
        nn_v_img = nn_v_img.resize((Tetris.BOARD_WIDTH * 27, Tetris.BOARD_HEIGHT * 27))
        cv2.imshow("NN Visualization", np.array(nn_v_img))

        img = self.renderImg(img)
        cv2.imshow("Tetris AI - DQN", np.array(img))
        
        if wait_key:
            # this is needed to render during training
            cv2.waitKey(1)

    def renderImg(self, img):
        """Import some assets"""
        # Assets' size is 960x540

        # board assets
        board_BG_img = Image.open(Tetris.ASSETS["board_BG"]).convert("RGBA")
        board_grid_img = Image.open(Tetris.ASSETS["board_grid"]).convert("RGBA")
        board_border_img = Image.open(Tetris.ASSETS["board_border"]).convert("RGBA")
        board_info_img = Image.open(Tetris.ASSETS["board_info"]).convert("RGBA")

        board_BG_img.paste(board_grid_img, (0, 0), board_grid_img)
        board_BG_img.paste(board_info_img, (0, 0), board_info_img)
        board_BG_img.paste(board_border_img, (0, 0), board_border_img)

        # process the blocks in map
        '''map_block = Image.open(Tetris.MAP_BLOCK_IMG).convert("RGBA")
        current_block = list(Tetris.TETROMINOS.keys()).index(self.current_piece)
        current_block_img = Image.open(Tetris.BLOCK_ASSETS[current_block]).convert("RGBA")'''
        for y in range(Tetris.BOARD_HEIGHT):
            for x in range(Tetris.BOARD_WIDTH):
                if self.board_label[y][x] != -1 \
                    and img[y, x, 0] != Tetris.COLORS[Tetris.MAP_EMPTY][2] \
                    and img[y, x, 1] != Tetris.COLORS[Tetris.MAP_EMPTY][1] \
                    and img[y, x, 2] != Tetris.COLORS[Tetris.MAP_EMPTY][0]:
                    block_img = Image.open(Tetris.BLOCK_ASSETS[self.board_label[y][x]]).convert("RGBA")
                    board_BG_img.paste(block_img, ((480 - 125) + x * 25, 13 + y * 25), block_img)
                    # 480 is (the width of the image / 2) (unit: pixels)
                    # 127 is the center of the image to the board's left border (unit: pixels)
                    # 13 is the board's top border to the image's top (unit: pixel)
                    # 25 is the block's size (unit: pixel)

        # show next block
        next_block = list(Tetris.TETROMINOS.keys()).index(self.next_piece)
        next_block_img = Image.open(Tetris.COMPLETED_BLOCK_ASSETS[next_block]).convert("RGBA")
        next_block_img = next_block_img.resize((int(next_block_img.size[0] * 0.6)
                                                , int(next_block_img.size[1] * 0.6))) # re-scale 0.6 times
        board_BG_img.paste(next_block_img, 
                           (480 + 127 + 6 + (61 // 2) - next_block_img.size[0] // 2, 13 + 50), 
                           next_block_img)
        # 6 is the board's right board to the right information frame (unit: pixels)
        # 20 is the "next block frame" to the image's top (unit: pixels)
        # 61 is the width of "next block frame" (unit: pixels)

        for i in range (1, 3):
            next_block = list(Tetris.TETROMINOS.keys()).index(self.bag[-i])
            next_block_img = Image.open(Tetris.COMPLETED_BLOCK_ASSETS[next_block]).convert("RGBA")
            next_block_img = next_block_img.resize((int(next_block_img.size[0] * 0.6)
                                                    , int(next_block_img.size[1] * 0.6))) # re-scale 0.6 times
            board_BG_img.paste(next_block_img, 
                            (480 + 127 + 6 + (61 // 2) - next_block_img.size[0] // 2, 13 + 50 + 80 * i), 
                            next_block_img)
        # 80 is the each blocks interval (unit: pixels)

        # other assets (bg, decoration...)
        bg_img = Image.open(Tetris.ASSETS["BG"]).convert("RGB")
        decorate_img = Image.open(Tetris.ASSETS["decorate"]).convert("RGBA")
        bg_img.paste(decorate_img, (0, 0), decorate_img)
        bg_img.paste(board_BG_img, (0, 0), board_BG_img)
        
        # information frame (text...)
        draw = ImageDraw.Draw(bg_img)
        draw.text(tuple((480 + 127 + 61 // 4, 355)), "Lines: \n" + str(self.lines).rjust(6, " "),
                  tuple((99, 99, 49)), font=font, stroke_width=2, stroke_fill="white")
        draw.text(tuple((480 + 127 + 61 // 4, 410)), "Score: \n" + str(self.score).rjust(6, " "),
                  tuple((99, 99, 49)), font=font, stroke_width=2, stroke_fill="white")
        # 355 and 410 is the "information block frame" to the image's top (unit: pixels)

        draw.text((480 + 127 + 6 + 61 // 8, 13 + 10), "NEXT", 
                  tuple((99, 99, 49)), font=font_title, stroke_width=2, stroke_fill="white") # the text of next
        draw.text((480 - 127 - 6 - 76 + 76 // 5, 13 + 10), "HOLD", 
                  tuple((99, 99, 49)), font=font_title, stroke_width=2, stroke_fill="white") # the text of hold
        # 76 is the width of "hold block frame" (unit: pixels)

        bg_img = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2BGRA)
        return bg_img

def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
       NB. taken from https://docs.python.org/release/2.3.5/lib/itertools-example.html
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

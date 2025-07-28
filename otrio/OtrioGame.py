from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np


def reserves_left(board: np.ndarray) -> int:
    """総残数を取得するヘルパー"""
    offset = OtrioGame.RESERVE_OFFSET
    return int(board[:, offset:, 0, 0].sum())

class OtrioGame(Game):
    """
    Alpha‑Zero ‘Game’ adapter for **Otrio**.

    ----------
    Board encoding
    ----------
    * Shape: ``(3, 3, 3)`` → ``(size, row, col)``
      * ``size`` 0 = small, 1 = medium, 2 = large
      * Value  0 = empty, 1 = current player, -1 = opponent (canonical form multiplies by player)
    * Action space: 27 moves (size × row × col) mapped via
      ``idx = size*9 + row*3 + col``

    TODO:
        – Extend `n_players` > 2 (update `getNextState` / win check)
    """

    SIZES = 3
    N = 3  # board is 3×3 pegs
    PIECES_PER_SIZE = 3
    RESERVE_OFFSET = SIZES

    def __init__(self, n_players: int = 2):
        self.n_players = n_players
        self.COLORS = 4 if n_players == 2 else n_players
        self.player_colors = (
            [[0, 1], [2, 3]]
            if n_players == 2
            else [[i] for i in range(n_players)]
        )
        self.next_color = np.zeros(n_players, np.int8)
        self.action_size = self.SIZES * self.N * self.N  # 27

    # ══════════════ Alpha‑Zero required API ══════════════
    def getInitBoard(self):
        """空の盤面を生成し、色とリザーブを初期化する。"""
        board = np.zeros((self.COLORS, self.SIZES * 2, self.N, self.N), dtype=np.int8)
        self.next_color[:] = 0
        for c in range(self.COLORS):
            for s in range(self.SIZES):
                board[c, self.RESERVE_OFFSET + s, 0, 0] = self.PIECES_PER_SIZE
        return board

    def getBoardSize(self):
        return (self.COLORS, self.SIZES * 2, self.N, self.N)

    def getActionSize(self):
        return self.action_size

    def getNextState(self, board: np.ndarray, player: int, action: int):
        """Execute `action` and switch to next player.

        Args:
            board: current board (*not* canonicalised)
            player: current player id (1 or -1 for 2‑player)
            action: 0‑26 encoded move
        Returns:
            (next_board, next_player)
        """
        b = board.copy()
        size, rem = divmod(action, 9)
        row, col = divmod(rem, 3)

        idx = player - 1
        colors = self.player_colors[idx]
        c_idx = self.next_color[idx] if self.n_players == 2 else 0
        color = colors[c_idx]

        assert b[color, size, row, col] == 0, "Illegal move!"
        assert b[color, self.RESERVE_OFFSET + size, 0, 0] > 0, "No pieces left!"

        b[color, size, row, col] = player
        b[color, self.RESERVE_OFFSET + size, 0, 0] -= 1
        if self.n_players == 2:
            self.next_color[idx] ^= 1

        if self.n_players == 2:
            next_player = -player
        else:
            # 1 → 2 → 3 → 4 → 1 …
            next_player = (player % self.n_players) + 1
        return b, next_player

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Returns:
            mask (np.ndarray[int8]): shape (27,), 1 = legal
        """
        idx = player - 1
        colors = self.player_colors[idx]
        c_idx = self.next_color[idx] if self.n_players == 2 else 0
        color = colors[c_idx]

        mask = np.zeros(self.action_size, np.int8)
        for size in range(self.SIZES):
            if board[color, self.RESERVE_OFFSET + size, 0, 0] == 0:
                continue
            plane = board[color, size]
            mask[size*9:(size+1)*9] = (plane.reshape(-1) == 0)

        return mask

    def getGameEnded(self, board: np.ndarray, player: int):
        """0 = ongoing, 1 = win for *player*, -1 = loss, 1e‑4 = draw."""
        if reserves_left(board) == 0:
            return 1e-4

        def _has_line(p):
            for c in range(self.COLORS):
                for size in range(self.SIZES):
                    plane = board[c, size] == p
                    if np.any(np.all(plane, axis=0)):
                        return True  # column
                    if np.any(np.all(plane, axis=1)):
                        return True  # row
                    if np.all(np.diag(plane)) or np.all(np.diag(np.fliplr(plane))):
                        return True  # diagonal
                if np.any(np.all(board[c] == p, axis=0)):
                    return True  # tower
            return False

        if _has_line(player):
            return 1
        if _has_line(-player):
            return -1

        if not (board == 0).any():
            return 1e-4  # draw
        return 0

    def getCanonicalForm(self, board: np.ndarray, player: int):
        b = board.copy()
        b *= player
        return b

    def getSymmetries(self, board: np.ndarray, pi: np.ndarray):
        """8 rotational / mirror symmetries.

        Args:
            board: np.ndarray (3×3×3)
            pi   : flat 27‑dim policy
        Returns:
            list[(board_sym, pi_sym)]
        """
        sym = []
        pi = np.asarray(pi, dtype=np.float32)
        pi_board = pi.reshape(self.SIZES, self.N, self.N)
        for k in range(4):
            rb = np.rot90(board, k, axes=(2, 3))
            rpi = np.rot90(pi_board, k, axes=(1, 2))
            sym.append((rb, rpi.flatten()))
            # mirror horizontally
            mb = np.flip(rb, axis=3)
            mpi = np.flip(rpi, axis=2)
            sym.append((mb, mpi.flatten()))
        return sym

    def stringRepresentation(self, board: np.ndarray):
        return board.tobytes()

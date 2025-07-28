from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np

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

    def __init__(self, n_players: int = 2):
        self.n_players = n_players
        self.action_size = self.SIZES * self.N * self.N  # 27

    # ══════════════ Alpha‑Zero required API ══════════════
    def getInitBoard(self):
        """Zero‑filled board with piece reserves."""
        channels = self.SIZES + self.n_players * self.SIZES
        board = np.zeros((channels, self.N, self.N), dtype=np.int8)
        for p in range(self.n_players):
            offset = self.SIZES + p * self.SIZES
            for s in range(self.SIZES):
                board[offset + s, :, :] = self.PIECES_PER_SIZE
        return board

    def getBoardSize(self):
        channels = self.SIZES + self.n_players * self.SIZES
        return (channels, self.N, self.N)

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
        assert b[size, row, col] == 0, "Illegal move!"

        if player == 1:
            offset = self.SIZES
        else:
            offset = self.SIZES * 2
        reserve_layer = offset + size
        assert b[reserve_layer, 0, 0] > 0, "No pieces left!"
        b[reserve_layer, :, :] -= 1

        b[size, row, col] = player

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
        mask = (board[:self.SIZES].reshape(-1) == 0).astype(np.int8)

        # 各サイズの残り駒が０ならそのレイヤすべて無効
        if player == 1:
            offset = self.SIZES
        else:
            offset = self.SIZES * 2
        for size in range(self.SIZES):
            remaining = board[offset + size, 0, 0]
            if remaining == 0:
                mask[size*9:(size+1)*9] = 0

        return mask

    def getGameEnded(self, board: np.ndarray, player: int):
        """0 = ongoing, 1 = win for *player*, -1 = loss, 1e‑4 = draw."""
        def _has_line(p):
            for size in range(self.SIZES):
                plane = board[size] == p
                if np.any(np.all(plane, axis=0)):
                    return True  # column
                if np.any(np.all(plane, axis=1)):
                    return True  # row
                if np.all(np.diag(plane)) or np.all(np.diag(np.fliplr(plane))):
                    return True  # diagonal
            # tower win
            if np.any(np.all(board[:self.SIZES] == p, axis=0)):
                return True
            return False

        if _has_line(player):
            return 1
        if _has_line(-player):
            return -1
        if not (board[:self.SIZES] == 0).any():
            return 1e-4  # draw
        return 0

    def getCanonicalForm(self, board: np.ndarray, player: int):
        b = board.copy()
        b[:self.SIZES] *= player
        if player == -1:
            start = self.SIZES
            mid = self.SIZES * 2
            b[start:mid], b[mid:] = b[mid:].copy(), b[start:mid].copy()
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
            rb = np.rot90(board, k, axes=(1, 2))
            rpi = np.rot90(pi_board, k, axes=(1, 2))
            sym.append((rb, rpi.flatten()))
            # mirror horizontally
            mb = np.flip(rb, axis=2)
            mpi = np.flip(rpi, axis=2)
            sym.append((mb, mpi.flatten()))
        return sym

    def stringRepresentation(self, board: np.ndarray):
        return board.tobytes()

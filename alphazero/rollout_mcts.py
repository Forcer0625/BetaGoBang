# coding: utf-8
import random
import numpy as np

from .chess_board import ChessBoard
from .node import Node


class RolloutMCTS:
    """ 基於隨機走棋策略的蒙特卡洛樹搜索 """

    def __init__(self, c_puct: float = 5, n_iters=1000):
        """
        Parameters
        ----------
        c_puct: float
            探索常數

        n_iters: int
            迭代搜索次數
        """
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.root = Node(1, c_puct, parent=None)

    def get_action(self, chess_board: ChessBoard) -> int:
        """ 根據當前局面返回下一步動作

        Parameters
        ----------
        chess_board: ChessBoard
            棋盤
        """
        for i in range(self.n_iters):
            # 拷貝一個棋盤用來模擬
            board = chess_board.copy()

            # 如果沒有遇到葉節點，就一直向下搜索並更新棋盤
            node = self.root
            while not node.is_leaf_node():
                action, node = node.select()
                board.do_action(action)

            # 判斷遊戲是否結束，如果沒結束就拓展葉節點
            is_over, winner = board.is_game_over()
            if not is_over:
                node.expand(self.__default_policy(board))

            # 模擬
            value = self.__rollout(board)
            # 反向傳播
            node.backup(-1*value)

        # 根據子節點的訪問次數來選擇動作
        action = max(self.root.children.items(), key=lambda i: i[1].N)[0]
        # 更新根節點
        self.root = Node(prior_prob=1)
        return action

    def __default_policy(self, chess_board: ChessBoard):
        """ 根據當前局面返回可進行的動作及其概率

        Returns
        -------
        action_probs: List[Tuple[int, float]]
            每個元素都為 `(action, prior_prob)` 元組，根據這個元組創建子節點，
            `action_probs` 的長度為當前棋盤的可用落點的總數
        """
        n = len(chess_board.available_actions)
        probs = np.ones(n) / n
        return zip(chess_board.available_actions, probs)

    def __rollout(self, board: ChessBoard):
        """ 快速走棋，模擬一局 """
        current_player = board.current_player

        while True:
            is_over, winner = board.is_game_over()
            if is_over:
                break
            action = random.choice(board.available_actions)
            board.do_action(action)

        # 計算 Value，平局為 0，當前玩家勝利則為 1, 輸為 -1
        if winner is not None:
            return 1 if winner == current_player else -1
        return 0
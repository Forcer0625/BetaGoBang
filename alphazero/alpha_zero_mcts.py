# coding: utf-8
from typing import Tuple, Union

import numpy as np

from .chess_board import ChessBoard
from .node import Node
from .policy_value_net import PolicyValueNet


class AlphaZeroMCTS:
    """ 基於策略-價值網絡的蒙特卡洛搜索樹 """

    def __init__(self, policy_value_net: PolicyValueNet, c_puct: float = 4, n_iters=1200, is_self_play=False) -> None:
        """
        Parameters
        ----------
        policy_value_net: PolicyValueNet
            策略價值網絡

        c_puct: float
            探索常數

        n_iters: int
            迭代次數

        is_self_play: bool
            是否處於自我博弈狀態
        """
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.is_self_play = is_self_play
        self.policy_value_net = policy_value_net
        self.root = Node(prior_prob=1, parent=None)

    def get_action(self, chess_board: ChessBoard) -> Union[Tuple[int, np.ndarray], int]:
        """ 根據當前局面返回下一步動作

        Parameters
        ----------
        chess_board: ChessBoard
            棋盤

        Returns
        -------
        action: int
            當前局面下的最佳動作

        pi: `np.ndarray` of shape `(board_len^2, )`
            執行動作空間中每個動作的概率，只在 `is_self_play=True` 模式下返回
        """
        for i in range(self.n_iters):
            # 拷貝棋盤
            board = chess_board.copy()

            # 如果沒有遇到葉節點，就一直向下搜索並更新棋盤
            node = self.root
            while not node.is_leaf_node():
                action, node = node.select()
                board.do_action(action)

            # 判斷遊戲是否結束，如果沒結束就拓展葉節點
            is_over, winner = board.is_game_over()
            p, value = self.policy_value_net.predict(board)
            if not is_over:
                # 添加狄利克雷噪聲
                if self.is_self_play:
                    p = 0.75*p + 0.25 * \
                        np.random.dirichlet(0.03*np.ones(len(p)))
                node.expand(zip(board.available_actions, p))
            elif winner is not None:
                value = 1 if winner == board.current_player else -1
            else:
                value = 0

            # 反向傳播
            node.backup(-value)

        # 計算 π，在自我博弈狀態下：遊戲的前三十步，溫度係數為 1，後面的溫度係數趨於無窮小
        T = 1 if self.is_self_play and len(chess_board.state) <= 30 else 1e-3
        visits = np.array([i.N for i in self.root.children.values()])
        pi_ = self.__getPi(visits, T)
        # if visits.size == 0:
        #     print("size 0!")
        # else:
        #     pi_ = self.__getPi(visits, T)

        # 根據 π 選出動作及其對應節點
        actions = list(self.root.children.keys())
        action = int(np.random.choice(actions, p=pi_))

        if self.is_self_play:
            # 創建維度為 board_len^2 的 π
            pi = np.zeros(chess_board.board_len**2)
            pi[actions] = pi_
            # 更新根節點
            self.root = self.root.children[action]
            self.root.parent = None
            return action, pi
        else:
            self.reset_root()
            return action
    def get_human_action(self, chess_board: ChessBoard) -> Union[Tuple[int, np.ndarray], int]:
        """ 根據當前局面返回下一步動作

        Parameters
        ----------
        chess_board: ChessBoard
            棋盤

        Returns
        -------
        action: int
            當前局面下的最佳動作

        pi: `np.ndarray` of shape `(board_len^2, )`
            執行動作空間中每個動作的概率，只在 `is_self_play=True` 模式下返回
        """
        action = input("輸入放置位置:")
        action = int(action)
        for i in range(self.n_iters):
            # 拷貝棋盤
            board = chess_board.copy()

            # 如果沒有遇到葉節點，就一直向下搜索並更新棋盤
            node = self.root
            while not node.is_leaf_node():
                action, node = node.select()
                board.do_action(action)

            # 判斷遊戲是否結束，如果沒結束就拓展葉節點
            is_over, winner = board.is_game_over()
            p, value = self.policy_value_net.predict(board)
            if not is_over:
                # 添加狄利克雷噪聲
                if self.is_self_play:
                    p = 0.75*p + 0.25 * \
                        np.random.dirichlet(0.03*np.ones(len(p)))
                node.expand(zip(board.available_actions, p))
            elif winner is not None:
                value = 1 if winner == board.current_player else -1
            else:
                value = 0

            # 反向傳播
            node.backup(-value)

        # 計算 π，在自我博弈狀態下：遊戲的前三十步，溫度係數為 1，後面的溫度係數趨於無窮小
        T = 1 if self.is_self_play and len(chess_board.state) <= 30 else 1e-3
        visits = np.array([i.N for i in self.root.children.values()])
        pi_ = self.__getPi(visits, T)
        # if visits.size == 0:
        #     print("size 0!")
        # else:
        #     pi_ = self.__getPi(visits, T)

        # 根據 π 選出動作及其對應節點
        actions = list(self.root.children.keys())
        action = int(np.random.choice(actions, p=pi_))

        if self.is_self_play:
            # 創建維度為 board_len^2 的 π
            pi = np.zeros(chess_board.board_len**2)
            pi[actions] = pi_
            # 更新根節點
            self.root = self.root.children[action]
            self.root.parent = None
            return action, pi
        else:
            self.reset_root()
            return action

    def __getPi(self, visits, T) -> np.ndarray:
        """ 根據節點的訪問次數計算 π """
        # pi = visits**(1/T) / np.sum(visits**(1/T)) 會出現標量溢出問題，所以使用對數壓縮
        x = 1/T * np.log(visits + 1e-11)
        x = np.exp(x - x.max())
        pi = x/x.sum()
        return pi

    def reset_root(self):
        """ 重置根節點 """
        self.root = Node(prior_prob=1, c_puct=self.c_puct, parent=None)

    def set_self_play(self, is_self_play: bool):
        """ 設置蒙特卡洛樹的自我博弈狀態 """
        self.is_self_play = is_self_play

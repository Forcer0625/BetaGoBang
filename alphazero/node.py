# coding: utf-8
from math import sqrt
from typing import Tuple, Iterable, Dict


class Node:
    """ 蒙特卡洛樹節點 """

    def __init__(self, prior_prob: float, c_puct: float = 5, parent=None):
        """
        Parameters
        ----------
        prior_prob: float
            節點的先驗概率 `P(s, a)`

        c_puct: float
            探索常數

        parent: Node
            父級節點
        """
        self.Q = 0
        self.U = 0
        self.N = 0
        self.score = 0
        self.P = prior_prob
        self.c_puct = c_puct
        self.parent = parent
        self.children = {}  # type:Dict[int, Node]

    def select(self) -> tuple:
        """ 返回 `score` 最大的子節點和該節點對應的 action

        Returns
        -------
        action: int
            動作

        child: Node
            子節點
        """
        return max(self.children.items(), key=lambda item: item[1].get_score())

    def expand(self, action_probs: Iterable[Tuple[int, float]]):
        """ 拓展節點

        Parameters
        ----------
        action_probs: Iterable
            每個元素都為 `(action, prior_prob)` 元組，根據這個元組創建子節點，
            `action_probs` 的長度為當前棋盤的可用落點的總數
        """
        for action, prior_prob in action_probs:
            self.children[action] = Node(prior_prob, self.c_puct, self)

    def __update(self, value: float):
        """ 更新節點的訪問次數 `N(s, a)`、節點的累計平均獎賞 `Q(s, a)`

        Parameters
        ----------
        value: float
            用來更新節點內部數據
        """
        self.Q = (self.N * self.Q + value)/(self.N + 1)
        self.N += 1

    def backup(self, value: float):
        """ 反向傳播 """
        if self.parent:
            self.parent.backup(-value)

        self.__update(value)

    def get_score(self):
        """ 計算節點得分 """
        self.U = self.c_puct * self.P * sqrt(self.parent.N)/(1 + self.N)
        self.score = self.U + self.Q
        return self.score

    def is_leaf_node(self):
        """ 是否為葉節點 """
        return len(self.children) == 0
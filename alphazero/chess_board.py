# coding: utf-8
from typing import Tuple
from copy import deepcopy
from collections import OrderedDict

import torch
import numpy as np


class ChessBoard:
    """ 棋盤類 """

    EMPTY = -1
    WHITE = 0
    BLACK = 1

    def __init__(self, board_len=9, n_feature_planes=7):
        """
        Parameters
        ----------
        board_len: int
            棋盤邊長

        n_feature_planes: int
            特徵平面的個數，必須為偶數
        """
        self.board_len = board_len
        self.current_player = self.BLACK
        self.n_feature_planes = n_feature_planes
        self.available_actions = list(range(self.board_len**2))
        # 棋盤狀態字典，key 為 action，value 為 current_player
        self.state = OrderedDict()
        # 上一個落點
        self.previous_action = None

    def copy(self):
        """ 複製棋盤 """
        return deepcopy(self)

    def clear_board(self):
        """ 清空棋盤 """
        self.state.clear()
        self.previous_action = None
        self.current_player = self.BLACK
        self.available_actions = list(range(self.board_len**2))

    def do_action(self, action: int):
        """ 落子並更新棋盤

        Parameters
        ----------
        action: int
            落子位置，範圍為 `[0, board_len^2 -1]`
        """
        self.previous_action = action
        self.available_actions.remove(action)
        self.state[action] = self.current_player
        self.current_player = self.WHITE + self.BLACK - self.current_player

    def do_action_(self, pos: tuple) -> bool:
        """ 落子並更新棋盤，只提供給 app 使用

        Parameters
        ----------
        pos: Tuple[int, int]
            落子在棋盤上的位置，範圍為 `(0, 0) ~ (board_len-1, board_len-1)`

        Returns
        -------
        update_ok: bool
            是否成功落子
        """
        action = pos[0]*self.board_len + pos[1]
        if action in self.available_actions:
            self.do_action(action)
            return True
        return False

    def is_game_over(self) -> Tuple[bool, int]:
        """ 判斷遊戲是否結束

        Returns
        -------
        is_over: bool
            遊戲是否結束，分出勝負或者平局則為 `True`, 否則為 `False`

        winner: int
            遊戲贏家，有以下幾種:
            * 如果遊戲分出勝負，則為 `ChessBoard.BLACK` 或 `ChessBoard.WHITE`
            * 如果還有分出勝負或者平局，則為 `None`
        """
        # 如果下的棋子不到 9 個，就直接判斷遊戲還沒結束
        if len(self.state) < 9:
            return False, None

        n = self.board_len
        act = self.previous_action
        player = self.state[act]
        row, col = act//n, act % n

        # 搜索方向
        directions = [[(0, -1),  (0, 1)],   # 水平搜索
                      [(-1, 0),  (1, 0)],   # 豎直搜索
                      [(-1, -1), (1, 1)],   # 主對角線搜索
                      [(1, -1),  (-1, 1)]]  # 副對角線搜索

        for i in range(4):
            count = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if 0 <= row_t < n and 0 <= col_t < n and self.state.get(row_t*n+col_t, self.EMPTY) == player:
                        # 遇到相同顏色時 count+1
                        count += 1
                    else:
                        flag = False
            # 分出勝負
            if count >= 5:
                return True, player

        # 平局
        if not self.available_actions:
            return True, None

        return False, None

    def get_feature_planes(self) -> torch.Tensor:
        """ 棋盤狀態特徵張量，維度為 `(n_feature_planes, board_len, board_len)`

        Returns
        -------
        feature_planes: Tensor of shape `(n_feature_planes, board_len, board_len)`
            特徵平面圖像
        """
        n = self.board_len
        feature_planes = torch.zeros((self.n_feature_planes, n**2))
        # 最後一張圖像代表當前玩家顏色
        # feature_planes[-1] = self.current_player
        # 添加歷史信息
        if self.state:
            actions = np.array(list(self.state.keys()))[::-1]
            players = np.array(list(self.state.values()))[::-1]
            Xt = actions[players == self.current_player]
            Yt = actions[players != self.current_player]
            for i in range((self.n_feature_planes-1)//2):
                if i < len(Xt):
                    feature_planes[2*i, Xt[i:]] = 1
                if i < len(Yt):
                    feature_planes[2*i+1, Yt[i:]] = 1

        return feature_planes.view(self.n_feature_planes, n, n)


class ColorError(ValueError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
# !/usr/bin/env python
# coding=utf-8

from sys import exit
import pygame
from pygame.locals import *
from game import Game

import torch
import torch.nn.functional as F
from torch import  cuda

from alphazero.alpha_zero_mcts import AlphaZeroMCTS
from alphazero.chess_board import ChessBoard

# 全域變數
black = 1
white = -1
blank = 0
tie = 2

class BetaGobang:
    def __init__(self, board_len=9, n_self_plays=1500, n_mcts_iters=500,
                 n_feature_planes=6, c_puct=4, is_use_gpu=True, **kwargs):
        self.c_puct = c_puct
        self.is_use_gpu = is_use_gpu
        self.n_self_plays = n_self_plays
        self.n_mcts_iters = n_mcts_iters
        self.device = torch.device(
            'cuda:0' if is_use_gpu and cuda.is_available() else 'cpu')
        self.chess_board = ChessBoard(board_len, n_feature_planes)
    
    def __do_mcts_action(self,mcts):
        """ get AI action """
        print("AI is computing...")
        action = mcts.get_action(self.chess_board)
        self.chess_board.do_action(action)
        print("AI's output:"+str(action))
        is_over, winner = self.chess_board.is_game_over()
        return is_over, winner, action
    
    def __do_human_action(self, action):
        """ get human action """
        # action = input("輸入放置位置:")
        # action = int(action)
        # action = mcts_human.get_human_action(chess_board)
        self.chess_board.do_action(action)
        is_over, winner = self.chess_board.is_game_over()
        return is_over, winner

    def start_game(self):
        easy_mode = 'model/history/best_policy_value_net_800.pth'
        normal_mode = 'model/history/best_policy_value_net_2000.pth'
        hard_mode = 'model/best_policy_value_net_6400.pth'
        model_path = hard_mode
        best_model = torch.load(model_path)
        best_model.eval()
        best_model.set_device(self.is_use_gpu)
        self.mcts = AlphaZeroMCTS(best_model, self.c_puct, self.n_mcts_iters)
        # start
        print('...正在開啟模型...')
        self.chess_board.clear_board()
        self.mcts.reset_root()
        print('...開啟模型成功...')
        
    def AI_action(self):
        is_over, winner, AIaction = self.__do_mcts_action(self.mcts)
        action = (AIaction // 9, AIaction % 9)
        return action
    
    def Human_action(self, action):
        self.__do_human_action(action[0]*9+action[1])

def human_action(env):
    """
    玩家的回合
    """
    while True:
        _action = None
        for event in pygame.event.get():  # 這是 pygame 接受輸入的迴圈，直到退出或者落子為止
            # 按下叉時退出遊戲
            if event.type == QUIT:
                exit()
            # 按下 Esc 時退出遊戲
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    exit()
            # 如果按下滑鼠，检查點擊位置，判断遊戲是否结束
            if event.type == MOUSEBUTTONDOWN:
                x, y = event.pos
                # 將點擊位置轉換為棋盤上的座標
                row, col = env.point_convert(x, y)
                _action = (row, col)
                if _action == (-1, -1) or env.has_piece(_action):   # 無效位置或已經有落子了
                    _action = None
                break
        if _action:
            return _action


def play_game():
    """
    遊戲主程式
    """
    env = Game(board_count=9, line_margin=40)
    play_config = {
    'c_puct': 3,
    'board_len': 9,
    'batch_size': 500,
    'is_use_gpu': True,
    'n_feature_planes': 6,
    }
    AI_model = BetaGobang(**play_config)
    while True:  # 每場遊戲的迴圈
        AI_model.start_game()
        print('New game', flush=True)
        state = env.reset()  # 每局遊戲開始之前重置環境
        step = 0
        pre_action = None
        while True:  # 每個回合的循環
            if env.get_color() == 1:  # 黑棋，玩家的回合
                action = human_action(env)
                AI_model.Human_action(action)
            else:  # 白棋，AI 的回合
                action = AI_model.AI_action()

            if not action:  # 若是無效 action，跳過
                raise TypeError("無效 action!")
            
            # 更新棋盤狀態
            next_state, result = env.step(action, pre_action)
            if result != 0:
                if result == black:
                    print("You Win!")
                elif result == white:
                    print("AI Win!")
                elif result == tie:
                    print("Tie!")
                break
            state = next_state
            pre_action = action
            step += 1


if __name__ == '__main__':
    print("Game start!")
    play_game()

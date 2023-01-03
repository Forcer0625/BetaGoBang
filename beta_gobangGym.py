# !/usr/bin/env python
# coding=utf-8

from sys import exit
import pygame
from pygame.locals import *
from game import Game
from stable_baselines3.a2c import A2C
import gobang_game
import gym

# 全域變數
black = 1
white = -1
blank = 0
tie = 2
board_size = 9

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
    front = Game(board_count=board_size, line_margin=40)
    # 載入遊戲環境及AI model
    env = gym.make("Gobang9x9-v1")
    model = A2C.load('./selfplay/model')
    while True:  # 每場遊戲的迴圈
        print('New game', flush=True)
        state = front.reset()  # 每局遊戲開始之前重置前端
        obs = env.reset() # 每局遊戲開始之前重置環境
        step = 0
        pre_action = None
        done = False
        while True:  # 每個回合的循環
            if front.get_color() == 1:  # 黑棋，玩家的回合
                action = human_action(front)
                next_state, result = front.step(action, pre_action)
                trans_action = action[0]*board_size+action[1]+1
                obs, reward, done, info = env.step(trans_action)
            else:  # 白棋，AI 的回合
                # AI API，輸入 state，輸出 action (棋盤上的座標)
                model_action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(model_action)
                model_action -= 1
                row = model_action//board_size
                col = model_action%board_size
                print("model action:", (row, col))
                next_state, result = front.step((row, col), pre_action)
                

            if done:
                print(info['winner'])
                if info['winner'] == black:
                    print("You Win!")
                elif info['winner'] == None:
                    print("Tie!")
                else:
                    print("AI Win!")
                break

            state = next_state
            pre_action = action
            step += 1
        model = A2C.load('./selfplay/model')


if __name__ == '__main__':
    print("Game start!")
    play_game()

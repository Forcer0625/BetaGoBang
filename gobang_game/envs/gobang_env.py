import sys
import gym
import torch
import numpy as np
from six import StringIO
from stable_baselines3 import PPO
from gym import spaces


class GobangEnv(gym.Env):

    EMPTY = -1
    WHITE = 0
    BLACK = 1

    def __init__(self, board_size=9, n_feature_planes=6):
        """
        Parameters
        ----------
        board_size: int
            棋盤長
        n_feature_planes: int
            特征平面的個數，必須為偶數
        """
        self.board_size = board_size
        self.current_player = self.BLACK
        self.n_feature_planes = n_feature_planes
        self.action_space = spaces.Discrete(self.board_size**2+1)
        observation = self.reset()
        self.observation_space = spaces.Box(
            low=np.zeros(observation.shape), high=np.ones(observation.shape))

    def reset(self):
        """ 清空棋盤 """
        self.prev_move = 0
        self.state = np.full(shape=(self.board_size**2), fill_value=-1, dtype=int)
        self.previous_actions = np.zeros(shape=(self.n_feature_planes*2, 2), dtype=int)# {step_count:action}
        self.done = False
        self.step_count = 0
        self.current_player = self.BLACK
        return self.get_observation()

    def step(self, action):
        """ 
        Parameters
        ----------
        action
            落子位置，範圍為 `[0, board_size**2]`
        """
        # 1.偵測錯誤及處理(reward=-1)
        winner = None
        if action not in list(range(self.board_size**2+1)) or action == 0:
            self.done = True

        if not self.done and self.state[action-1]!=self.EMPTY:
            self.done = True
            
        if self.done:
            reward = -1
            winner = self.WHITE + self.BLACK - self.current_player
            return self.get_observation(), reward, self.done, {'winner': winner, 'step_count': self.step_count}
        # 2.step_count+1及action儲存、sorting
        self.step_count += 1
        self.prev_move = action
        index_min = np.argmin(self.previous_actions, axis=0)
        self.previous_actions[index_min[0]] = np.array([self.step_count, action])
        observation = self.get_observation()
        # 3.落子並更新棋盤
        self.state[action-1] = self.current_player
        # 4.檢查是否結束(勝利、平手)
        if self.is_tie():
            winner = None
            reward = 0
            self.done = True
        elif self.is_done():
            winner = self.current_player
            reward = 1
            self.done = True
        else:
            self.current_player = self.WHITE + self.BLACK - self.current_player
            reward = 0
            self.done = False

        # 5.回傳狀態、獎勵等資訊
        return observation, reward, self.done, {'winner': winner, 'step_count': self.step_count}

    def render(self, mode='human', close=False):
        board = self.state
        outfile =  StringIO() if mode == 'ansi' else sys.stdout

        outfile.write('To play: ')
        outfile.write('black' if self.current_player == self.BLACK else 'white')
        outfile.write('\n')
        d = self.board_size#board.shape[1]
        if d > 9:
            outfile.write(' ' * 24)
            for j in range(10, d + 1):
                outfile.write(' ' + str(int(j/10)))
            outfile.write('\n')
        outfile.write(' ' * 6)
        for j in range(d):
            outfile.write(' ' + str((j + 1) % 10))
        outfile.write('\n')
        outfile.write(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+\n')
        for i in range(d):
            outfile.write(' ' * (3 if i < 9 else 2) + str(i + 1) + ' | ')
            for j in range(d):
                #step = self.previous_actions[self.previous_actions.argmax(axis=0)[0]][0]
                if board[i*self.board_size+j] == self.EMPTY:
                    outfile.write('. ')
                elif board[i*self.board_size+j] == self.BLACK:
                    if self.prev_move == i*self.board_size+j+1:
                        outfile.write('X)')
                    else:
                        outfile.write('X ')
                else:
                    if self.prev_move == i*self.board_size+j+1:
                        outfile.write('O)')
                    else:
                        outfile.write('O ')
            outfile.write('|\n')
        outfile.write(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+\n')
    
    def is_done(self):
        # 如果下子數不足9子，則遊戲一定還沒結束
        if self.step_count<9:
            return False

        n = self.board_size
        act = self.previous_actions[self.previous_actions.argmax(axis=0)[0]][1] - 1
        row, col = act//n, act % n

        # 搜尋方向
        directions = [[(0, -1),  (0, 1)],   # 水平搜尋
                      [(-1, 0),  (1, 0)],   # 垂直搜尋
                      [(-1, -1), (1, 1)],   # 主對角搜尋
                      [(1, -1),  (-1, 1)]]  # 副對角搜尋

        for i in range(4):
            count = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if 0 <= row_t < n and 0 <= col_t < n and self.state[row_t*n+col_t] == self.current_player:
                        # 遇到相同顏色時 count+1
                        count += 1
                    else:
                        flag = False
            # 分出勝負
            if count >= 5:
                return True

        return False

    def is_tie(self):
        return self.state[self.state.argmin()] > self.EMPTY
        
    def get_observation(self):
        # 開始疊棋盤落子狀態:自己n/2個，對手n/2個
        feature_planes = torch.zeros(self.n_feature_planes+1, self.board_size**2, dtype=torch.int8)
        n = self.n_feature_planes//2
        index_sorted = np.argsort(self.previous_actions, axis=0)[::-1]
        for i in range(n):
            # 自己
            feature_planes[i] = self.make_board(index=index_sorted, start=i*2)
            # 對手
            feature_planes[i+n] = self.make_board(index=index_sorted, start=i*2+1)

        # 加上己方的特徵平面(BLACK為1, WHITE為0)
        feature_planes[self.n_feature_planes] = torch.full(size=(self.board_size**2,), fill_value=self.current_player, dtype=torch.int8)

        return feature_planes.view(self.n_feature_planes+1, self.board_size, self.board_size)
    
    def make_board(self, index, start=0):
        # 取得過去n_feature_planes/2步的棋盤狀態
        feature = torch.zeros(self.board_size**2, dtype=torch.int8)
        for i in range(self.n_feature_planes//2):
            act = self.previous_actions[index[start+i][0]][1]
            if act != 0:
                feature[act-1] = 1
            start+=1
        return feature

class SelfPlayGobangEnv(gym.Env):
    
    EMPTY = -1
    WHITE = 0
    BLACK = 1

    def __init__(self, board_size=9, n_feature_planes=6, is_self_play=False, render_mode=False):
        """
        Parameters
        ----------
        board_size: int
            棋盤長
        n_feature_planes: int
            特征平面的個數，必須為偶數
        """
        self.is_self_play = is_self_play
        self.render_mode = render_mode
        self.board_size = board_size
        self.current_player = self.BLACK
        self.n_feature_planes = n_feature_planes
        self.action_space = spaces.Discrete(self.board_size**2+1)
        observation = self.reset()
        self.observation_space = spaces.Box(
            low=np.zeros(observation.shape), high=np.ones(observation.shape))

    def reset(self):
        """ 清空棋盤 """
        self.model = PPO.load("./gobang_game/model")#, env=self.model.get_env())
        self.prev_move = 0
        self.state = np.full(shape=(self.board_size**2), fill_value=-1, dtype=int)
        self.previous_actions = np.zeros(shape=(self.n_feature_planes*2, 2), dtype=int)# {step_count:action}
        self.done = False
        self.step_count = 0
        self.current_player = self.BLACK
        return self.get_observation()

    def single_step(self, action):
        """ 
        Parameters
        ----------
        action
            落子位置，範圍為 `[0, board_size**2]`
        """
        # 1.偵測錯誤及處理(reward=-1)
        winner = None
        if action not in list(range(self.board_size**2+1)) or action == 0:
            self.done = True

        if not self.done and self.state[action-1]!=self.EMPTY:
            self.done = True
            
        if self.done:
            reward = -1
            winner = self.WHITE + self.BLACK - self.current_player
            return self.get_observation(), reward, self.done, {'winner': winner, 'step_count': self.step_count}
        # 2.step_count+1及action儲存、sorting
        self.step_count += 1
        self.prev_move = action
        index_min = np.argmin(self.previous_actions, axis=0)
        self.previous_actions[index_min[0]] = np.array([self.step_count, action])
        observation = self.get_observation()
        # 3.落子並更新棋盤
        self.state[action-1] = self.current_player
        # 4.檢查是否結束(勝利、平手)
        if self.is_tie():
            winner = None
            reward = 0
            self.done = True
        elif self.is_done():
            winner = self.current_player
            reward = 1
            self.done = True
        else:
            self.current_player = self.WHITE + self.BLACK - self.current_player
            reward = 0
            self.done = False

        # 5.回傳狀態、獎勵等資訊
        return observation, reward, self.done, {'winner': winner, 'step_count': self.step_count}

    def step(self, action):

        obs, reward, done, info = self.single_step(action)
        if done:
            return obs, reward, done, info

        model_action, _state = self.model.predict(obs, deterministic=True)
        obs, reward, done, info = self.single_step(model_action)
        if not done and self.render_mode:
            print("model action:", model_action)
            self.render()
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        board = self.state
        outfile =  StringIO() if mode == 'ansi' else sys.stdout

        outfile.write('To play: ')
        outfile.write('black' if self.current_player == self.BLACK else 'white')
        outfile.write('\n')
        d = self.board_size#board.shape[1]
        if d > 9:
            outfile.write(' ' * 24)
            for j in range(10, d + 1):
                outfile.write(' ' + str(int(j/10)))
            outfile.write('\n')
        outfile.write(' ' * 6)
        for j in range(d):
            outfile.write(' ' + str((j + 1) % 10))
        outfile.write('\n')
        outfile.write(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+\n')
        for i in range(d):
            outfile.write(' ' * (3 if i < 9 else 2) + str(i + 1) + ' | ')
            for j in range(d):
                #step = self.previous_actions[self.previous_actions.argmax(axis=0)[0]][0]
                if board[i*self.board_size+j] == self.EMPTY:
                    outfile.write('. ')
                elif board[i*self.board_size+j] == self.BLACK:
                    if self.prev_move == i*self.board_size+j+1:
                        outfile.write('X)')
                    else:
                        outfile.write('X ')
                else:
                    if self.prev_move == i*self.board_size+j+1:
                        outfile.write('O)')
                    else:
                        outfile.write('O ')
            outfile.write('|\n')
        outfile.write(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+\n')
    
    def is_done(self):
        # 如果下子數不足9子，則遊戲一定還沒結束
        if self.step_count<9:
            return False

        n = self.board_size
        act = self.previous_actions[self.previous_actions.argmax(axis=0)[0]][1] - 1
        row, col = act//n, act % n

        # 搜尋方向
        directions = [[(0, -1),  (0, 1)],   # 水平搜尋
                      [(-1, 0),  (1, 0)],   # 垂直搜尋
                      [(-1, -1), (1, 1)],   # 主對角搜尋
                      [(1, -1),  (-1, 1)]]  # 副對角搜尋

        for i in range(4):
            count = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if 0 <= row_t < n and 0 <= col_t < n and self.state[row_t*n+col_t] == self.current_player:
                        # 遇到相同顏色時 count+1
                        count += 1
                    else:
                        flag = False
            # 分出勝負
            if count >= 5:
                return True

        return False

    def is_tie(self):
        return self.state[self.state.argmin()] > self.EMPTY
        
    def get_observation(self):
        # 開始疊棋盤落子狀態:自己n/2個，對手n/2個
        feature_planes = torch.zeros(self.n_feature_planes+1, self.board_size**2, dtype=torch.int8)
        n = self.n_feature_planes//2
        index_sorted = np.argsort(self.previous_actions, axis=0)[::-1]
        for i in range(n):
            # 自己
            feature_planes[i] = self.make_board(index=index_sorted, start=i*2)
            # 對手
            feature_planes[i+n] = self.make_board(index=index_sorted, start=i*2+1)

        # 加上己方的特徵平面(BLACK為1, WHITE為0)
        feature_planes[self.n_feature_planes] = torch.full(size=(self.board_size**2,), fill_value=self.current_player, dtype=torch.int8)

        return feature_planes.view(self.n_feature_planes+1, self.board_size, self.board_size)
    
    def make_board(self, index, start=0):
        # 取得過去n_feature_planes/2步的棋盤狀態
        feature = torch.zeros(self.board_size**2, dtype=torch.int8)
        for i in range(self.n_feature_planes//2):
            act = self.previous_actions[index[start+i][0]][1]
            if act != 0:
                feature[act-1] = 1
            start+=1
        return feature

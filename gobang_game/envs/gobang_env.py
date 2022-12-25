import sys
import gym
import torch
import numpy as np
from .node import Node
from six import StringIO
from stable_baselines3 import PPO
from gym import spaces
from copy import deepcopy
from random import sample

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
            reward = -100
            winner = self.WHITE + self.BLACK - self.current_player
            return self.get_observation(), reward, self.done, {'winner': winner, 'step_count': self.step_count}
        # 2.step_count+1及action儲存、sorting
        self.step_count += 1
        self.prev_move = action
        if self.n_feature_planes>0:
            index_min = np.argmin(self.previous_actions, axis=0)
            self.previous_actions[index_min[0]] = np.array([self.step_count, action])
        observation = self.get_observation()
        # 3.落子並更新棋盤
        self.state[action-1] = self.current_player
        # 4.檢查是否結束(勝利、平手)
        if self.is_tie():
            winner = None
            reward = 50
            self.done = True
        elif self.is_done():
            winner = self.current_player
            reward = 100
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
        if self.n_feature_planes>0:
            act = self.previous_actions[self.previous_actions.argmax(axis=0)[0]][1] - 1
        else:
            act = self.prev_move - 1
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
        feature_planes = torch.zeros(self.n_feature_planes+2, self.board_size**2, dtype=torch.int8)
        n = self.n_feature_planes//2
        index_sorted = np.argsort(self.previous_actions, axis=0)[::-1]
        for i in range(n):
            # 自己
            feature_planes[i] = self.make_board(index=index_sorted, start=i*2)
            # 對手
            feature_planes[i+n] = self.make_board(index=index_sorted, start=i*2+1)

        # 加上己方的特徵平面(BLACK為1, WHITE為0)
        feature_planes[-1] = torch.full(size=(self.board_size**2,), fill_value=self.current_player, dtype=torch.int8)
        # 加上目前棋盤狀態
        feature_planes[-2] = self.make_state()
        return feature_planes.view(self.n_feature_planes+2, self.board_size, self.board_size)
    
    def make_board(self, index, start=0):
        # 取得過去n_feature_planes/2步的棋盤狀態
        feature = torch.zeros(self.board_size**2, dtype=torch.int8)
        for i in range(self.n_feature_planes//2):
            act = self.previous_actions[index[start+i][0]][1]
            if act != 0:
                feature[act-1] = 1
            start+=1
        return feature
        
    def make_state(self):
        feature = torch.full(size=(self.board_size**2,), fill_value=0 if self.n_feature_planes>0 else -1, dtype=torch.int8)
        for i in range(self.board_size**2):
            if self.state[i]!=self.EMPTY:
                if self.n_feature_planes>0:
                    feature[i] = 1
                else:
                    feature[i] = self.state[i]
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
        self.model = PPO.load("./selfplay/model")#, env=self.model.get_env())
        self.prev_move = 0
        self.state = np.full(shape=(self.board_size**2), fill_value=-1, dtype=int)
        self.previous_actions = np.zeros(shape=(self.n_feature_planes*2, 2), dtype=int)# {step_count:action}
        self.done = False
        self.step_count = 0
        if self.is_self_play:
            self.rand_player = sample([self.WHITE, self.BLACK], 1)[0]
        else:
            self.rand_player = self.BLACK
        self.current_player = self.rand_player#self.BLACK
        self.available_actions = list(range(self.board_size**2))
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
            reward = -100
            winner = self.WHITE + self.BLACK - self.current_player
            # debug
            print("Game Step:", self.step_count)
            return self.get_observation(), reward, self.done, {'winner': winner, 'step_count': self.step_count}
        # 2.step_count+1及action儲存、sorting
        self.step_count += 1
        self.prev_move = action
        self.available_actions.remove(action-1)
        if self.n_feature_planes>0:
            index_min = np.argmin(self.previous_actions, axis=0)
            self.previous_actions[index_min[0]] = np.array([self.step_count, action])
        observation = self.get_observation()
        # 3.落子並更新棋盤
        self.state[action-1] = self.current_player
        # 4.檢查是否結束(勝利、平手)
        if self.is_tie():
            winner = None
            reward = 50
            self.done = True
        elif self.is_done():
            winner = self.current_player
            reward = 200
            self.done = True
        else:
            self.current_player = self.WHITE + self.BLACK - self.current_player
            reward = 0
            self.done = False
        
        # 5.回傳狀態、獎勵等資訊
        return observation, reward, self.done, {'winner': winner, 'step_count': self.step_count}

    def step(self, action):
        if self.rand_player == self.BLACK:
            obs, reward, done, info = self.single_step(action)
            if done:
                return obs, reward, done, info

            if self.is_self_play:
                model_action, _state = self.model.predict(obs, deterministic=True)
                if model_action == 0 or (model_action-1) not in self.available_actions:
                    model_action = self.available_action_sample()
                obs, model_reward, done, info = self.single_step(model_action)
                if not done and self.render_mode:
                    print("model action:", model_action)
                    self.render()
                if done:
                    reward = -model_reward
        else:
            if self.is_self_play:
                model_action, _state = self.model.predict(self.get_observation(), deterministic=True)
                if model_action == 0 or (model_action-1) not in self.available_actions:
                    model_action = self.available_action_sample()
                obs, model_reward, done, info = self.single_step(model_action)
                if not done and self.render_mode:
                    print("model action:", model_action)
                    self.render()
            obs, reward, done, info = self.single_step(action)
            if done:
                return obs, reward, done, info

        return obs, reward, done, info
        
    def available_action_sample(self):
        action = sample(self.available_actions, 1)
        return action[0]+1

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
        if self.n_feature_planes>0:
            act = self.previous_actions[self.previous_actions.argmax(axis=0)[0]][1] - 1
        else:
            act = self.prev_move - 1
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
        feature_planes = torch.zeros(self.n_feature_planes+2, self.board_size**2, dtype=torch.int8)
        n = self.n_feature_planes//2
        index_sorted = np.argsort(self.previous_actions, axis=0)[::-1]
        for i in range(n):
            # 自己
            feature_planes[i] = self.make_board(index=index_sorted, start=i*2)
            # 對手
            feature_planes[i+n] = self.make_board(index=index_sorted, start=i*2+1)

        # 加上己方的特徵平面(BLACK為1, WHITE為0)
        feature_planes[-1] = torch.full(size=(self.board_size**2,), fill_value=self.current_player, dtype=torch.int8)
        # 加上目前棋盤狀態
        feature_planes[-2] = self.make_state()
        return feature_planes.view(self.n_feature_planes+2, self.board_size, self.board_size)
    
    def make_board(self, index, start=0):
        # 取得過去n_feature_planes/2步的棋盤狀態
        feature = torch.zeros(self.board_size**2, dtype=torch.int8)
        for i in range(self.n_feature_planes//2):
            act = self.previous_actions[index[start+i][0]][1]
            if act != 0:
                feature[act-1] = 1
            start+=1
        return feature

    def make_state(self):
        feature = torch.full(size=(self.board_size**2,), fill_value=0 if self.n_feature_planes>0 else -1, dtype=torch.int8)
        for i in range(self.board_size**2):
            if self.state[i]!=self.EMPTY:
                if self.n_feature_planes>0:
                    feature[i] = 1
                else:
                    feature[i] = self.state[i]
        return feature

class MCTSGobangEnv(gym.Env):

    EMPTY = -1
    WHITE = 0
    BLACK = 1
    
    def __init__(self, board_size=9, n_feature_planes=6, is_self_play=False, render_mode=False, c_puct: float = 4, n_iters=1200):
        """
        Parameters
        ----------
        board_size: int
            棋盤長
        n_feature_planes: int
            特征平面的個數，必須為偶數
        """

        self.c_puct = c_puct
        self.n_iters = n_iters
        self.root = Node(prior_prob=1, parent=None)

        self.is_self_play = is_self_play
        self.render_mode = render_mode

        self.board_size = board_size
        self.current_player = self.BLACK
        self.n_feature_planes = n_feature_planes
        self.action_space = spaces.Discrete(self.board_size**2+1)
        observation = self.reset()
        self.observation_space = spaces.Box(
            low=np.zeros(observation.shape), high=np.ones(observation.shape))
    def best_action(self):
        '''根據目前局面用MCTS回傳下一步最佳動作'''
        for _ in range(self.n_iters):
            env = deepcopy(self)
            # 未到達葉節點前一直向下探索並更新棋盤
            node = self.root
            while not node.is_leaf_node():
                action, node = node.select()
                obs, reward, done, info = env.step(action+1)

            # 判斷是否結束，沒有就expand葉節點
            p, value = self.model.policy.forward(obs=env.get_observation())
            if not done:
                # 加入dirichlet noise
                if self.is_self_play:
                    p = 0.75*p+0.25*\
                        np.random.dirichlet(0.03*np.ones(len(p)))
                    node.expand(zip(self.available_actions, p))
            elif info['winner'] is not None:
                value = 1 if info['winner'] == env.current_player else -1
            else:
                value = 0
            
            # Backprpogation
            node.backup(-value)
        
        # 在self-play時，讓前30步有較大的探索空間
        T = 1 if self.is_self_play and self.step_count <= 30 else 1e-3
        visits = np.array([node.N for node in self.root.children.values()])
        pi_ = self.__getPi(visits, T)

        # 根據pi的大小決定動作
        actions = list(self.root.children.keys())
        action = int(np.random.choice(actions, p=pi_))

        if self.is_self_play:
            # 更新node值
            pi = np.zeros(self.board_size**2)
            pi[actions] = pi_
            self.root = self.root.children[action]
            self.root.parent = None
            return action, pi
        else:
            self.reset_root()
            return action

    def __getPi(self, visits, T) -> np.ndarray:
        """ 紀錄node的visit次數並計算探索可能值pi """
        x = 1/T * np.log(visits + 1e-11)
        # 以期望值當作pi
        x = np.exp(x - x.max())
        pi = x/x.sum()
        return pi

    def reset_root(self):
        self.root = Node(prior_prob=1, c_puct=self.c_puct, parent=None)


    def mcts_step(self, state, action, current_player):
        '''模擬探索，並不會真的更新自身的環境狀態'''
        #　複製棋盤狀態
        state = deepcopy(state)
        current_player = current_player
         # 1.偵測錯誤及處理(reward=-1)
        winner = None
        done = False
        if action not in list(range(self.board_size**2+1)) or action == 0:
            done = True

        if not done and state[action-1]!=self.EMPTY:
            done = True
            
        if done:
            reward = -1
            winner = self.WHITE + self.BLACK - current_player
            return state, reward, done, {'winner': winner}
       
        # 3.落子並更新棋盤
        state[action-1] = current_player
        # 4.檢查是否結束(勝利、平手)
        if state[state.argmin()] > self.EMPTY:
            winner = None
            reward = 0
            done = True
        elif self.mcts_is_done(state, action, current_player):
            winner = current_player
            reward = 1
            done = True
        else:
            current_player = self.WHITE + self.BLACK - current_player
            reward = 0
            done = False

        return state, reward, done, {'winner': winner, 'current_player':current_player}

    def reset(self):
        """ 清空棋盤 """
        self.model = PPO.load("./selfplay/model", device='cpu')#, env=self.model.get_env())
        self.prev_move = 0
        self.state = np.full(shape=(self.board_size**2), fill_value=-1, dtype=int)
        self.previous_actions = np.zeros(shape=(self.n_feature_planes*2, 2), dtype=int)# {step_count:action}
        self.done = False
        self.step_count = 0
        self.current_player = self.BLACK
        self.available_actions = list(range(self.board_size**2))
        self.best_action()
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
        self.available_actions.remove(action-1)
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

        if self.is_self_play:
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
    def mcts_is_done(self, state, action, current_player):

        n = self.board_size
        act = action - 1
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
                    if 0 <= row_t < n and 0 <= col_t < n and state[row_t*n+col_t] == current_player:
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
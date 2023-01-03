# coding:utf-8
import json
import os
import time
import traceback
import threading

import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from .alpha_zero_mcts import AlphaZeroMCTS
from .chess_board import ChessBoard
from .policy_value_net import PolicyValueNet
from .self_play_dataset import SelfPlayData, SelfPlayDataSet


def exception_handler(train_func):
    """ 異常處理裝飾器 """
    def wrapper(train_pipe_line, *args, **kwargs):
        try:
            train_func(train_pipe_line)
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()

            t = time.strftime('%Y-%m-%d_%H-%M-%S',
                              time.localtime(time.time()))
            train_pipe_line.save_model(
                f'last_policy_value_net_{t}.pth', 'train_losses', 'games')

    return wrapper


class PolicyValueLoss(nn.Module):
    """ 根據 self-play 產生的 `z` 和 `π` 計算誤差 """

    def __init__(self):
        super().__init__()

    def forward(self, p_hat, pi, value, z):
        """ 前饋

        Parameters
        ----------
        p_hat: Tensor of shape (N, board_len^2)
            對數動作概率向量

        pi: Tensor of shape (N, board_len^2)
            `mcts` 產生的動作概率向量

        value: Tensor of shape (N, )
            對每個局面的估值

        z: Tensor of shape (N, )
            最終的遊戲結果相對每一個玩家的獎賞
        """
        value_loss = F.mse_loss(value, z)
        policy_loss = -torch.sum(pi*p_hat, dim=1).mean()
        loss = value_loss + policy_loss
        return loss


class TrainModel:
    """ 訓練模型 """

    def __init__(self, board_len=9, lr=0.0001, n_self_plays=1500, n_mcts_iters=500,
                 n_feature_planes=4, batch_size=500, start_train_size=500, check_frequency=100,
                 n_test_games=10, c_puct=4, is_use_gpu=True, is_save_game=False, **kwargs):
        """
        Parameters
        ----------
        board_len: int
            棋盤大小

        lr: float
            學習率

        n_self_plays: int
            自我博弈遊戲局數

        n_mcts_iters: int
            蒙特卡洛樹搜索次數

        n_feature_planes: int
            特徵平面個數

        batch_size: int
            mini-batch 的大小

        start_train_size: int
            開始訓練模型時的最小數據集尺寸

        check_frequency: int
            測試模型的頻率

        n_test_games: int
            測試模型時與歷史最優模型的比賽局數

        c_puct: float
            探索常數

        is_use_gpu: bool
            是否使用 GPU

        is_save_game: bool
            是否保存自對弈的棋譜
        """
        self.c_puct = c_puct
        self.is_use_gpu = is_use_gpu
        self.batch_size = batch_size
        self.n_self_plays = n_self_plays
        self.n_test_games = n_test_games
        self.n_mcts_iters = n_mcts_iters
        self.is_save_game = is_save_game
        self.check_frequency = check_frequency
        self.start_train_size = start_train_size
        self.device = torch.device(
            'cuda:0' if is_use_gpu and cuda.is_available() else 'cpu')
        self.chess_board = ChessBoard(board_len, n_feature_planes)

        # 創建策略-價值網絡和蒙特卡洛搜索樹
        self.policy_value_net = self.__get_policy_value_net(board_len)
        self.mcts = AlphaZeroMCTS(
            self.policy_value_net, c_puct=c_puct, n_iters=n_mcts_iters, is_self_play=True)

        # 創建優化器和損失函數
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = PolicyValueLoss()
        self.lr_scheduler = MultiStepLR(
            self.optimizer, [1600, 2500], gamma=0.5)

        # 創建數據集
        self.dataset = SelfPlayDataSet(board_len)

        # 記錄數據
        self.train_losses = self.__load_data('log/train_losses2.json')
        self.games = self.__load_data('log/games2.json')

    def __self_play(self):
        """ 自我博弈一局

        Returns
        -------
        self_play_data: namedtuple
            自我博弈數據，有以下三個成員:
            * `pi_list`: 蒙特卡洛樹搜索產生的動作概率向量 π 組成的列表
            * `z_list`: 一局之中每個動作的玩家相對最後的遊戲結果的獎賞列表
            * `feature_planes_list`: 一局之中每個動作對應的特徵平面組成的列表
        """
        # 初始化棋盤和數據容器
        self.policy_value_net.eval()
        self.chess_board.clear_board()
        pi_list, feature_planes_list, players = [], [], []
        action_list = []

        # 開始一局遊戲
        while True:
            action, pi = self.mcts.get_action(self.chess_board)

            # 保存每一步的數據
            feature_planes_list.append(self.chess_board.get_feature_planes())
            players.append(self.chess_board.current_player)
            action_list.append(action)
            pi_list.append(pi)
            self.chess_board.do_action(action)

            # 判斷遊戲是否結束
            is_over, winner = self.chess_board.is_game_over()
            if is_over:
                if winner is not None:
                    z_list = [1 if i == winner else -1 for i in players]
                else:
                    z_list = [0]*len(players)
                break

        # 重置根節點
        self.mcts.reset_root()

        # 返回數據
        if self.is_save_game:
            self.games.append(action_list)

        self_play_data = SelfPlayData(
            pi_list=pi_list, z_list=z_list, feature_planes_list=feature_planes_list)
        return self_play_data
    
    @exception_handler
    def train(self):
        """ 訓練模型 """
        for i in range(self.n_self_plays):
            print(f'🏹 正在進行第 {i*2+1} & {i*2+2} 局自我博弈遊戲...')
            threads = []
            for t in range(2):
                threads.append(threading.Thread(target = self.job, args = (i*2+t,)))
                threads[t].start()
            for t in range(2):
                threads[t].join()
            self.dataset.append(self.__self_play())

            # 如果數據集中的數據量大於 start_train_size 就進行一次訓練
            if len(self.dataset) >= self.start_train_size:
                data_loader = iter(DataLoader(
                    self.dataset, self.batch_size, shuffle=True, drop_last=False))
                print('💊 開始訓練...')

                self.policy_value_net.train()
                # 隨機選出一批數據來訓練，防止過擬合
                feature_planes, pi, z = next(data_loader)
                feature_planes = feature_planes.to(self.device)
                pi, z = pi.to(self.device), z.to(self.device)
                for _ in range(5):
                    # 前饋
                    p_hat, value = self.policy_value_net(feature_planes)
                    # 梯度清零
                    self.optimizer.zero_grad()
                    # 計算損失
                    loss = self.criterion(p_hat, pi, value.flatten(), z)
                    # 誤差反向傳播
                    loss.backward()
                    # 更新參數
                    self.optimizer.step()
                    # 學習率退火
                    self.lr_scheduler.step()

                # 記錄誤差
                self.train_losses.append([i, loss.item()])
                print(f"🚩 train_loss = {loss.item():<10.5f}\n")

            # 測試模型
            if (i+1) % self.check_frequency == 0:
                self.__test_model()

    def __test_model(self):
        """ 測試模型 """
        os.makedirs('model', exist_ok=True)

        model_path = 'model/best_policy_value_net.pth'

        # 如果最佳模型不存在保存當前模型為最佳模型
        if not os.path.exists(model_path):
            torch.save(self.policy_value_net, model_path)
            return

        # 載入歷史最優模型
        best_model = torch.load(model_path)  # type:PolicyValueNet
        best_model.eval()
        best_model.set_device(self.is_use_gpu)
        mcts = AlphaZeroMCTS(best_model, self.c_puct, self.n_mcts_iters)
        self.mcts.set_self_play(False)
        self.policy_value_net.eval()

        # 開始比賽
        print('🩺 正在測試當前模型...')
        n_wins = 0
        for i in range(self.n_test_games):
            self.chess_board.clear_board()
            self.mcts.reset_root()
            mcts.reset_root()
            while True:
                # 當前模型走一步
                is_over, winner = self.__do_mcts_action(self.mcts)
                if is_over:
                    n_wins += int(winner == ChessBoard.BLACK)
                    break
                # 歷史最優模型走一步
                is_over, winner = self.__do_mcts_action(mcts)
                if is_over:
                    break

        # 如果勝率大於 55%，就保存當前模型為最優模型
        win_prob = n_wins/self.n_test_games
        if win_prob > 0.55:
            torch.save(self.mcts.policy_value_net, model_path)
            print(f'🥇 保存當前模型為最優模型，當前模型勝率為：{win_prob:.1%}\n')
        else:
            print(f'🎃 保持歷史最優模型不變，當前模型勝率為：{win_prob:.1%}\n')

        self.mcts.set_self_play(True)

    def save_model(self, model_name: str, loss_name: str, game_name: str):
        """ 保存模型

        Parameters
        ----------
        model_name: str
            模型文件名稱，不包含後綴

        loss_name: str
            損失文件名稱，不包含後綴

        game_name: str
            自對弈棋譜名稱，不包含後綴
        """
        os.makedirs('model', exist_ok=True)

        path = f'model/{model_name}.pth'
        self.policy_value_net.eval()
        torch.save(self.policy_value_net, path)
        print(f'🎉 已將當前模型保存到 {os.path.join(os.getcwd(), path)}')

        # 保存數據
        with open(f'log/{loss_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_losses, f)

        if self.is_save_game:
            with open(f'log/{game_name}.json', 'w', encoding='utf-8') as f:
                json.dump(self.games, f)


    def __do_mcts_action(self, mcts):
        """ 獲取動作 """
        action = mcts.get_action(self.chess_board)
        self.chess_board.do_action(action)
        is_over, winner = self.chess_board.is_game_over()
        return is_over, winner

    def __get_policy_value_net(self, board_len=9):
        """ 創建策略-價值網絡，如果存在歷史最優模型則直接載入最優模型 """
        os.makedirs('model', exist_ok=True)

        best_model = 'history/best_policy_value_net_4400.pth'
        history_models = sorted(
            [i for i in os.listdir('model') if i.startswith('last')])

        # 從歷史模型中選取最新模型
        model = history_models[-1] if history_models else best_model
        model = f'model/{model}'
        if os.path.exists(model):
            print(f'💎 載入模型 {model} ...\n')
            net = torch.load(model).to(self.device)  # type:PolicyValueNet
            net.set_device(self.is_use_gpu)
        else:
            net = PolicyValueNet(n_feature_planes=self.chess_board.n_feature_planes,
                                 is_use_gpu=self.is_use_gpu, board_len=board_len).to(self.device)

        return net

    def __load_data(self, path: str):
        """ 載入歷史損失數據 """
        data = []
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except:
            os.makedirs('log', exist_ok=True)

        return data

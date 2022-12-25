# !/usr/bin/env python
# coding=utf-8

import operator
from typing import List
from sys import exit
import pygame
from pygame.locals import *

# 全域變數
black = 1
white = -1
blank = 0
tie = 2

class Game:
    def __init__(self, enable_pygame=True, **kwargs):
        self.enable_pygame = enable_pygame
        self.board_count = kwargs['board_count'] if 'board_count' in kwargs else 9   # 棋盤一行有幾個棋子
        self.line_margin = kwargs['line_margin'] if 'line_margin' in kwargs else 40  # 兩條線之間的距離
        self.ignore_wait = True if 'ignore_wait' in kwargs else False
        self.board_size = self.line_margin * (self.board_count + 1)  # 棋盤大小

        if self.enable_pygame:
            pygame.init()
            self.screen_width = self.board_size # 視窗寬度
            self.screen_height = self.board_size+self.line_margin   # 視窗高度
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 0, 32)
            pygame.display.set_caption("Beta Gobang")
            self.font = pygame.font.SysFont("arial", 32)
        
        # 棋盤和棋子的顏色
        self.cheese_board = [[0] * self.board_count for _ in range(self.board_count)]
        self.piece_color = black

    def reset(self):
        """
        重置環境：將棋盤置為空，將棋子顏色置為黑色，重新繪製screen（如果需要）
        """
        # 重置棋盤
        self.cheese_board = [[0] * self.board_count for _ in range(self.board_count)]
        # 重置棋子顏色
        self.piece_color = black
        if not self.enable_pygame:
            return self.get_obs()
        # 重新繪製screen
        background_color, line_color = (238, 154, 73), (0, 0, 0)
        screen = self.screen
        border_count, line_margin = self.board_count, self.line_margin
        screen.fill(background_color)
        board_size = self.board_size
        # 畫橫線
        for row in range(border_count):
            if row == 0 or row == border_count-1:   # 外圍較粗
                pygame.draw.line(screen, line_color, (line_margin, (row + 1) * line_margin), (board_size-line_margin, (row + 1) * line_margin), 2)
            else:
                pygame.draw.line(screen, line_color, (line_margin, (row + 1) * line_margin), (board_size-line_margin, (row + 1) * line_margin))
        # 畫直線   
        for col in range(border_count):
            if col == 0 or col == border_count-1:   # 外圍較粗
                pygame.draw.line(screen, line_color, ((col + 1) * line_margin, line_margin), ((col + 1) * line_margin, board_size-line_margin), 2)
            else:
                pygame.draw.line(screen, line_color, ((col + 1) * line_margin, line_margin), ((col + 1) * line_margin, board_size-line_margin))
        # 畫中心點
        pygame.draw.circle(screen, line_color, [int(board_size/2), int(board_size/2)], 4, 0)
        # 更新畫布
        pygame.display.update()
        
        return self.get_obs()

    def point_convert(self, px: int, py: int, point_thres: float = 0.375):
        """
        將滑鼠點擊的坐標轉換成棋盤上的坐標
        :param px: 滑鼠點擊的 x 座標
        :param py: 滑鼠點擊的 y 座標
        :param point_thres: 誤差值
        :return: 棋盤上的坐標
        """
        line_margin = self.line_margin
        line_thres = line_margin * point_thres
        for row in range(self.board_count):
            for col in range(self.board_count):
                if (row + 1) * line_margin - line_thres < py < (row + 1) * line_margin + line_thres and \
                        (col + 1) * line_margin - line_thres < px < (col + 1) * line_margin + line_thres:
                    return row, col
        # 如果找不到，回傳 -1, -1
        return -1, -1

    def has_piece(self, _pos) -> bool:
        """
        判斷給定位置是否有棋子
        :param _pos: 給定棋子位置
        :return: 是否有棋子
        """
        row, col = _pos
        return self.cheese_board[row][col] != 0

    def piece_down(self, row: int, col: int, c: int in [1, -1], pre_action):
        """
        Pygame 繪製落下的棋子
        :param row: 棋子所在的行
        :param col: 棋子所在的列
        :param c: 棋子顏色
        """
        piece_color = (0, 0, 0) if c == black else (255, 255, 255)
        line_margin = self.line_margin
        self.cheese_board[row][col] = c  # 紀錄顏色
        # Pygame 繪製棋子
        if self.enable_pygame:
            if pre_action != None:
                previous_row, previous_col = pre_action
                self.piece_box(previous_row, previous_col, blank)   # 清掉原本的提示框
            
            self.piece_box(row, col, c) # 繪製新的提示框
            pygame.draw.circle(self.screen, piece_color, ((col + 1) * line_margin, (row + 1) * line_margin), 17)    # 繪製新的落子
            pygame.display.update()
            
    def piece_box(self, row: int, col: int, c: int in [1, -1, 0]):
        """
        上一個落子的提示框
        :param row: 棋子所在的行
        :param col: 棋子所在的列
        :param c: 提示框顏色
        """
        # 提示框顏色
        if c == black:
            line_color = (255, 0, 0)
        elif c == white:
            line_color = (0, 0, 255)
        elif c == blank:
            line_color = (238, 154, 73)
        # 落子中心座標
        screen = self.screen
        line_margin = self.line_margin
        line_width = 2
        x = (col + 1) * line_margin
        y = (row + 1) * line_margin
        # 八條線
        pygame.draw.line(screen, line_color, (x-int(line_margin/2), y-int(line_margin/2)), (x-int(line_margin/4), y-int(line_margin/2)), line_width)
        pygame.draw.line(screen, line_color, (x-int(line_margin/2), y-int(line_margin/2)), (x-int(line_margin/2), y-int(line_margin/4)), line_width)
        pygame.draw.line(screen, line_color, (x-int(line_margin/2), y+int(line_margin/2)), (x-int(line_margin/4), y+int(line_margin/2)), line_width)
        pygame.draw.line(screen, line_color, (x-int(line_margin/2), y+int(line_margin/2)), (x-int(line_margin/2), y+int(line_margin/4)), line_width)
        pygame.draw.line(screen, line_color, (x+int(line_margin/2), y-int(line_margin/2)), (x+int(line_margin/4), y-int(line_margin/2)), line_width)
        pygame.draw.line(screen, line_color, (x+int(line_margin/2), y-int(line_margin/2)), (x+int(line_margin/2), y-int(line_margin/4)), line_width)
        pygame.draw.line(screen, line_color, (x+int(line_margin/2), y+int(line_margin/2)), (x+int(line_margin/4), y+int(line_margin/2)), line_width)
        pygame.draw.line(screen, line_color, (x+int(line_margin/2), y+int(line_margin/2)), (x+int(line_margin/2), y+int(line_margin/4)), line_width)
        
    def pygame_settle(self, winner):
        """
        Pygame 繪製遊戲結果
        :param winner: 勝利者
        :return:
        """
        if winner == black:
            content = 'You Win!'
            content_color = (0, 0, 0)
        elif winner == white:
            content = 'AI Win!'
            content_color = (255, 255, 255)
        elif winner == tie:
            content = 'Tie!'
            content_color = (128, 128, 128)
            
        # 印出文字
        screen, font = self.screen, self.font
        text = font.render(content, True, content_color)
        text_rect = text.get_rect(center=(int(self.board_size/2), self.screen_height-self.line_margin))
        screen.blit(text, text_rect)
        
        pygame.display.update()
        
        # 清空事件，等待下一局
        if not self.ignore_wait:
            pygame.event.clear()
            while True:
                for event in pygame.event.get():
                    if event.type == MOUSEBUTTONDOWN:
                        break
                    if event.type == KEYDOWN:
                        break
                    if event.type == QUIT:
                        exit()
                else:
                    continue
                break

    def step(self, _action, pre_action):
        """
        執行一步動作，回傳環境狀態
        :param _action: 落子座標 (x, y)
        :return: 環境狀態
        """
        piece_x, piece_y = _action
        # 落子
        self.piece_down(piece_x, piece_y, self.piece_color, pre_action)
        # 判斷遊戲是否結束
        result = self.game_end(self.cheese_board)
        # 如果遊戲結束，則 Pygame 繪製結果
        if result != 0 and self.enable_pygame:
            self.pygame_settle(result)
        # 切換棋子顏色
        self.piece_color = -self.piece_color
        
        return self.cheese_board, result
    
    def game_end(self, cheeses: List[List[int]]) -> int in [1, -1, 0]:
        """
        判断勝負
        :return: 勝利者 {1: 黑棋, -1: 白棋, 0: 繼續遊戲}
        """
        board_count = len(cheeses)
        border_res = 4  # 因為是五子棋，只需要看四個就夠了
        black_five, white_five = [black for _ in range(5)], [white for _ in range(5)]
        # 縱向判斷是否有五子連現
        for row in range(board_count - border_res):
            for column in range(board_count):
                five_pieces = [cheeses[row + i][column] for i in range(5)]  # 縱向五子
                if operator.eq(five_pieces, black_five):  # 黑勝
                    return black
                if operator.eq(five_pieces, white_five):  # 白勝
                    return white
        # 橫向判斷是否有五子連環
        for row in range(board_count):
            for column in range(board_count - border_res):
                five_pieces = [cheeses[row][column + j] for j in range(5)]  # 横向五子
                if operator.eq(five_pieces, black_five):  # 黑勝
                    return black
                if operator.eq(five_pieces, white_five):  # 白勝
                    return white
        # 右下斜向判斷是否有五子連線
        for row in range(board_count - border_res):
            for column in range(board_count - border_res):
                five_pieces = [cheeses[row + i][column + i] for i in range(5)]  # 右下斜向五子
                if operator.eq(five_pieces, black_five):  # 黑勝
                    return black
                if operator.eq(five_pieces, white_five):  # 白勝
                    return white
        # 左下斜向判斷是否有五子連線
        for row in range(border_res, board_count):
            for column in range(board_count - border_res):
                five_pieces = [cheeses[row - i][column + i] for i in range(5)]  # 左下斜向五子
                if operator.eq(five_pieces, black_five):  # 黑勝
                    return black
                if operator.eq(five_pieces, white_five):  # 白勝
                    return white
        # 棋盤滿了，則平手
        if sum([cheese_board_row.count(0) for cheese_board_row in cheeses]) == 0:
            return tie
        
        return 0

    def get_color(self):
        """
        回傳目前回合的顏色
        """
        return self.piece_color

    def get_obs(self):
        """
        回傳棋盤狀態
        :return: 棋盤
        """
        return self.cheese_board


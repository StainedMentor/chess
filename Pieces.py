from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsItem

from constants import *
from util import *




class Piece(QGraphicsItem):
    def __init__(self, type, x, y):
        super().__init__()
        self.image = QPixmap("pieces/Chess_"+type+"t60.png").scaledToWidth(50)
        self.didMove = False
        self.type = type
        self.x = x
        self.y = y
        self.isWhite = type[-1]=="l"

    def boundingRect(self):
        return QRectF(self.image.rect())

    def paint(self, painter, option, widget):
        painter.drawPixmap(self.image.rect(), self.image)



    def drawSelf(self, scene):
        scene.addItem(self)
        self.setPos(self.x * 50, self.y * 50)

    def move(self,x,y):
        self.didMove = True
        self.x = x
        self.y = y
        self.setPos(self.x * 50, self.y * 50)


    def select(self):
        self.setPos(self.x * 50+10, self.y * 50+10)

    def deselect(self):
        self.setPos(self.x * 50, self.y * 50)

    def delete(self, scene):
        scene.removeItem(self)

    def getAvailableMoves(self, board):
        match self.type[0]:
            case "p":
                return self.pawnMoves(board)
            case "r":
                return self.rookMoves(board)
            case "n":
                return self.knightMoves(board)
            case "b":
                return self.bishopMoves(board)
            case "k":
                return self.kingMoves(board)
            case "q":
                return self.queenMove(board)

    def pawnMoves(self,board):
        moves = []
        dir = -1 if self.isWhite else 1

        pos1 = (self.x, self.y + 1 * dir)
        if board[pos1[1]][pos1[0]] is None:
            moves.append(pos1)

            if self.y == 6 or self.y == 1:
                pos2 = (self.x,self.y+2*dir)
                if board[pos2[1]][pos2[0]] is None:

                    moves.append(pos2)

        pos3 = (self.x+1, self.y + 1 * dir)
        pos4 = (self.x-1, self.y + 1 * dir)
        if pos3[0] in range(8) and pos3[1] in range(8):
            if board[pos3[1]][pos3[0]] is not None:
                moves.append(pos3)
        if pos4[0] in range(8) and pos4[1] in range(8):
            if board[pos4[1]][pos4[0]] is not None:
                moves.append(pos4)

        return moves





    def rookMoves(self,board):
        moves = []
        blockedDir = [False,False,False,False]
        for i in range(1,8):
            temp = []
            temp.append((self.x+i,self.y))
            temp.append((self.x-i,self.y))
            temp.append((self.x,self.y+i))
            temp.append((self.x,self.y-i))

            for index, pos in enumerate(temp):
                if pos[0] in range(8) and pos[1] in range(8):
                    if blockedDir[index]:
                        continue
                    if board[pos[1]][pos[0]] is not None:
                        blockedDir[index] = True

                        if board[pos[1]][pos[0]].isWhite == self.isWhite:
                            continue
                        else:
                            moves.append(pos)

                    moves.append(pos)

        return moves

    def bishopMoves(self,board):
        moves = []
        blockedDir = [False,False,False,False]
        for i in range(1,8):
            temp = []
            temp.append((self.x+i,self.y+i))
            temp.append((self.x-i,self.y-i))
            temp.append((self.x-i,self.y+i))
            temp.append((self.x+i,self.y-i))

            for index, pos in enumerate(temp):
                if pos[0] in range(8) and pos[1] in range(8):
                    if blockedDir[index]:
                        continue
                    if board[pos[1]][pos[0]] is not None:
                        blockedDir[index] = True

                        if board[pos[1]][pos[0]].isWhite == self.isWhite:
                            continue
                        else:
                            moves.append(pos)

                    moves.append(pos)

        return moves

    def knightMoves(self,board):
        moves = []
        temp = []
        temp.append((self.x + 2, self.y + 1))
        temp.append((self.x + 2, self.y - 1))
        temp.append((self.x - 2, self.y + 1))
        temp.append((self.x - 2, self.y - 1))
        temp.append((self.x - 1, self.y + 2))
        temp.append((self.x + 1, self.y + 2))
        temp.append((self.x - 1, self.y - 2))
        temp.append((self.x + 1, self.y - 2))


        for pos in temp:
            if pos[0] in range(8) and pos[1] in range(8):
                if board[pos[1]][pos[0]] is not None:
                    if board[pos[1]][pos[0]].isWhite == self.isWhite:
                        continue

                moves.append(pos)


        return moves


    def queenMove(self,board):
        moves = []

        blockedDir = [False,False,False,False]
        for i in range(1,8):
            temp = []
            temp.append((self.x+i,self.y))
            temp.append((self.x-i,self.y))
            temp.append((self.x,self.y+i))
            temp.append((self.x,self.y-i))

            for index, pos in enumerate(temp):
                if pos[0] in range(8) and pos[1] in range(8):
                    if blockedDir[index]:
                        continue
                    if board[pos[1]][pos[0]] is not None:
                        blockedDir[index] = True

                        if board[pos[1]][pos[0]].isWhite == self.isWhite:
                            continue
                        else:
                            moves.append(pos)

                    moves.append(pos)

        blockedDir = [False,False,False,False]
        for i in range(1,8):
            temp = []
            temp.append((self.x+i,self.y+i))
            temp.append((self.x-i,self.y-i))
            temp.append((self.x-i,self.y+i))
            temp.append((self.x+i,self.y-i))

            for index, pos in enumerate(temp):
                if pos[0] in range(8) and pos[1] in range(8):
                    if blockedDir[index]:
                        continue
                    if board[pos[1]][pos[0]] is not None:
                        blockedDir[index] = True

                        if board[pos[1]][pos[0]].isWhite == self.isWhite:
                            continue
                        else:
                            moves.append(pos)

                    moves.append(pos)
        return moves

    def kingMoves(self,board):
        moves = []
        if not self.didMove:
            if board[self.y][self.x+1] is None and board[self.y][self.x+2] is None and board[self.y][self.x+3] is not None:
                pos = (self.x+2,self.y)
                moves.append(pos)
            if board[self.y][self.x-1] is None and board[self.y][self.x-2] is None and board[self.y][self.x-3] is None and board[self.y][self.x-4] is not None:
                pos = (self.x-2,self.y)
                moves.append(pos)


        for i in range(3):
            for j in range(3):
                pos = (self.x-1 + i,self.y-1+j)
                if pos[0] in range(8) and pos[1] in range(8):
                    if i == 1 and j == 1:
                        continue

                    if board[pos[1]][pos[0]] is not None:
                        if board[pos[1]][pos[0]].isWhite == self.isWhite:
                            continue
                        else:
                            moves.append(pos)
                    moves.append(pos)

        return moves







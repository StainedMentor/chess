from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsItem

from constants import *
from util import *




class Piece(QGraphicsItem):
    def __init__(self, type, x, y):
        super().__init__()
        self.image = QPixmap("pieces/Chess_"+type+"t60.png").scaledToWidth(DIMS["tile"])
        self.didMove = False
        self.type = type
        self.x = x
        self.y = y
        self.isWhite = type[-1]=="l"

    def boundingRect(self):
        return QRectF(self.image.rect())

    def paint(self, painter, option, widget):
        painter.drawPixmap(self.image.rect(), self.image)

    def getPos(self):
        return (self.x,self.y)

    def drawSelf(self, scene):
        scene.addItem(self)
        self.setPos(self.x * DIMS["tile"], self.y * DIMS["tile"])

    def move(self,x,y):
        self.didMove = True
        self.x = x
        self.y = y
        self.setPos(self.x * DIMS["tile"], self.y * DIMS["tile"])


    def select(self):
        self.setPos(self.x * DIMS["tile"]+10, self.y * DIMS["tile"]+10)

    def deselect(self):
        self.setPos(self.x * DIMS["tile"], self.y * DIMS["tile"])

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

        if isEmpty(pos1,board):
            moves.append(pos1)
            if not self.didMove:

                pos2 = (self.x,self.y+2*dir)
                if isEmpty(pos2,board):
                    moves.append(pos2)

        # diagonal attack
        pos3 = (self.x+1, self.y + 1 * dir)
        pos4 = (self.x-1, self.y + 1 * dir)
        if onBoard(pos3):
            if not isEmpty(pos3,board) and board[pos3[1]][pos3[0]].isWhite != self.isWhite:
                moves.append(pos3)
        if onBoard(pos4):
            if not isEmpty(pos4, board) and board[pos4[1]][pos4[0]].isWhite != self.isWhite:
                moves.append(pos4)

        return moves



    def rookMoves(self,board):
        return hvMoves(self.x,self.y,board,self.isWhite)

    def bishopMoves(self,board):
        return diagMoves(self.x,self.y,board,self.isWhite)

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
            if onBoard(pos):
                if not isEmpty(pos,board):
                    if board[pos[1]][pos[0]].isWhite == self.isWhite:
                        continue
                moves.append(pos)

        return moves


    def queenMove(self,board):
        moves = []
        hv = hvMoves(self.x,self.y,board,self.isWhite)
        diag = diagMoves(self.x,self.y,board,self.isWhite)
        moves.extend(diag)
        moves.extend(hv)

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
                if onBoard(pos):
                    if i == 1 and j == 1:
                        continue

                    if not isEmpty(pos,board):
                        if board[pos[1]][pos[0]].isWhite == self.isWhite:
                            continue
                    moves.append(pos)

        return moves


    def __deepcopy__(self, memodict={}):
        tmp = Piece(self.type, self.x,self.y)
        return tmp





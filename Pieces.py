from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QGraphicsItem

from constants import *
from util import *




class Piece(QGraphicsItem):
    def __init__(self, type, x, y, didMove=False):
        super().__init__()
        self.image = QPixmap("pieces/Chess_"+type+"t60.png").scaledToWidth(DIMS["tile"])
        self.didMove = didMove
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

    def getAvailableMoves(self, board, check, doCheck=True):
        temp = []
        match self.type[0]:
            case "p":
                temp = self.pawnMoves(board)
            case "r":
                temp = self.rookMoves(board)
            case "n":
                temp = self.knightMoves(board)
            case "b":
                temp = self.bishopMoves(board)
            case "k":
                temp = self.kingMoves(board)
            case "q":
                temp = self.queenMove(board)

        if not doCheck:
            return temp
        moves = []
        for move in temp:
            full = (self.getPos(),move)
            if not check(self.isWhite, full):
                moves.append(move)

        return moves

    def pawnMoves(self, board):
        moves = []
        direction = -1 if self.isWhite else 1
        forward_one = (self.x, self.y + direction)
        forward_two = (self.x, self.y + 2 * direction)
        diagonal_right = (self.x + 1, self.y + direction)
        diagonal_left = (self.x - 1, self.y + direction)

        if onBoard(forward_one) and isEmpty(forward_one, board):
            moves.append(forward_one)
            if not self.didMove and isEmpty(forward_two, board):
                moves.append(forward_two)

        # Check diagonal
        for pos in [diagonal_right, diagonal_left]:
            if onBoard(pos) and not isEmpty(pos, board) and board[pos[1]][pos[0]].isWhite != self.isWhite:
                moves.append(pos)

        return moves



    def rookMoves(self,board):
        return hvMoves(self.x,self.y,board,self.isWhite)

    def bishopMoves(self,board):
        return diagMoves(self.x,self.y,board,self.isWhite)

    def knightMoves(self, board):
        moves = []
        directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (-1, 2), (1, 2), (-1, -2), (1, -2)]

        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if onBoard((new_x, new_y)):
                if isEmpty((new_x, new_y), board) or board[new_y][new_x].isWhite != self.isWhite:
                    moves.append((new_x, new_y))

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
        tmp = Piece(self.type, self.x,self.y, self.didMove)
        return tmp




def diagMoves(x, y, board, isWhite):
    dirs = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
    moves = lineMoves(x,y,board,isWhite,dirs)
    return moves

def hvMoves(x,y, board, isWhite):
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    moves = lineMoves(x,y,board,isWhite,dirs)
    return moves


def lineMoves(x, y, board, isWhite, dirs):
    moves = []

    for dx, dy in dirs:
        for i in range(1, 8):
            new_x, new_y = x + i * dx, y + i * dy
            if not onBoard((new_x, new_y)):
                break
            if not isEmpty((new_x, new_y), board):
                if board[new_y][new_x].isWhite != isWhite:
                    moves.append((new_x, new_y))
                break
            moves.append((new_x, new_y))

    return moves



def onBoard(pos):
    return pos[0] in range(8) and pos[1] in range(8)

def isEmpty(pos, board):
    return board[pos[1]][pos[0]] is None
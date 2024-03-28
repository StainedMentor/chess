from copy import deepcopy

from constants import *
from Pieces import *

class Board:
    def __init__(self, scene):
        self.map = BOARD_DEF
        self.scene = scene

    def initPieces(self):
        for i in range(8):
            for j in range(8):
                if BOARD_DEF[j][i] is not None:
                    piece = Piece(BOARD_DEF[j][i],i,j)
                    piece.drawSelf(self.scene)

                    self.map[j][i] = piece

    def drag(self,selected, pos):
        self.map[selected[1]][selected[0]].setPos(pos.x(),pos.y())

    def isEmpty(self, pos):
        return self.map[pos[1]][pos[0]] is None

    def checkTurn(self, pos, whiteTurn):
        return self.map[pos[1]][pos[0]].isWhite == whiteTurn

    def move(self, origin, new):
        originx,originy = origin
        newx, newy = new

        if not self.isEmpty(new):
            self.map[newy][newx].delete(self.scene)

        if self.map[originy][originx].type[0] == "k" and abs(newx-originx) == 2:
            dir = newx - originx
            self.castle(dir, newy)

        self.map[newy][newx] = self.map[originy][originx]
        self.map[originy][originx] = None
        self.map[newy][newx].move(newx, newy)

        return originx,originy,newx,newy


    def castle(self, dir, y):
        if dir > 0:
            self.map[y][5] = self.map[y][7]
            self.map[y][7] = None
            self.map[y][5].move(5, y)

        else:
            self.map[y][3] = self.map[y][0]
            self.map[y][0] = None
            self.map[y][3].move(3, y)


    def checkCheck(self,map, isWhite):
        kingPos = self.getKingPos(map, isWhite)
        moves = getAllMoves(map, not isWhite)

        if kingPos in moves:
            return True
        else:
            return False

    def checkMate(self, isWhite):

        boards = self.getAllPossibleBoard( isWhite)

        for b in boards:
            if not self.checkCheck(b, isWhite):
                return False
        return True

    def getAllPossibleBoard(self, isWhite):
        boards = []


        for x in range(8):
            for y in range(8):
                if self.map[y][x] is None:
                    continue
                if self.map[y][x].isWhite == isWhite:
                    available = self.map[y][x].getAvailableMoves(self.map)
                    for move in available:
                        temp = deepcopy(self.map)
                        newx,newy = move
                        temp[newy][newx] = temp[y][x]
                        temp[y][x] = None
                        boards.append(temp)

        return boards

    def getAllMoves(self, map, isWhite):
        possibleMoves = []

        for x in range(8):
            for y in range(8):
                if map[y][x] is None:
                    continue
                if map[y][x].isWhite == isWhite:
                    available = map[y][x].getAvailableMoves(map)
                    possibleMoves.extend(available)

        return possibleMoves

    def getKingPos(self,map, isWhite):
        for x in range(8):
            for y in range(8):
                if map[y][x] is None:
                    continue

                if map[y][x].type[0] != 'k':
                    continue

                if map[y][x].isWhite == isWhite:
                    return (x,y)
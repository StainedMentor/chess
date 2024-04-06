

from constants import *

from copy import deepcopy, copy

# from Pieces import *

class Board:
    def __init__(self):
        self.map = deepcopy(BOARD_DEF)
        self.whiteChecked = False
        self.blackChecked = False
        self.selectedPiece = None

    def initPieces(self):
        for i in range(8):
            for j in range(8):
                if BOARD_DEF[j][i] is not None:
                    piece = Piece(BOARD_DEF[j][i],i,j)


                    self.map[j][i] = piece



    def isEmpty(self, pos):
        return self.map[pos[1]][pos[0]] is None

    def checkTurn(self, pos, whiteTurn):
        return self.map[pos[1]][pos[0]].isWhite == whiteTurn

    def move(self, origin, new):
        originx,originy = origin
        newx, newy = new


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


    def checkCheck(self,map, isWhite, doChecks=True):
        kingPos = self.getKingPos(map, isWhite)
        moves = self.getSidesAvailableMoves(map, not isWhite, doChecks)

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

    def checkMove(self, isWhite, move):

        temp = deepcopy(self.map)
        origin, new = move
        newx, newy = new
        x,y = origin
        temp[newy][newx] = temp[y][x]
        temp[y][x] = None

        return self.checkCheck(temp, isWhite, False)

    def getAllPossibleBoard(self, isWhite):
        boards = []


        for x in range(8):
            for y in range(8):
                if self.map[y][x] is None:
                    continue
                if self.map[y][x].isWhite == isWhite:
                    available = self.map[y][x].getAvailableMoves(self.map, self.checkMove)
                    for move in available:
                        temp = deepcopy(self.map)
                        newx,newy = move
                        temp[newy][newx] = temp[y][x]
                        temp[y][x] = None
                        boards.append(temp)

        return boards

    def getSidesAvailableMoves(self, map, isWhite, doCheck=True):
        possibleMoves = []

        for x in range(8):
            for y in range(8):
                if map[y][x] is None:
                    continue
                if map[y][x].isWhite == isWhite:
                    available = map[y][x].getAvailableMoves(map, self.checkMove, doCheck)
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




    # neural net functions

    def encodeBoard(self):
        encoded = [[0 for _ in range(8)] for _ in range(8)]
        piece_to_index = {'pd': 1, 'nd': 2, 'bd': 3, 'rd': 4, 'qd': 5, 'kd': 6,
                          'pl': -1, 'nl': -2, 'bl': -3, 'rl': -4, 'ql': -5, 'kl': -6}

        for x in range(8):
            for y in range(8):
                if self.map[y][x] is None:
                    continue
                t = self.map[y][x].type
                encoded[y][x] = piece_to_index[t]

        return encoded


    def getAllMovesEncoded(self, isWhite):
        possibleMoves = []

        for x in range(8):
            for y in range(8):
                if self.map[y][x] is None:
                    continue
                if self.map[y][x].isWhite == isWhite:
                    available = self.map[y][x].getAvailableMoves(self.map, self.checkMove)
                    for a in available:
                        possibleMoves.append(((x,y),a))
        return possibleMoves


class Piece:
    def __init__(self, type, x, y, didMove=False):
        self.didMove = didMove
        self.type = type
        self.x = x
        self.y = y
        self.isWhite = type[-1]=="l"



    def getPos(self):
        return (self.x,self.y)



    def move(self,x,y):
        self.didMove = True
        self.x = x
        self.y = y



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
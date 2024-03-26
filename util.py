

def getAllMoves(board, fromWhite = True):
    possibleMoves = []

    for x in range(8):
        for y in range(8):
            if board[y][x] is None:
                continue
            if board[y][x].isWhite == fromWhite:
                available = board[y][x].getAvailableMoves(board)
                possibleMoves.extend(available)

    return possibleMoves


def hvMoves(x,y, board, isWhite):
    moves = []
    blockedDir = [False, False, False, False]
    for i in range(1, 8):
        temp = []
        temp.append((x + i, y))
        temp.append((x - i, y))
        temp.append((x, y + i))
        temp.append((x, y - i))

        for index, pos in enumerate(temp):
            if onBoard(pos):
                if blockedDir[index]:
                    continue
                if not isEmpty(pos,board):
                    blockedDir[index] = True

                    if board[pos[1]][pos[0]].isWhite == isWhite:
                        continue


                moves.append(pos)

    return moves

def diagMoves(x,y, board, isWhite):

    moves = []
    blockedDir = [False, False, False, False]
    for i in range(1, 8):
        temp = []
        temp.append((x + i, y + i))
        temp.append((x - i, y - i))
        temp.append((x - i, y + i))
        temp.append((x + i, y - i))

        for index, pos in enumerate(temp):
            if onBoard(pos):
                if blockedDir[index]:
                    continue
                if not isEmpty(pos,board):
                    blockedDir[index] = True

                    if board[pos[1]][pos[0]].isWhite == isWhite:
                        continue

                moves.append(pos)

    return moves



def onBoard(pos):
    return pos[0] in range(8) and pos[1] in range(8)

def isEmpty(pos, board):
    return board[pos[1]][pos[0]] is None


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

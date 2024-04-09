
class MinMax:
    def __init__(self, state, isWhite):
        self.depth = 3
        self.cost = 0
        self.isWhite = isWhite
        self.state = state
        self.tempState = state
        self.run()

    def pieceValue(self, piece):
        match piece:
            case "p":
                return 1
            case "r":
                return 3
            case "n":
                return 3
            case "b":
                return 3
            case "k":
                return 99
            case "q":
                return 10

    def getAllMoves(self, state, isWhite):
        possibleMoves = []

        for x in range(8):
            for y in range(8):
                if self.state[y][x] is None:
                    continue
                if not self.state[y][x].isWhite:
                    available = self.state[y][x].getAvailableMoves(self.state)
                    for a in available:
                        possibleMoves.append(((x,y),a))
        return possibleMoves



    def getCostOfMove(self, state, move, isWhite):
        dir = -1 if isWhite else 1

        start,stop = move
        if state[stop[1]][stop[0]] is not None:
            val = self.pieceValue(state[stop[1]][stop[0]].type[0])
            return val*dir

        return 0

    def run(self):
        self.moves = self.getAllMoves(self.state,self.isWhite)
        self.cost = []
        for move in self.moves:
            c = self.getCostOfMove(self.state,move,self.isWhite)
            self.cost.append(c)


        return sum(self.cost)



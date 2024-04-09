import random
import numpy as np

from ai.boardNoGui import Board

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt




class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        self.fc1 = nn.Linear(64+4,1088)
        self.fc2 = nn.Linear(1088,1088)
        self.fc3 = nn.Linear(1088,1088)
        self.fc4 = nn.Linear(1088,1088)

        self.fc5 = nn.Linear(1088,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.fc5(x)
        return x


def getBestMove(moves, bstate, net):
    vals = torch.Tensor()
    for move in moves:
        o, n = move
        inVec = np.concatenate((bstate, o, n))
        inVec = torch.Tensor(inVec)
        output = net(inVec)
        vals = torch.cat((vals, output))
    vals = vals.detach().numpy()
    bestIndex = vals.argmax()
    return moves[bestIndex]

def updateRewards(buffer, num, modifier, fade,offset=0):
    if num == -1:
        num = int(len(buffer)/2)

    for i in range(offset,num*2,2):
        if i >= len(buffer): break
        state, action, reward = buffer[-1 - i]
        newReward = reward + (modifier/(fade*i+1))

        buffer[-1 - i] = [state, action, newReward]

def decodeBoard(boardState):
    temp = np.array(boardState)
    temp = temp.reshape(8,8)
    return temp

def train():
    # init
    net = ChessNet()
    net.load_state_dict(torch.load('ai/dict.pth'))

    num_episodes = 1000
    maxSteps = 120
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    memoryBuffer = []
    stats = None

    gameSamples = []
    lossArr = []

    # Training loop
    for episode in range(num_episodes):
        whiteTook, blackTook, whiteWon, blackWon = 0,0,0,0
        newBoard = Board()
        newBoard.initPieces()

        episodeBuffer = []
        episodeEnded = False
        for step in range(maxSteps):

            turn = (step+1)%2
            bstate = newBoard.encodeBoard()
            bstate = np.array(bstate)
            bstate = bstate.flatten()
            moves = newBoard.getAllMovesEncoded(True if turn == 1 else False)


            o, n = getBestMove(moves,bstate,net)

            newBoard.move(o, n)
            check = newBoard.checkCheck(newBoard.map, not (True if turn == 1 else False))
            mate = newBoard.checkMate(not (True if turn == 1 else False))

            reward = 0
            if check:
                reward = 5

            if mate:
                episodeEnded = True


            newState = newBoard.encodeBoard()
            newState = np.array(newState)
            newState = newState.flatten()

            p1 = np.count_nonzero(bstate)
            p2 = np.count_nonzero(newState)
            if p2 < p1:
                # reward += 10
                # updateRewards(episodeBuffer,3,10,0.1,1)
                # updateRewards(episodeBuffer,3,-10,0.1)

                if turn == 1:
                    whiteTook += 1
                else:
                    blackTook += 1

            episodeBuffer.append([bstate, (o, n), reward])

            if episodeEnded:
                if turn == 1:
                    whiteWon += 1
                else:
                    blackWon += 1
                updateRewards(episodeBuffer,-1,100,0.1)
                updateRewards(episodeBuffer,-1,-100,0.1,1)

                break
        current = np.array([whiteTook,blackTook,whiteWon,blackWon])
        if stats is None:
            stats = np.array([current])
        else:
            stats = np.append(stats,[current], axis=0)
        print("\r"+"episode: " + str(episode)+ " white: " + str(np.sum(stats[:,2]))+ " black: " + str(np.sum(stats[:,3])) + " finished " + str((np.sum(stats[:,2]) +np.sum(stats[:,3]))/(episode+1)), end="")

        if not episodeEnded:
            updateRewards(episodeBuffer,-1,-100,0)
            updateRewards(episodeBuffer,-1,-100,0,1)

        memoryBuffer.extend(episodeBuffer)


        # Network training
        batchSize = min(len(memoryBuffer), 64)
        miniBatch = random.sample(memoryBuffer, batchSize)
        batchSizeEpisode = min(len(episodeBuffer), 64)
        miniBatchEpisode = random.sample(episodeBuffer, batchSizeEpisode)


        miniBatch.extend(miniBatchEpisode)

        tempLoss = 0
        for state, action, reward in miniBatch:
            inVec = np.concatenate((state, *action))
            inVec = torch.Tensor(inVec)
            predicted_values = net(inVec)

            target = [reward]
            target = torch.Tensor(target)
            loss = F.mse_loss(predicted_values, target)
            tempLoss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lossArr.append(loss/len(miniBatch))


        if episode % 10 ==0:
            gameSamples.append(episodeBuffer)


    # for sample in gameSamples:
    #     for turn in sample:
    #         print(decodeBoard(turn[0]))


    stDict = net.state_dict()
    torch.save(stDict, 'ai/dict.pth')

    stats = np.array(stats)

    print("")
    print(np.sum(stats[:,2]),np.sum(stats[:,3]))

    plt.plot(np.arange(0,num_episodes), stats[:,0], marker='o', linestyle='-')
    plt.plot(np.arange(0,num_episodes), stats[:,1], marker='o', linestyle='-')

    plt.xlabel('Iteration')
    plt.ylabel('Pieces taken')

    plt.grid(True)
    plt.show()

    lossArr = torch.Tensor(lossArr)
    lossArr = lossArr.detach().numpy()

    plt.plot(np.arange(0,num_episodes), lossArr, marker='o', linestyle='-')

    plt.xlabel('Iteration')
    plt.ylabel('Pieces taken')

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train()
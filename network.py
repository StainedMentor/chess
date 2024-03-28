import random
import numpy as np

from boardNoGui import Board

import torch
import torch.nn as nn
import torch.nn.functional as F



class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        self.fc1 = nn.Linear(64+4,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# net = ChessNet()
def train():
    # init
    net = ChessNet()

    net.load_state_dict(torch.load('dict.pth'))
    net.eval()

    num_episodes = 1000
    max_steps_per_episode = 200
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # Training loop
    memoryBuffer = []
    for episode in range(num_episodes):
        print("episode: " + str(episode))
        newBoard = Board()
        newBoard.initPieces()

        episodeBuffer = []

        for step in range(max_steps_per_episode):

            turn = step%2
            bstate = newBoard.encodeBoard()
            bstate = np.array(bstate)

            bstate = bstate.flatten()
            moves = newBoard.getAllMovesEncoded(True if turn == 1 else False)

            vals = torch.Tensor()
            for move in moves:
                o, n = move
                inVec = np.concatenate((bstate, o,n))
                inVec = torch.Tensor(inVec)
                output = net(inVec)
                vals = torch.cat((vals,output))

            vals = vals.detach().numpy()
            bestIndex = vals.argmax()

            o, n = moves[bestIndex]
            newBoard.move(o, n)
            check = newBoard.checkCheck(newBoard.map, not (True if turn == 1 else False))
            mate = newBoard.checkMate(not (True if turn == 1 else False))

            reward = 0
            episodeEnded = False
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
                reward += 5

            episodeBuffer.append((bstate, (o, n), reward, newState))

            if episodeEnded:
                final_reward = 100
                episodeBuffer.append((bstate, (o, n), final_reward, None))
                num = range(len(episodeBuffer))
                alt = 1
                for i in num:
                    state, action, reward, next_state = episodeBuffer[-1 - i]
                    modified_reward = reward + 10/(i+1)*alt
                    alt*(-1)

                    episodeBuffer[-1 - i] = (state, action, modified_reward, next_state)
                break


        memoryBuffer.extend(episodeBuffer)
        batchSize = min(len(memoryBuffer), 32)
        miniBatch = random.sample(memoryBuffer, batchSize)

        for state, action, reward, next_state in miniBatch:
            inVec = np.concatenate((state, *action))
            inVec = torch.Tensor(inVec)
            predicted_values = net(inVec)

            target = [reward]
            target = torch.Tensor(target)
            loss = F.mse_loss(predicted_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    stDict = net.state_dict()
    torch.save(stDict, 'dict.pth')





if __name__ == "__main__":
    train()
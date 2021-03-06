'''
Created on Oct 28, 2017

@author: cclausen
'''


from core.AbstractState import AbstractState

from core.FieldAugments import initField, augmentFieldAndMovesDistribution


import math

class MNK():
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        
        assert m >= k or n >= k
        
        # 0 is an empty field, 1 is the first player, 2 is the second player
        self.board = initField(m,n,-1)
        self.turn = 0
        
        self.winningPlayer = -1
    
    def clone(self):
        c = MNK(self.m, self.n, self.k)
        for y in range(self.n):
            for x in range(self.m):
                c.board[y][x] = self.board[y][x]
        c.turn = self.turn
        c.winningPlayer = self.winningPlayer
        return c
    
    def hasEnded(self):
        return self.winningPlayer != -1 or self.turn >= self.m * self.n
    
    def _searchWinner(self, lx, ly):
        """
        return -1 if the game is not over
        else the number of the winning player
        """
        if self.winningPlayer != -1:
            return
        
        p = self.board[ly][lx]
        if p != -1:
            dirs = [(1,0),(0,1),(1,-1),(1,1)]
            for direction in dirs:
                l = 0
                for d in [1,-1]:
                    x = lx
                    y = ly
                    while x > -1 and y > -1 and x < self.m and y < self.n and self.board[y][x] == p:
                        l += 1
                        x += d * direction[0]
                        y += d * direction[1]
                        
                        if l - 1 >= self.k:
                            self.winningPlayer = p
                            return
     
    def place(self, x, y):
        self.board[y][x] = (self.turn % 2)
        self.turn += 1
        self._searchWinner(x, y)
    
    def __str__(self):
        mm = ['-', 'X', 'O']
        s = "MNK(%i,%i,%i), " %  (self.m, self.n, self.k)
        if not self.hasEnded():
            s += "On turn: %s\n" % mm[(self.turn % 2)+1]
        elif self.winningPlayer > -1:
            s += "Winner: %s\n" % mm[self.winningPlayer+1]
        else:
            s += "Draw\n"
        for y in range(self.n):
            for x in range(self.m):
                s += mm[self.board[y][x]+1]
            s += "\n"
        return s
    
class MNKState(AbstractState):
    def __init__(self, mnk):
        super(MNKState, self).__init__()
        self.mnk = mnk
        
        
    def canTeachSomething(self):
        return True

    def getMoveLocation(self, key):
        y = math.floor(key / self.mnk.m)
        x = int(key % self.mnk.m)
        return x, y

    def getWinner(self):
        return self.mnk.winningPlayer

    def getMoveKey(self, x, y):
        return y * self.mnk.m + x

    def isMoveLegal(self, move):
        x, y = self.getMoveLocation(move)
        return self.mnk.board[y][x] == -1

    def describeMove(self, move):
        x, y = self.getMoveLocation(move)
        return str(x) + "-" + str(y)

    def augmentFrame(self, frame):
        frame = [frame[0].clone(), list(frame[1]), frame[2], list(frame[3])]
        
        fState = frame[0].mnk
        fMoves = frame[1]

        augmentFieldAndMovesDistribution(fState.m, fState.n, fState.board, fMoves, 
                                         lambda idx: self.getMoveLocation(idx),
                                         lambda x,y: self.getMoveKey(x, y))

        return frame

    def getPlayerOnTurnIndex(self):
        return self.mnk.turn % 2
    
    def getTurn(self):
        return self.mnk.turn
    
    def isEarlyGame(self):
        # TODO: hmmmm how does this affect things actually?!
        c = max(self.mnk.m, self.mnk.n)
        c += c % 2
        return self.mnk.turn < c#True #self.mnk.turn < self.mnk.m * self.mnk.n * 0.75
    
    def getNewGame(self):
        return MNKState(MNK(self.mnk.m, self.mnk.n, self.mnk.k))
    
    def getPlayerCount(self):
        return 2
    
    def getMoveCount(self):
        return self.mnk.m * self.mnk.n
    
    def isEqual(self, other):
        assert self.mnk.m == other.mnk.m and self.mnk.n == other.mnk.n and self.mnk.k == other.mnk.k, "Why are these states with different rules even compared?"
        if self.mnk.turn != other.mnk.turn:
            return False
        for idx, myLine in enumerate(self.mnk.board):
            oLine = other.mnk.board[idx]
            for idx in range(self.mnk.m):
                if oLine[idx] != myLine[idx]:
                    return False
        return True
    
    def clone(self):
        return MNKState(self.mnk.clone())
        
    def getFrameClone(self):
        return self.clone()
        
    def simulate(self, move):
        super(MNKState, self).simulate(move)
        x, y = self.getMoveLocation(move)
        self.mnk.place(x, y)
    
    def isTerminal(self):
        return self.mnk.hasEnded()
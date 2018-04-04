'''
Created on Apr 3, 2018

@author: cclausen
'''

from mnk.MNKGame import MNK, MNKState

def stateFormat(state):
    return str(state.mnk)

class MNKInit():
    
    def __init__(self):
        self.stateFormat = stateFormat
    
    def setConfig(self, config):
        self.config = config
        gconf = self.config["game"]
        self.m = gconf["m"]
        self.n = gconf["n"]
        self.k = gconf["k"]
    
    def getStateTemplate(self):
        return MNKState(MNK(self.m,self.n,self.k))
    
    def getPlayerCount(self):
        return 2
    
    def getMoveCount(self):
        return self.m * self.n
    
    def getGameDimensions(self):
        return [self.m, self.n, 1]
    
    def fillNetworkInput(self, state, tensor, batchIndex):
        for x in range(self.m):
            for y in range(self.n):
                b = state.mnk.board[y][x]
                if b != -1:
                    b = state.mapPlayerIndexToTurnRel(b)
                tensor[batchIndex,0,x,y] = b
                
    
    def mkParseCommand(self):
        m = self.m
        n = self.n
        k = self.k
        def p(cmd):
            try:
                ms = cmd.split("-")
                x = int(ms[0]) - 1
                y = int(ms[1]) - 1
                return MNKState(MNK(m,n,k)).getMoveKey(x,y)
            except:
                return -1
        return p
    
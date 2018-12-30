'''
Created on Apr 4, 2018

@author: cclausen
'''

from connect6.Connect6Game import Connect6State, Connect6

def stateFormat(state):
    return str(state.c6)

class Connect6Init():
    def __init__(self):
        assert False, "assert: cython version is used!"
        self.stateFormat = stateFormat

    def setConfig(self, config):
        self.config = config
        gconf = self.config["game"]
        self.m = gconf["m"]
        self.n = gconf["n"]
        
    def getStateTemplate(self):
        return Connect6State(Connect6(m=self.m, n=self.n))

    def getPlayerCount(self):
        return 2
    
    def getMoveCount(self):
        return self.m * self.n
    
    def getGameDimensions(self):
        return [self.m, self.n, 1]
    
    def fillNetworkInput(self, state, tensor, batchIndex):
        assert False, "assert: cython version is used!"
        for y in range(self.n):
            bline = state.c6.board[y]
            for x in range(self.m):
                b = bline[x]
                if b != -1:
                    b = state.mapPlayerIndexToTurnRel(b)
                tensor[batchIndex,0,y,x] = b

    def mkParseCommand(self):
        m = self.m
        n = self.n
        def p(cmd):
            try:
                ms = cmd.split("-")
                chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
                cIdx = chars.find(ms[1]);
                if cIdx != -1:
                    ms[1] = cIdx + 1
                x = int(ms[0]) - 1
                y = int(ms[1]) - 1
                return Connect6State(Connect6(m=m,n=n)).getMoveKey(x,y)
            except:
                return -1
        return p
        
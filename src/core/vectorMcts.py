'''
Created on Dec 31, 2018

@author: cclausen
'''

import numpy as np

def randargmax(b):
    # https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking
    return np.random.choice(np.flatnonzero(b == b.max()))

class TreeNode():
    def __init__(self, state, parentNode = None, parentMove = 0, noiseMix = 0.1):
        self.state = state
        mc = self.state.getMoveCount()
    
        self.noiseMix = noiseMix
        
        self.isExpanded = False
        
        self.parentMove = parentMove
        self.parentNode = parentNode
        
        self.children = {}
        
        self.dconst = np.asarray([0.03] * mc, dtype="float32")
        
        self.numMoves = mc
        self.edgePriors = np.zeros(mc, dtype=np.float32)
        self.edgeVisits = np.zeros(mc, dtype=np.float32)
        self.edgeTotalValues = np.zeros(mc, dtype=np.float32)
        self.edgeMeanValues = np.zeros(mc, dtype=np.float32)
        
        self.edgeLegal = np.zeros(mc, dtype=np.float32)
        for m in range(self.numMoves):
            if self.state.isMoveLegal(m):
                self.edgeLegal[m] = 1
        
        self.terminalResult = None
        
        self.stateValue = 0.5
        
        self.allVisits = 0
        
    def executeMove(self, move):
        newState = self.state.clone()
        newState.simulate(move)
        return TreeNode(newState, self, parentMove = move, noiseMix = self.noiseMix)
    
    def getBestValue(self):
        """
        returns a single float that is meant to tell what the best 
        possible expected outcome is by choosing the best possible actions
        """
        return np.max(self.edgeMeanValues)
    
    def cutTree(self):
        """
        deletes all children, reducing the tree to the root
        resets all counters
        meant to be used when different solvers are used in an alternating fashion on the same tree.
        maybe instead a completely different tree should be used for each solver. But meh.
        Training does reuse the trees, test play doesn't. Better than nothing...
        """
        
        self.children = {}
        self.isExpanded = False
        self.parentMove = 0
        self.parentNode = None
        self.terminalResult = None
        
        self.edgePriors.fill_(0)
        self.edgeVisits.fill_(0)
        self.edgeTotalValues.fill_(0)
        self.edgeMeanValues.fill_(0)
        self.stateValue = 0.5
        self.allVisits = 0
       
    def getVisitsFactor(self):
        # .0001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0
        # but there is little effect in other cases
        return self.allVisits ** 0.5 + 0.0001

    def selectDown(self, cpuct):
        node = self
        while node.isExpanded and not node.state.isTerminal():
            node = node.selectMove(cpuct)
        return node
        
    def getChildForMove(self, move):
        assert self.isExpanded
        
        child = None
        
        if not move in self.children:
            child = self.executeMove(move)
            self.children[move] = child
        else:
            child = self.children[move]
            
        child.parentNode = None
        return child
    
    def getMoveDistribution(self):
        return self.edgeVisits / float(self.allVisits)
    
    def pickMove(self, cpuct):
        useNoise = self.allVisits < 5
        
        p = self.edgePriors
        
        if useNoise:
            dirNoise = np.random.dirichlet(self.dconst)
            p = (1 - self.noiseMix) * p + self.noiseMix * dirNoise
        
        nodeQs = self.edgeMeanValues #this will assume a zero value for unvisited moves. Bad for speed (apparently), bad for the search? Annoying to fix in this kind of setup. cvectorMcts does it better.
        nodeUs = cpuct * self.edgePriors * (self.getVisitsFactor() / (1.0 + self.edgeVisits))
        
        values = (nodeQs + nodeUs) * self.edgeLegal
        
        result = randargmax(values)
        
        assert self.edgeLegal[result] == 1
        
        return result
    
    def selectMove(self, cpuct):
        move = self.pickMove(cpuct)
        
        if not move in self.children:
            self.children[move] = self.executeMove(move)
            
        return self.children[move]
    
    def backup(self, vs):
        pNode = self.parentNode
        if pNode != None:
            pMove = self.parentMove
            pNode.edgeVisits[pMove] += 1
            pNode.allVisits += 1
            pNode.edgeTotalValues[pMove] += vs[pNode.state.getPlayerOnTurnIndex()]
            pNode.edgeMeanValues[pMove] = float(pNode.edgeTotalValues[pMove]) / pNode.edgeVisits[pMove]
            pNode.backup(vs)
    
    def getTerminalResult(self):
        if self.terminalResult is None:
            r = [0] * self.state.getPlayerCount()
            winner = self.state.getWinner()
            if winner != -1:
                r[winner] = 1
            else:
                r = [1.0 / self.state.getPlayerCount()] * self.state.getPlayerCount()
            self.terminalResult = r
        return self.terminalResult
    
    def expand(self, movePMap, vs):
        np.copyto(self.edgePriors, movePMap, casting="no")
        self.isExpanded = True
        self.stateValue = vs[self.state.getPlayerOnTurnIndex()]
    
def batchedMcts(states, expansions, evaluator, cpuct):
    workspace = states
    for _ in range(expansions):
        workspace = [s.selectDown(cpuct) for s in workspace]
        
        evalout = evaluator(workspace)
        for idx, ev in enumerate(evalout):
            node = workspace[idx]
            
            w = ev[1]
            if node.state.isTerminal():
                w = node.getTerminalResult()
            else:
                node.expand(ev[0], ev[1])
            node.backup(w)
            workspace[idx] = states[idx]

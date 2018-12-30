'''
Created on Oct 27, 2017

@author: cclausen
'''

import numpy as np

import random

# internally used data holder
# represents information about previous tree traversal between parent and child treenode 
# cython: use a struct
class TreeEdge():
    def __init__(self, priorP, parentNode):
        self.visitCount = 0
        self.totalValue = 0
        self.meanValue = 0.5 # TODO have a look at modeling this as a distribution instead of a mean, see arXiv 1707.06887 as detailed inspiration. How to apply that to MCTS?
        self.priorP = priorP
        self.parentNode = parentNode
        self.childNode = None

        
# cython: turn into an extension type shell that adapts the raw c tree search
class TreeNode():
    def __init__(self, state, parentEdge=None, noiseMix = 0.1): 
        mc = state.getMoveCount()

        # internal mapping of move keys -> TreeNode associated with taking that move,
        # or None if this move was never tried before 
        
        self.edges = [None] * mc
        
        # internal TreeEdge that points up the tree, used for backup()
        self.parent = parentEdge
        
        # internal used value for numpy dirichlet thingy
        self.dconst = [0.03] * mc
        
        # internal used integer
        self.allVisits = 0

        # internal boolean flag for pickmove optimization
        self.hasHighs = False
        # internal list of move keys for pickmove optimization
        self.highs = None
        
        # internally used floats for pickmove optimization
        self.lowS = 0
        self.lowQ = 0

        # internal move key -> move prob mapping, i.e. an array
        self.movePMap = None
        
        # internal use
        self.stateValue = 0.5

        # externally provided float, usage only internal
        self.noiseMix = noiseMix

        # python array of floats that represent the result in case this node is a terminal state
        # lazy initialized
        # external access via getter that wants a list
        self.terminalResult = None

        # external use: an abstract state
        self.state = state
        
        # externally used boolean flag 
        self.isExpanded = False
        
        
        
    # PRIVATE APIS
    
    def executeMove(self, move):
        assert self.edges[move] != None
        newState = self.state.clone()
        newState.simulate(move)
        return TreeNode(newState, parentEdge=self.edges[move], noiseMix = self.noiseMix)
    
    def getVisitsFactor(self):
        # .0001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0
        # but there is little effect in other cases
        return self.allVisits ** 0.5 + 0.0001
    
    def groupCurrentMoves(self, cpuct):
        lMoves = self.state.getLegalMoves()
        numLegalMoves = len(lMoves)
        
        moves = []
        
        dirNoise = np.random.dirichlet(self.dconst[:numLegalMoves])
        
        # collect for each move:
        # id, q, s
        # q and s are the parts of the forumla that determine how future backups will cause the 
        # probability of a move to change
        for idx in range(numLegalMoves):
            move = lMoves[idx]
            
            iNoise = dirNoise[idx]
            
            e = self.edges[move]
            
            if e != None:
                q = e.meanValue
                p = e.priorP
                vc = e.visitCount
            else:
                q = self.stateValue
                p = self.movePMap[move]
                vc = 0.0
                
            p = (1-self.noiseMix) * p + self.noiseMix * iNoise
            
            s = cpuct * p / (1.0+vc)
            
            moves.append((move, q, s))
        
        f = self.getVisitsFactor()
        
        moves.sort(key=lambda x: x[1] + x[2] * f, reverse=True)
        
        # tuning this number can increase or decrease performance
        lowFactor = 0.89133742
        
        highLen = len(moves) - int(len(moves) * lowFactor)
        
        minHighs = 5
        if highLen < minHighs:
            highLen = minHighs
        
        self.highs = list(map(lambda x: x[0], moves[:highLen]))
        
        lows = moves[highLen:]
        
        if len(lows) > 0:
            self.lowQ = max(map(lambda x: x[1], lows))
            self.lowS = max(map(lambda x: x[2], lows))
        else:
            self.lowQ = 0
            self.lowS = 0
        
        self.hasHighs = True

    def pickMoveFromMoveKeys(self, moveKeys, cpuct):
        allVisitsSq = self.getVisitsFactor()
        
        numKeys = len(moveKeys)
        assert numKeys > 0
        useNoise = self.allVisits < 5 
        if useNoise:
            dirNoise = np.random.dirichlet(self.dconst[:numKeys])
        startIdx = random.randint(0, numKeys-1)
        
        moveName = None
        moveValue = 0
        
        for biasedIdx in range(numKeys):
            idx = (biasedIdx + startIdx) % numKeys
            
            if useNoise:
                iNoise = dirNoise[idx]
            
            idx = moveKeys[idx]
            
            e = self.edges[idx]
            
            if e != None:
                q = e.meanValue
                p = e.priorP
                vc = e.visitCount
            else:
                q = self.stateValue
                p = self.movePMap[idx]
                vc = 0
                
            if useNoise:
                p = (1-self.noiseMix) * p + self.noiseMix * iNoise
            u = cpuct * p * (allVisitsSq / (1.0 + vc))
            
            value = q + u
            if (moveName == None or value > moveValue) and self.state.isMoveLegal(idx):
                moveName = idx
                moveValue = value
                
        return moveName, moveValue
    
    # END OF PRIVATE APIS
    
    def getBestValue(self):
        """
    returns a single float that is meant to tell what the best 
    possible expected outcome is by chosing the best possible actions
        """
        bv = 0
        for e in self.edges:
            if e != None and e.meanValue > bv:
                bv = e.meanValue
        return bv
        
    def cutTree(self):
        """
        deletes all children, reducing the tree to the root
        resets all counters
        meant to be used when different solvers are used in an alternating fashion on the same tree.
        maybe instead a completely different tree should be used for each solver. But meh.
        Training does reuse the trees, test play doesn't. Better than nothing...
        """
        self.edges = [None] * self.state.getMoveCount()
        self.parent = None
        self.terminalResult = None
        self.movePMap = None
        self.isExpanded = False
        self.allVisits = 0
        
        self.hasHighs = False
        self.highs = None
        self.lowS = 0
        self.lowQ = 0

    
    def getChildForMove(self, move):
        assert self.isExpanded
        
        if self.edges[move] == None:
            self.edges[move] = TreeEdge(self.movePMap[move], self)
        
        child = self.edges[move].childNode
        
        if child == None:
            self.edges[move].childNode = self.executeMove(move)
            child = self.edges[move].childNode 
        
        child.parent = None
        return child
    
    def getMoveDistribution(self):
        sumv = float(self.allVisits)
        
        r = [0] * len(self.edges)
        for m in range(len(r)):
            e = self.edges[m]
            if e != None:
                r[m] = e.visitCount / sumv
        
        return r
    
    def selectMove(self, cpuct):
        if not self.hasHighs:
            self.groupCurrentMoves(cpuct)
              
        moveName, fastMoveValue = self.pickMoveFromMoveKeys(self.highs, cpuct)
        lowersBestValue = self.lowQ + self.lowS * self.getVisitsFactor()
         
        if lowersBestValue >= fastMoveValue:
            self.groupCurrentMoves(cpuct)
            moveName, _ = self.pickMoveFromMoveKeys(self.highs, cpuct)
        
#         moveName, _ = self.pickMoveFromMoveKeys(self.state.getLegalMoves(), cpuct)
        
        # this is the slow code replaced by the high-low split of groupCurrentMoves
        # !!!!!! to verify via this assertion remove the randomness from pickMoveFromMoveKeys!
#         moveNameSlow, _ = self.pickMoveFromMoveKeys(self.state.getLegalMoves(), cpuct)
#         assert moveName == moveNameSlow
        
        if self.edges[moveName] == None:
            self.edges[moveName] = TreeEdge(self.movePMap[moveName], self)

        selectedEdge = self.edges[moveName]
        if selectedEdge.childNode == None:
            selectedEdge.childNode = self.executeMove(moveName)
        
        return selectedEdge.childNode

    def backup(self, vs):
        if self.parent != None:
            self.parent.visitCount += 1
            self.parent.parentNode.allVisits += 1
            self.parent.totalValue += vs[self.parent.parentNode.state.getPlayerOnTurnIndex()]
            self.parent.meanValue = float(self.parent.totalValue) / self.parent.visitCount
            self.parent.parentNode.backup(vs)

    def getTerminalResult(self):
        # assert self.state.isTerminal()
        if self.terminalResult == None:
            r = [0] * self.state.getPlayerCount()
            winner = self.state.getWinner()
            if winner != -1:
                r[winner] = 1
            else:
                r = [1.0 / self.state.getPlayerCount()] * self.state.getPlayerCount()
            self.terminalResult = r
        return self.terminalResult

    # movePMap is a torch tensor that encodes move probabilities from the network
    # in a 1d array fashion, move key -> probability
    # it is our own copy, we don't need to copy it again
    def expand(self, movePMap, vs):
        self.movePMap = movePMap
        self.isExpanded = True
        self.stateValue = vs[self.state.getPlayerOnTurnIndex()]

def _selectDown(cpuct, node):
    while node.isExpanded and not node.state.isTerminal():
        node = node.selectMove(cpuct)
    return node

def batchedMcts(states, expansions, evaluator, cpuct):
    workspace = states
    for _ in range(expansions):
        workspace = [_selectDown(cpuct, s) if s != None else None for s in workspace]

        evalout = evaluator(workspace)
        for idx, ev in enumerate(evalout):
            node = workspace[idx]
            if node is None:
                continue
            
            w = ev[1]
            if node.state.isTerminal():
                w = node.getTerminalResult()
            else:
                node.expand(ev[0], ev[1])
            node.backup(w)
            workspace[idx] = states[idx]

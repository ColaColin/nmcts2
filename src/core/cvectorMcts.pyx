# cython: profile=False
'''
Created on Dec 31, 2018

@author: cclausen
'''
# arbitrary performance scale
# Base cython version: 1.27
# randargmax         : 1.57
# cdef TreeNode 1    : 1.69
# by hand pickMove   : 2.46
# q value init       : 4.37 <- different (better?!) search behavior, but the same as in cMctsTree, this was an oversight that made this one behave different (slower & worse)
# dconst cache       : 4.47

import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

from libc.stdlib cimport rand, RAND_MAX

# negative values should be not possible for moves in general?!
cdef float illegalMoveValue = -1

cdef int bestLegalValue(float [:] ar):
    cdef int n = ar.shape[0]
    
    if n == 0:
        return -1
    
    cdef int startIdx = int((rand()/(<float>RAND_MAX)) * n)
    
    cdef int idx, bestIdx, i
    cdef float bestValue
    
    bestIdx = -1
    bestValue = -1
    
    for i in range(n):
        idx = (i + startIdx) % n
        
        if ar[idx] > bestValue and ar[idx] != illegalMoveValue:
            bestValue = ar[idx]
            bestIdx = idx
    
    return bestIdx

cdef object dconsts = {}

cdef object getDconst(int n):
    global dconsts
    if not n in dconsts:
        dconsts[n] = np.asarray([0.03] * n, dtype=np.float32)
    return dconsts[n]

cdef class TreeNode():
    
    cdef readonly object state
    
    cdef float noiseMix
    cdef int isExpanded
    
    cdef object dconst
    
    cdef int parentMove
    cdef TreeNode parentNode

    cdef object children
    
    cdef int numMoves
    cdef float [:] edgePriors
    cdef float [:] edgeVisits
    cdef float [:] edgeTotalValues
    cdef float [:] edgeMeanValues
    cdef signed char [:] edgeLegal
    cdef float [:] valueTmp

    cdef float[:] noiseCache

    cdef object terminalResult

    cdef float stateValue
    cdef int allVisits
    
    cdef object netValueEvaluation

    def __init__(self, state, parentNode = None, parentMove = 0, noiseMix = 0.1):
        self.state = state
        cdef int mc = self.state.getMoveCount()
    
        self.noiseMix = noiseMix
        
        self.isExpanded = 0
        
        self.parentMove = parentMove
        self.parentNode = parentNode
        
        self.children = {}
        
        self.dconst = getDconst(mc)

        self.numMoves = mc

        self.edgePriors = np.zeros(mc, dtype=np.float32)
        self.edgeVisits = np.zeros(mc, dtype=np.float32)
        self.edgeTotalValues = np.zeros(mc, dtype=np.float32)
        self.edgeMeanValues = np.zeros(mc, dtype=np.float32)
        self.valueTmp = np.zeros(mc, dtype=np.float32)
        
        self.edgeLegal = np.zeros(mc, dtype=np.int8)
        cdef int m
        for m in range(self.numMoves):
            if self.state.isMoveLegal(m):
                self.edgeLegal[m] = 1
        
        self.terminalResult = None
        
        self.noiseCache = None
        
        self.stateValue = 0.5
        
        self.allVisits = 0
    
    cdef TreeNode executeMove(self, int move):
        newState = self.state.clone()
        newState.simulate(move)
        return TreeNode(newState, self, parentMove = move, noiseMix = self.noiseMix)

    def exportTree(self):
        """
        create an independent data structure that describes the entire tree 
        that starts at this node, meant for storage and later analysis
        """
        me = {}
        me["state"] = self.state.packageForDebug()
        me["expanded"] = self.isExpanded
        me["winner"] = self.state.getWinner()
        me["priors"] = np.asarray(self.edgePriors).tolist()
        me["netValue"] = np.asarray(self.netValueEvaluation).tolist()

        edges = {}
        
        cdef int move
        
        for move in self.children:
            child = self.children[move]
            e = {}
            e["move"] = self.state.getHumanMoveDescription(move)
            e["tree"] = child.exportTree()
            e["visits"] = self.edgeVisits[move]
            e["totalValue"] = self.edgeTotalValues[move]
            e["meanValue"] = self.edgeMeanValues[move]
            
            edges[move] = e

        me["edges"] = edges
        
        return me
    
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
        self.isExpanded = 0
        self.parentMove = 0
        self.parentNode = None
        self.terminalResult = None
        
        for i in range(self.numMoves):
            self.edgePriors[i] = 0
            self.edgeVisits[i] = 0
            self.edgeTotalValues[i] = 0
            self.edgeMeanValues[i] = 0
       
        self.stateValue = 0.5
        self.allVisits = 0
       
    cdef float getVisitsFactor(self):
        # .0001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0
        # but there is little effkedect in other cases
        return self.allVisits ** 0.5 + 0.0001

    cdef TreeNode selectDown(self, float cpuct):
        cdef TreeNode node = self
        while node.isExpanded and not node.state.isTerminal():
            node = node.selectMove(cpuct)
        return node
        
    def getChildForMove(self, int move):
        assert self.isExpanded
        
        cdef TreeNode child = None
        
        if not move in self.children:
            child = self.executeMove(move)
            self.children[move] = child
        else:
            child = self.children[move]
            
        child.parentNode = None
        return child
    
    def getEdgePriors(self):
        return np.copy(np.asarray(self.edgePriors, dtype=np.float32))
    
    def getMoveDistribution(self):
        return np.asarray(self.edgeVisits, dtype=np.float32) / float(self.allVisits)
    
    cdef int pickMove(self, float cpuct):
        cdef int useNoise = self.parentNode == None
        
        cdef int i

        cdef float nodeQ, nodeU

        if useNoise and self.noiseCache is None:
            self.noiseCache = np.random.dirichlet(self.dconst).astype(np.float32)
        
        cdef float vFactor = self.getVisitsFactor() 

        cdef float bestKnownEdgeMeanValue = np.max(self.edgeMeanValues)

        for i in range(self.numMoves):
            if self.edgeLegal[i] == 1:
                if useNoise:
                    self.valueTmp[i] = (1 - self.noiseMix) * self.edgePriors[i] + self.noiseMix * self.noiseCache[i]
                else:
                    self.valueTmp[i] = self.edgePriors[i]
                
                # not using an initialization of zero is a pretty good idea.
                # not only for search quality (to be proven) but also for search speed by like 50%
                # zero may be bad, stateValue is far worse! That means that especially for very clear cut
                # situations the tree search will start to extensively explore bad plays to the point of diminishing the winning play probability quite considerably.
                if self.edgeVisits[i] == 0:
                    # idea: if the current position is expected to be really good: Follow the network
                    # if the current position is expected to be really bad: explore more, especially if all known options are bad
                    nodeQ = self.stateValue * self.edgePriors[i] + (1 - self.stateValue) * (1 - bestKnownEdgeMeanValue)
                else:
                    nodeQ = self.edgeMeanValues[i]
                    
                nodeU = self.valueTmp[i] * (vFactor / (1.0 + self.edgeVisits[i]))
                
                self.valueTmp[i] = nodeQ + cpuct * nodeU
            else:
                self.valueTmp[i] = illegalMoveValue
       
        cdef int result = bestLegalValue(self.valueTmp)
        
        return result
        
    
    cdef TreeNode selectMove(self, float cpuct):
        move = self.pickMove(cpuct)
        
        if not move in self.children:
            self.children[move] = self.executeMove(move)

        return self.children[move]
    
    cdef void backup(self, object vs):
        cdef int pMove
        cdef TreeNode pNode = self.parentNode
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
    
    cdef void expand(self, object movePMap, object vs):
        np.copyto(np.asarray(self.edgePriors), movePMap, casting="no")
        self.isExpanded = 1
        self.netValueEvaluation = np.array(vs)
        self.stateValue = vs[self.state.getPlayerOnTurnIndex()]
    
    def getNetValueEvaluation(self):
        return self.netValueEvaluation
    
def batchedMcts(object states, int expansions, evaluator, float cpuct):
    workspace = states
    
    cdef TreeNode tmp, node
    
    for _ in range(expansions):
        tlst = workspace
        workspace = [] 
        for i in range(len(tlst)):
            tmp = tlst[i]
            workspace.append(tmp.selectDown(cpuct))
        
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

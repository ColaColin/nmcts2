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

import time

# TODO this should be part of the config....
cdef float DRAW_VALUE = 0.1 # 1 means draws are considered just as good as wins, 0 means draws are considered as bad as losses

cdef float DESPERATION_FACTOR = 0.1

# negative values should be not possible for moves in general?!
cdef float illegalMoveValue = -1

cdef int hasAncestor(TreeNode child, TreeNode ancestor):
    if child == ancestor:
        return 1
    for pNode, _ in child.parentNodes:
        if hasAncestor(pNode, ancestor):
            return 1
    return 0

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
        dconsts[n] = np.asarray([10.0 / n] * n, dtype=np.float32)
    return dconsts[n]

cdef class TreeNode():
    
    cdef readonly object state
    
    cdef float noiseMix
    cdef int isExpanded
    
    cdef object parentNodes

    cdef object children
    
    cdef int useNodeRepository
    cdef object nodeRepository

    cdef unsigned short [:] compressedMoveToMove
    
    cdef int numMoves
    cdef float [:] edgePriors
    cdef float [:] edgeVisits
    cdef float [:] edgeTotalValues

    cdef float[:] noiseCache

    cdef int winningMove
    cdef object terminalResult

    cdef float stateValue
    cdef int allVisits
    
    cdef object netValueEvaluation

    def __init__(self, state, parentNodes = [], noiseMix = 0.25, nodeRepository = None, useNodeRepository = True):
        self.state = state
        
        self.useNodeRepository = useNodeRepository
    
        if self.useNodeRepository:
            if nodeRepository is None:
                self.nodeRepository = {}
            else:
                self.nodeRepository = nodeRepository
    
        self.noiseMix = noiseMix
        
        self.isExpanded = 0
        
        self.parentNodes = parentNodes
        
        self.children = {}
        
        self.winningMove = -1
        self.terminalResult = None
        
        self.noiseCache = None
        
        self.stateValue = 0.5 #overwritten before first use
        
        self.allVisits = 0
        
        self.numMoves = -1
        self.compressedMoveToMove = None
        self.edgeVisits = None
        self.edgeTotalValues = None
        
        self.edgePriors = None

    cdef void lazyInitEdgeData(self):
        if self.numMoves == -1:
            legalMoves = self.state.getLegalMoves()
            self.numMoves = len(legalMoves)
            self.compressedMoveToMove = np.array(legalMoves, dtype=np.uint16)
            self.edgeVisits = np.zeros(self.numMoves, dtype=np.float32)
            self.edgeTotalValues = np.zeros(self.numMoves, dtype=np.float32)
    
    cdef void backupWinningMove(self, int move):
        if self.winningMove != -1:
            return
    
        self.winningMove = move
        
        cdef int pMove
        cdef TreeNode pNode
    
        for pNode, pMove in self.parentNodes:
            if pNode.state.getPlayerOnTurnIndex() == self.state.getPlayerOnTurnIndex():
                pNode.backupWinningMove(pNode.compressedMoveToMove[pMove])
            else:
                break
    
    cdef TreeNode executeMove(self, int move):
        cdef object newState = self.state.clone()
        newState.simulate(move)
        
        if newState.isTerminal() and newState.getWinner() == self.state.getPlayerOnTurnIndex():
            self.backupWinningMove(move)
        
        cdef TreeNode knownNode
        
        cdef int ix
        cdef int compressedNodeIdx = -1
        
        for ix in range(self.numMoves):
            if self.compressedMoveToMove[ix] == move:
                compressedNodeIdx = ix
                break
            
        if self.useNodeRepository and newState in self.nodeRepository:
            knownNode = self.nodeRepository[newState]
            knownNode.parentNodes.append((self, compressedNodeIdx))
            return knownNode
        
        cdef TreeNode newNode = TreeNode(newState, [(self, compressedNodeIdx)], noiseMix = self.noiseMix, nodeRepository = self.nodeRepository, useNodeRepository = self.useNodeRepository) 
        
        if self.useNodeRepository:
            self.nodeRepository[newState] = newNode
        
        return newNode

    def exportTree(self):
        """
        create an independent data structure that describes the entire tree 
        that starts at this node, meant for storage and later analysis
        """
        me = {}
        me["state"] = self.state.packageForDebug()
        me["expanded"] = self.isExpanded
        me["winner"] = self.state.getWinner()
        if self.isExpanded:
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
            e["meanValue"] = self.edgeTotalValues[move] / self.edgeVisits[move]
            
            edges[move] = e

        me["edges"] = edges
        
        return me
    
    def getBestValue(self):
        """
        returns a single float that is meant to tell what the best 
        possible expected outcome is by choosing the best possible actions
        """
        
        if self.winningMove != -1:
            return 1
        
        if self.numMoves == -1:
            return 0
        
        cdef float bestValue = 0
        cdef int i
        
        for i in range(self.numMoves):
            if self.edgeVisits[i] > 0 and (self.edgeTotalValues[i] / self.edgeVisits[i]) > bestValue:
                bestValue = (self.edgeTotalValues[i] / self.edgeVisits[i])
        
        return bestValue
    
    def cutTree(self):
        """
        deletes all children, reducing the tree to the root
        resets all counters
        meant to be used when different solvers are used in an alternating fashion on the same tree.
        maybe instead a completely different tree should be used for each solver. But meh.
        Training does reuse the trees, test play doesn't. Better than nothing...
        
        TODO Why even have a function like this, instead of just grabbing the game state and creating a new root node around it?!
        """
        
        self.children = {}
        self.isExpanded = 0
        self.parentNodes = []
        self.terminalResult = None
        self.nodeRepository = {}

        cdef int i

        if self.edgePriors is not None:
            for i in range(self.state.getMoveCount()):
                self.edgePriors[i] = 0 
        
        if self.numMoves != -1:
            for i in range(self.numMoves):
                self.edgeVisits[i] = 0
                self.edgeTotalValues[i] = 0
      
        self.stateValue = 0.5
        self.allVisits = 0
       
    cdef float getVisitsFactor(self):
        # .0001 means that in the case of a new node with zero visits it will chose whatever has the best P
        # instead of just the move with index 0
        # but there is little effect in other cases
        return self.allVisits ** 0.5 + 0.0001

    cdef TreeNode selectDown(self, float cpuct):
        cdef TreeNode node = self
        while node.isExpanded and not node.state.isTerminal():
            node = node.selectMove(cpuct)
        return node
    
    def getTreeDepth(self):
        if len(self.children) == 0:
            return 1
        return max([self.children[ckey].getTreeDepth() for ckey in self.children]) + 1 
    
    def getChildForMove(self, int move):
        self.lazyInitEdgeData()
        
        cdef TreeNode child = None
        
        if not move in self.children:
            child = self.executeMove(move)
            self.children[move] = child
        else:
            child = self.children[move]
            
        child.parentNodes = []
        
        cdef TreeNode cached
        
        if self.useNodeRepository:
            cKeys = list(self.nodeRepository.keys())
            for childKey in cKeys:
                cached = self.nodeRepository[childKey]
                if not hasAncestor(cached, child):
                    del self.nodeRepository[childKey]
                    cached.parentNodes = []
                    cached.children = {}
        
        return child
    
    def getEdgePriors(self):
        if self.edgePriors is None:
            return np.zeros(self.state.getMoveCount(), dtype=np.float32)
        else:
            return np.copy(np.asarray(self.edgePriors, dtype=np.float32))
    
    def getMoveDistribution(self):
        result = np.zeros(self.state.getMoveCount(), dtype=np.float32)
        
        if self.winningMove != -1:
            result[self.winningMove] = 1
        else:
            result[self.compressedMoveToMove] = self.edgeVisits
            result /= float(self.allVisits)
            
        return result
    
    cdef int pickMove(self, float cpuct):
        if self.winningMove != -1:
            return self.winningMove
    
        cdef int useNoise = len(self.parentNodes) == 0
        
        cdef int i

        cdef float nodeQ, nodeU

        if useNoise and self.noiseCache is None:
            self.noiseCache = np.random.dirichlet(getDconst(self.numMoves)).astype(np.float32)
        
        cdef float vFactor = self.getVisitsFactor() 

        cdef float [:] valueTmp = np.zeros(self.numMoves, dtype=np.float32)

        cdef int decompressedMove

        for i in range(self.numMoves):
            decompressedMove = self.compressedMoveToMove[i]
            if useNoise:
                valueTmp[i] = (1 - self.noiseMix) * self.edgePriors[decompressedMove] + self.noiseMix * self.noiseCache[i]
            else:
                valueTmp[i] = self.edgePriors[decompressedMove]
            
            # not using an initialization of zero is a pretty good idea.
            # not only for search quality (to be proven) but also for search speed by like 50%
            # zero may be bad, stateValue is far worse! That means that especially for very clear cut
            # situations the tree search will start to extensively explore bad plays to the point of diminishing the winning play probability quite considerably.
            if self.edgeVisits[i] == 0:
                # idea: if the current position is expected to be really good: Follow the network
                nodeQ = self.stateValue * self.edgePriors[decompressedMove] + (1 - self.stateValue) * DESPERATION_FACTOR
            else:
                nodeQ = self.edgeTotalValues[i] / self.edgeVisits[i]
                
            nodeU = valueTmp[i] * (vFactor / (1.0 + self.edgeVisits[i]))
            
            valueTmp[i] = nodeQ + cpuct * nodeU
       
        cdef int result = bestLegalValue(valueTmp)
        
        return self.compressedMoveToMove[result]
    
    cdef TreeNode selectMove(self, float cpuct):
        self.lazyInitEdgeData()
    
        move = self.pickMove(cpuct)
        
        if not move in self.children:
            self.children[move] = self.executeMove(move)

        return self.children[move]
    
    cdef void backup(self, object vs):
        cdef int pMove
        cdef TreeNode pNode
        
        for pNode, pMove in self.parentNodes:
            pNode.edgeVisits[pMove] += 1
            pNode.allVisits += 1
            pNode.edgeTotalValues[pMove] += vs[pNode.state.getPlayerOnTurnIndex()]
            if pNode.state.hasDraws():
                pNode.edgeTotalValues[pMove] += vs[self.state.getPlayerCount()] * DRAW_VALUE
            pNode.backup(vs)
    
    def getTerminalResult(self):
        if self.terminalResult is None:
            numOutputs = self.state.getPlayerCount()
            if self.state.hasDraws():
                numOutputs += 1
                
            r = [0] * numOutputs
            winner = self.state.getWinner()
            if winner != -1:
                r[winner] = 1
            else:
                if self.state.hasDraws():
                    r[numOutputs-1] = 1
                else:
                    r = [1.0 / self.state.getPlayerCount()] * self.state.getPlayerCount()
            self.terminalResult = np.array(r, dtype=np.float32)

        return self.terminalResult
    
    cdef void expand(self, object movePMap, object vs):
        self.edgePriors = np.zeros(self.state.getMoveCount(), dtype=np.float32)
        np.copyto(np.asarray(self.edgePriors), movePMap, casting="no")
        self.isExpanded = 1
        self.netValueEvaluation = vs
        self.stateValue = vs[self.state.getPlayerOnTurnIndex()]
        if (self.state.hasDraws()):
            self.stateValue += vs[self.state.getPlayerCount()] * DRAW_VALUE
    
    def getNetValueEvaluation(self):
        return self.netValueEvaluation



    # init
    # prepareA
    # evaluate A
    # prepare B
    
    # GPU        - CPU
    # do:
    # evaluate B - backup A, prepare A 
    # evaluate A - backup B, prepare B
    # repeat
    
    # complete
    # backupA
    
def backupWork(backupSet, evalout):
    cdef TreeNode node

    for idx, ev in enumerate(evalout):
        node = backupSet[idx]
        w = ev[1]
        if node.state.isTerminal():
            w = node.getTerminalResult()
        else:
            node.expand(ev[0], ev[1])
        node.backup(w)

def cpuWork(prepareSet, backupSet, evalout, cpuct):
    prepareResult = []
    
    cdef TreeNode tnode
    
    if backupSet is not None:
        backupWork(backupSet, evalout)
    
    for i in range(len(prepareSet)):
        tnode = prepareSet[i]
        prepareResult.append(tnode.selectDown(cpuct))

    return prepareResult

def batchedMcts(object states, int expansions, evaluator, float cpuct):
    workspace = states
    
    halfw = len(workspace) // 2
    
    workspaceA = workspace[:halfw]
    workspaceB = workspace[halfw:]

    asyncA = True

    preparedDataA = cpuWork(workspaceA, None, None, cpuct)
    evaloutA = evaluator(preparedDataA)
    
    preparedDataB = cpuWork(workspaceB, None, None, cpuct)
    evaloutB = None
    
    def asyncWork():
        nonlocal preparedDataA
        nonlocal preparedDataB
        nonlocal asyncA
        nonlocal cpuct
        nonlocal evaloutA
        nonlocal evaloutB
        nonlocal workspaceA
        nonlocal workspaceB
        
        if asyncA:
            preparedDataA = cpuWork(workspaceA, preparedDataA, evaloutA, cpuct)
        else:
            preparedDataB = cpuWork(workspaceB, preparedDataB, evaloutB, cpuct)

    for _ in range(expansions):
        for _ in range(2):
            if asyncA:
                evaloutB = evaluator(preparedDataB, asyncWork)
            else:
                evaloutA = evaluator(preparedDataA, asyncWork)
            asyncA = not asyncA

    backupWork(preparedDataA, evaloutA)

#     cdef TreeNode tmp, node
#     
#     for _ in range(expansions):
#         tlst = workspace
#         workspace = [] 
#         for i in range(len(tlst)):
#             tmp = tlst[i]
#             workspace.append(tmp.selectDown(cpuct))
# 
#         evalout = evaluator(workspace)
#         
#         for idx, ev in enumerate(evalout):
#             node = workspace[idx]
#             
#             w = ev[1]
#             if node.state.isTerminal():
#                 w = node.getTerminalResult()
#             else:
#                 node.expand(ev[0], ev[1])
#             node.backup(w)
#             workspace[idx] = states[idx]

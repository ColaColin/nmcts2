# cython: profile=False

from libc.stdlib cimport qsort
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdlib cimport rand, RAND_MAX

import numpy as np
import random

cdef struct MoveData:
    int move
    float q
    float s
    double value

cdef int cmp_move_data(const void* a, const void *b) nogil:
    cdef MoveData* a_v = <MoveData*> a
    cdef MoveData* b_v = <MoveData*> b
    if a_v.value < b_v.value:
        return 1
    elif a_v.value == b_v.value:
        return 0
    else:
        return -1

cdef class TreeEdge():
    cdef int visitCount
    cdef float totalValue
    cdef float meanValue
    cdef float priorP
    cdef TreeNode parentNode
    cdef TreeNode childNode
    
    def __init__(self, float priorP, TreeNode parentNode):
        self.visitCount = 0
        self.totalValue = 0
        # TODO this should not be a hardcoded value
        self.meanValue = 0.5 # TODO have a look at modeling this as a distribution instead of a mean, see arXiv 1707.06887 as detailed inspiration. How to apply that to MCTS?
        self.priorP = priorP
        self.parentNode = parentNode
        self.childNode = None

cdef class TreeNode():
    
    cdef object edges
    cdef TreeEdge parent
    # how many times an action was chosen on this TreeNode
    cdef int allVisits
    
    cdef int hasHighs
    cdef int numHighs
    cdef int* highs
    
    cdef float lowS
    cdef float lowQ

    cdef float stateValue
    
    cdef object movePMap
    
    cdef float noiseMix
    
    cdef int isExpanded

    cdef readonly object state

    cdef object terminalResult

    cdef object dconst
    
    # index of the player that will get more mcts expansions in this state and all it's children
    # used for unbalanceTrainingMctsFactor
    cdef readonly int advantagePlayerIdx

    cdef float currentGroupSizeFactor
    cdef int noRegroupsNeededCount

    def __init__(self, state, noiseMix = 0.1):
        mc = state.getMoveCount()
        self.state = state
        self.noiseMix = noiseMix
        self.edges = [None] * mc
        self.movePMap = None
        self.highs = NULL
        self.hasHighs = 0
        self.terminalResult = None
        self.dconst = np.asarray([0.03] * mc, dtype="float32")
        self.currentGroupSizeFactor = 0.89133742
        self.noRegroupsNeededCount = 0
        self.allVisits = 0
        self.advantagePlayerIdx = int((rand()/(<float>RAND_MAX)) * state.getPlayerCount())
    
    def __dealloc__(self):
        if self.highs != NULL:
            free(self.highs)
    
    # .0001 means that in the case of a new node with zero visits it will chose whatever has the best P
    # instead of just the move with index 0
    # but there is little effect in other cases
    cdef inline float getVisitsFactor(self):
        return self.allVisits ** 0.5 + 0.0001
    
    cdef void prepareMoveGrouping(self, int numLegalMoves, object lMoves, float f, float cpuct, MoveData* moves_c):
        cdef int move, vc, idx, biasedIdx
        cdef TreeEdge e
        cdef float q, p, s
        
        cdef MoveData* mtmp
        
        cdef int startIdx = int((rand()/(<float>RAND_MAX)) * numLegalMoves)
        
        for biasedIdx in range(numLegalMoves):
            idx = (biasedIdx + startIdx) % numLegalMoves
            
            move = lMoves[idx]
            e = self.edges[move]
            
            if e is not None:
                q = e.meanValue
                p = e.priorP
                vc = e.visitCount
            else:
                q = self.stateValue
                p = self.movePMap[move]
                vc = 0
                
            s = cpuct * p / (1.0 + vc)
            
            mtmp = &moves_c[idx]
            mtmp.move = move
            mtmp.q = q
            mtmp.s = s
            mtmp.value = (<double>q) + (<double>s) * (<double>f)
        
        qsort(&moves_c[0], numLegalMoves, sizeof(MoveData), &cmp_move_data)
    
    cdef void groupCurrentMoves(self, float cpuct, int* bestMove):
        cdef object lMoves = self.state.getLegalMoves()
        cdef int numLegalMoves = len(lMoves)
        
        cdef float f = self.getVisitsFactor()
        
        cdef MoveData* moves_c = <MoveData*> malloc(numLegalMoves * sizeof(MoveData))
        
        self.prepareMoveGrouping(numLegalMoves, lMoves, f, cpuct, moves_c)
       
        cdef int highLen = numLegalMoves - <int>(numLegalMoves * self.currentGroupSizeFactor)
        
        cdef int minHighs = 5
        if highLen < minHighs:
            highLen = min(minHighs, numLegalMoves)
        
        cdef int needMalloc = not self.hasHighs
        if self.hasHighs and self.numHighs < highLen:
            free(self.highs)
            needMalloc = 1
        
        self.hasHighs = 1
        self.numHighs = highLen
        if needMalloc:
            self.highs = <int*> malloc(self.numHighs * sizeof(int));
        
        cdef int i
        for i in range(self.numHighs):
            self.highs[i] = moves_c[i].move
        
        # at this point all moves in highs are legal, no need to check
        bestMove[0] = self.highs[0]
        
        cdef float lqc, lsc
        
        cdef int lowElems = numLegalMoves - highLen
        cdef int lowIdx
        
        if lowElems > 0:
            self.lowQ = moves_c[highLen].q
            self.lowS = moves_c[highLen].s
            
            for lowIdx in range(1, lowElems):
                lqc = moves_c[highLen + lowIdx].q
                lsc = moves_c[highLen + lowIdx].s
                
                if lqc > self.lowQ:
                    self.lowQ = lqc
                    
                if lsc > self.lowS:
                    self.lowS = lsc
        else:
            self.lowQ = 0
            self.lowS = 0
            
        free(moves_c)
    
    cdef void pickMoveFromHighs(self, float cpuct, int* moveName, float* moveValue, int useGrouping):
        
        cdef float allVisitsSq = self.getVisitsFactor()

        cdef int numKeys = self.numHighs
        
        if not useGrouping:
            moveKeys = self.state.getLegalMoves()
            numKeys = len(moveKeys)
        
        cdef int useNoise = self.allVisits < 5
        
        cdef double[:] dirNoise
        if useNoise:
            dirNoise = np.random.dirichlet(self.dconst[:numKeys])
        
        cdef int startIdx = int((rand()/(<float>RAND_MAX)) * numKeys)
        
#             startIdx = 0
#             dirNoise = np.zeros_like(dirNoise)
        
        cdef int biasedIdx, idx
        
        cdef TreeEdge e
        
        cdef float p, u, value, iNoise
        iNoise = 0
        
        cdef int vc
        
        for biasedIdx in range(numKeys):
            idx = (biasedIdx + startIdx) % numKeys
            
            if useNoise:
                iNoise = dirNoise[idx]
            
            if useGrouping:
                idx = self.highs[idx]
            else:
                idx = moveKeys[idx]
            
            e = self.edges[idx]
            
            if e is not None:
                q = e.meanValue
                p = e.priorP
                vc = e.visitCount
            else:
                q = self.stateValue
                p = self.movePMap[idx]
                vc = 0
            
            if useNoise:
                p = (1 - self.noiseMix) * p + self.noiseMix * iNoise
            u = cpuct * p * (allVisitsSq / (1.0 + vc))
            
            value = q + u
            
            if (moveName[0] == -1 or value > moveValue[0]) and self.state.isMoveLegal(idx):
                moveName[0] = idx
                moveValue[0] = value
    
    cdef TreeNode selectMove(self, float cpuct):
        cdef int moveName = -1
        cdef float fastMoveValue = 0
        cdef float lowersBestValue = 0
        
        # on a 19x19 board this can yield 100%+ improvement...
        cdef int useGrouping = 1
        
        if not useGrouping:
            self.pickMoveFromHighs(cpuct, &moveName, &fastMoveValue, useGrouping)
        else:
            if not self.hasHighs:
                self.groupCurrentMoves(cpuct, &moveName)
            else:
                self.pickMoveFromHighs(cpuct, &moveName, &fastMoveValue, useGrouping)
                lowersBestValue = self.lowQ + self.lowS * self.getVisitsFactor()
                
                if lowersBestValue >= fastMoveValue:
                    self.groupCurrentMoves(cpuct, &moveName)
        
        cdef TreeEdge edge = self.edges[moveName]
        
        if edge is None:
            edge = TreeEdge(self.movePMap[moveName], self)
            self.edges[moveName] = edge
        
        if edge.childNode is None:
            edge.childNode = self.executeMove(moveName)
        
        return edge.childNode
    
    cdef TreeNode selectDown(self, float cpuct):
        cdef TreeNode node = self
        while node.isExpanded and not node.state.isTerminal():
            node = node.selectMove(cpuct)
        return node
    
    cdef void backup(self, object vs):
        if self.parent is not None:
            self.parent.visitCount += 1
            self.parent.parentNode.allVisits += 1
            self.parent.totalValue += vs[self.parent.parentNode.state.getPlayerOnTurnIndex()]
            self.parent.meanValue = self.parent.totalValue / self.parent.visitCount
            self.parent.parentNode.backup(vs)

    cdef void expand(self, object movePMap, object vs):
        self.movePMap = movePMap
        self.isExpanded = 1
        self.stateValue = vs[self.state.getPlayerOnTurnIndex()]
    
    cdef TreeNode executeMove(self, int move):
        cdef object newState = self.state.clone();
        newState.simulate(move)

        cdef TreeNode result = TreeNode(newState, noiseMix = self.noiseMix)
        result.advantagePlayerIdx = self.advantagePlayerIdx
        result.parent = self.edges[move] 
        
        return result
    
    # methods used by other python to get results or modify stuff somehow
    
    def getBestValue(self):
        """
        returns a single float that is meant to tell what the best 
        possible expected outcome is by chosing the best possible actions
        """
        cdef float bv = 0
        cdef int i = 0
        cdef int elen = len(self.edges)
        cdef TreeEdge e 
        for i in range(elen):
            e = self.edges[i]
            if e is not None and e.meanValue > bv:
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
        self.allVisits = 0
        self.isExpanded = 0
        
        self.hasHighs = 0
        if self.highs != NULL:
            free(self.highs)
        self.highs = NULL
        self.lowS = 0
        self.lowQ = 0
    
    def getChildForMove(self, int move):
        """
        call this to do a move, returning a new root of the tree
        """
        assert self.isExpanded
        
        cdef TreeEdge edge
        cdef TreeNode child 
        
        edge = self.edges[move]
        if edge is None:
            edge = TreeEdge(self.movePMap[move], self)
            self.edges[move] = edge
            
        child = edge.childNode
        
        if child is None:
            edge.childNode = self.executeMove(move)
            child = edge.childNode
        
        child.parent = None
        return child
    
    def getMoveDistribution(self):
        cdef float sumv = float(self.allVisits)
        
        cdef int elen = len(self.edges)
        cdef object r = [0] * elen
        cdef TreeEdge e
        for i in range(elen):
            e = self.edges[i]
            if e is not None:
                r[i] = e.visitCount / sumv
        
        return r
    
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

def batchedMcts(object states, int expansions, evaluator, float cpuct):
    workspace = [s for s in states]
    
    cdef TreeNode tmp, node
    cdef int evallen, ixx
    
    for ixx in range(expansions):
        tlst = workspace
        
        workspace = []
        for i in range(len(tlst)):
            tmp = tlst[i]
            if tmp is None:
                workspace.append(None)
            else:
                workspace.append(tmp.selectDown(cpuct))

        evalout = evaluator(workspace)
        evallen = len(evalout)
        for idx in range(evallen):
            ev = evalout[idx]
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

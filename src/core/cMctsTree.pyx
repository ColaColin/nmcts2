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

cdef struct TreeEdge:
    int visitCount
    float totalValue
    float meanValue
    float priorP
    int exists

cdef void setupEdge(TreeEdge* ptr, float priorP):
    ptr[0].visitCount = 0
    ptr[0].totalValue = 0
    ptr[0].exists = 1
    ptr[0].meanValue = 0.5 # TODO this should not be a hardcoded value
    ptr[0].priorP = priorP

cdef class TreeNode():
    
    cdef int numEdges
    cdef TreeEdge* edges
    cdef TreeEdge* parentEdge
    
    cdef TreeNode parentNode
    
    cdef object childNodes
    
    cdef int* legalMoveKeys
    cdef int numLegalMoves
    
    # how many times an action was chosen on this TreeNode
    cdef int allVisits
    
    cdef int hasHighs
    cdef int numHighs
    cdef int* highs
    
    cdef float lowS
    cdef float lowQ

    cdef float stateValue
    
    cdef float* movePMap
    
    cdef float noiseMix
    
    cdef int isExpanded

    cdef readonly object state

    cdef object terminalResult

    cdef object dconst

    cdef float currentGroupSizeFactor
    cdef int noRegroupsNeededCount

    cdef MoveData* moves_c

    def __init__(self, state, noiseMix = 0.1):
        self.state = state
        self.noiseMix = noiseMix

        mc = self.state.getMoveCount()
        
        self.numEdges = mc
        
        self.edges = <TreeEdge*> malloc(mc * sizeof(TreeEdge))
        for i in range(self.numEdges):
            self.edges[i].exists = 0

        self.parentEdge = NULL
        
        self.childNodes = [None] * self.numEdges
        self.parentNode = None
        
        cdef object lObjMoves = self.state.getLegalMoves()
        self.numLegalMoves = len(lObjMoves)
        self.legalMoveKeys = <int*> malloc(self.numLegalMoves * sizeof(int))
        
        for i in range(self.numLegalMoves):
            self.legalMoveKeys[i] = lObjMoves[i]
        
        self.moves_c = <MoveData*> malloc(self.numLegalMoves * sizeof(MoveData))
        
        self.movePMap = <float*> malloc(self.numEdges * sizeof(float))
        
        self.highs = NULL
        self.hasHighs = 0
        self.terminalResult = None
        self.dconst = np.asarray([0.03] * mc, dtype="float32")
        self.currentGroupSizeFactor = 0.89133742
        self.noRegroupsNeededCount = 0
        self.allVisits = 0
    
    def __dealloc__(self):
        if self.highs != NULL:
            free(self.highs)
        free(self.edges)
        free(self.legalMoveKeys)
        free(self.moves_c)
        free(self.movePMap)
    
    # .0001 means that in the case of a new node with zero visits it will chose whatever has the best P
    # instead of just the move with index 0
    # but there is little effect in other cases
    cdef inline float getVisitsFactor(self):
        return self.allVisits ** 0.5 + 0.0001
    
    cdef void prepareMoveGrouping(self, float f, float cpuct, MoveData* moves_c):
        cdef int move, vc, idx, biasedIdx
        cdef TreeEdge* e
        cdef float q, p, s
        
        cdef MoveData* mtmp
        
        cdef int startIdx = int((rand()/(<float>RAND_MAX)) * self.numLegalMoves)
        
        for biasedIdx in range(self.numLegalMoves):
            idx = (biasedIdx + startIdx) % self.numLegalMoves
            
            move = self.legalMoveKeys[idx]
            e = self.edges + move
            
            if e[0].exists:
                q = e[0].meanValue
                p = e[0].priorP
                vc = e[0].visitCount
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
        
        qsort(&moves_c[0], self.numLegalMoves, sizeof(MoveData), &cmp_move_data)
    
    cdef void groupCurrentMoves(self, float cpuct, int* bestMove):
        cdef float f = self.getVisitsFactor()
        
        self.prepareMoveGrouping(f, cpuct, self.moves_c)
       
        cdef int highLen = self.numLegalMoves - <int>(self.numLegalMoves * self.currentGroupSizeFactor)
        
        cdef int minHighs = 5
        if highLen < minHighs:
            highLen = min(minHighs, self.numLegalMoves)
        
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
            self.highs[i] = self.moves_c[i].move
        
        # at this point all moves in highs are legal, no need to check
        bestMove[0] = self.highs[0]
        
        cdef float lqc, lsc
        
        cdef int lowElems = self.numLegalMoves - highLen
        cdef int lowIdx
        
        if lowElems > 0:
            self.lowQ = self.moves_c[highLen].q
            self.lowS = self.moves_c[highLen].s
            
            for lowIdx in range(1, lowElems):
                lqc = self.moves_c[highLen + lowIdx].q
                lsc = self.moves_c[highLen + lowIdx].s
                
                if lqc > self.lowQ:
                    self.lowQ = lqc
                    
                if lsc > self.lowS:
                    self.lowS = lsc
        else:
            self.lowQ = 0
            self.lowS = 0
    
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
        
        cdef TreeEdge* e
        
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
            
            e = self.edges + idx
            
            if e[0].exists:
                q = e[0].meanValue
                p = e[0].priorP
                vc = e[0].visitCount
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
        
        cdef TreeEdge* edge = self.edges + moveName
        
        if edge[0].exists == 0:
            setupEdge(edge, self.movePMap[moveName])
        
        if self.childNodes[moveName] is None:
            self.childNodes[moveName] = self.executeMove(moveName)
        
        return self.childNodes[moveName]
    
    cdef TreeNode selectDown(self, float cpuct):
        cdef TreeNode node = self
        while node.isExpanded and not node.state.isTerminal():
            node = node.selectMove(cpuct)
        return node
    
    cdef void backup(self, object vs):
        if self.parentNode is not None:
            self.parentEdge[0].visitCount += 1
            self.parentEdge[0].totalValue += vs[self.parentNode.state.getPlayerOnTurnIndex()]
            self.parentEdge[0].meanValue = self.parentEdge[0].totalValue / self.parentEdge[0].visitCount
            
            self.parentNode.allVisits += 1
            self.parentNode.backup(vs)

    cdef void expand(self, object movePMapTensor, object vs):
        for i in range(self.numEdges):
            self.movePMap[i] = movePMapTensor[i]
        self.isExpanded = 1
        self.stateValue = vs[self.state.getPlayerOnTurnIndex()]
    
    cdef TreeNode executeMove(self, int move):
        cdef object newState = self.state.clone();
        newState.simulate(move)

        cdef TreeNode result = TreeNode(newState, noiseMix = self.noiseMix)
        result.parentEdge = self.edges + move
        result.parentNode = self
        
        return result
    
    # methods used by other python to get results or modify stuff somehow
    
    def getBestValue(self):
        """
        returns a single float that is meant to tell what the best 
        possible expected outcome is by chosing the best possible actions
        """
        cdef float bv = 0
        cdef int i = 0
        cdef int elen = self.numEdges
        cdef TreeEdge* e 
        for i in range(elen):
            e = self.edges + i
            if e[0].exists and e[0].meanValue > bv:
                bv = e[0].meanValue
        return bv
    
    def cutTree(self):
        """
        deletes all children, reducing the tree to the root
        resets all counters
        meant to be used when different solvers are used in an alternating fashion on the same tree.
        maybe instead a completely different tree should be used for each solver. But meh.
        Training does reuse the trees, test play doesn't. Better than nothing...
        """
        for i in range(self.numEdges):
            self.edges[i].exists = 0
            
        self.parentEdge = NULL
        
        self.childNodes = [None] * self.numEdges
        self.parentNode = None
            
        self.terminalResult = None
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
        
        cdef TreeEdge* edge
        cdef TreeNode child 
        
        edge = self.edges + move
        if not edge[0].exists:
            self.parentNode = self
            setupEdge(edge, self.movePMap[move])
            
        child = self.childNodes[move]
        
        if child is None:
            self.childNodes[move] = self.executeMove(move)
            child = self.childNodes[move]
        
        child.parentNode = None
        return child
    
    def getMoveDistribution(self):
        cdef float sumv = float(self.allVisits)
        
        cdef int i
        
        cdef int elen = self.numEdges
        cdef object r = [0] * elen
        cdef TreeEdge* e
        for i in range(elen):
            e = self.edges + i
            if e[0].exists:
                r[i] = e[0].visitCount / sumv
        
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

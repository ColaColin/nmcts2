# cython: profile=False

'''
Created on Apr 15, 2018

@author: cclausen
'''

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

import random

# todo put the field handling in another pyx file and figure out how to organize compilation

# WHITE is player index 1
# BLACK is player index 0

cdef inline float readFloatField(float* f, int m, int x, int y):
    return f[y * m + x];

cdef inline void writeFloatField(float* f, int m, int x, int y, float value):
    f[y * m + x] = value;

cdef float* initFloatField(int m, int n, float v):
    cdef float* result = <float *> malloc(m * n * sizeof(float));
    
    for i in range(n * m):
        result[i] = v
    
    return result;

cdef void mirrorFloatField(float* f, int m, int n):
    cdef int x, y;
    cdef float tmp;
    for y in range(n):
        for x in range(m / 2):
            tmp = readFloatField(f, m, x, y)
            writeFloatField(f, m, x, y, readFloatField(f, m, m-x-1, y))
            writeFloatField(f, m, m-1-x, y, tmp)

cdef void rotateFloatField(float* f, int m, int n):
    cdef float* tmp = initFloatField(m, n, 0);
    memcpy(tmp, f, m * n * sizeof(float));
    
    cdef int x, y
    for y in range(n):
        for x in range(m):
            writeFloatField(f, m, x, y, readFloatField(tmp, m, y, n-x-1))
    
    free(tmp);

cdef inline signed char readField(signed char* f, int m, int x, int y):
    return f[y * m + x];

cdef inline void writeField(signed char* f, int m, int x, int y, signed char value):
    f[y * m + x] = value;

cdef signed char* initField(int m, int n, signed char v):
    cdef signed char* result = <signed char *> malloc(m * n * sizeof(signed char));
    
    for i in range(n * m):
        result[i] = v
    
    return result;

cdef int hashC6(Connect6_c* c6):
    cdef int result = c6.turn
    
    cdef int sums = 0
    
    for i in range(c6.n * c6.m):
        sums += (i + 1) * c6.board[i]
    
    return result | ((<unsigned int> sums) << 8)

cdef int areFieldsEqual(int m, int n, signed char* fieldA, signed char* fieldB):
    cdef int eq = 1

    for i in range(n * m):
        if fieldA[i] != fieldB[i]:
            eq = 0
            break
    
    return eq

cdef void mirrorField(signed char* f, int m, int n):
    cdef int x, y;
    cdef signed char tmp;
    for y in range(n):
        for x in range(m / 2):
            tmp = readField(f, m, x, y)
            writeField(f, m, x, y, readField(f, m, m-1-x, y))
            writeField(f, m, m-1-x, y, tmp)

cdef printField(int m, int n, signed char * field):
    s = ""
    
    for y in range(n):
        for x in range(m):
            s += "{0:.4f}".format(readField(field, m, x, y)) + " "
        s += "\n"
    
    print(s)

cdef void rotateField(signed char* f, int m, int n):
    cdef signed char* tmp = initField(m, n, 0);
    memcpy(tmp, f, m * n * sizeof(signed char));
        
    cdef int x, y
    for y in range(n):
        for x in range(m):
            writeField(f, m, x, y, readField(tmp, m, y, n-1-x))
    
    free(tmp);
    
cdef printFloatField(int m, int n, float* field):
    s = ""
    
    for y in range(n):
        for x in range(m):
            s += "{0:.4f}".format(readFloatField(field, m, x, y)) + " "
        s += "\n"
    
    print(s)

cdef void augmentFieldAndMovesDistribution(int m, int n, signed char * board, 
                                           object moves, object moveIdxToPos, object movePosToIdx):
    
    cdef int dbg = 0
    
    if dbg:
        print("Pre Augment")
        printField(m, n, board)
        print(moves)
    
    cdef signed char * fStateField = board;
    cdef float * fMovesField = initFloatField(m, n, 0);
    
    cdef int idx
    
    for idx in range(m * n):
        x, y = moveIdxToPos(idx)
        writeFloatField(fMovesField, m, x, y, moves[idx])
    
    if dbg:
        printFloatField(m, n, fMovesField)
        
    if random.random() > 0.5:
        if dbg:
            print("Do mirror")
        mirrorField(fStateField, m, n)
        mirrorFloatField(fMovesField, m, n)
        
    rotations = random.randint(0, 3)
    
    if dbg:
        print("Do %i rotations" % rotations)
        
    for _ in range(rotations):
        rotateField(fStateField, m, n)
        rotateFloatField(fMovesField, m, n)
    
    cdef int xi, yi
    
    for yi in range(n):
        for xi in range(m):
            moves[movePosToIdx(xi, yi)] = readFloatField(fMovesField, m, xi, yi)
    
    if dbg:
        print("Post Augment")
        printField(m,n,fStateField)
        print(moves)
        for idx, fMove in enumerate(moves):
            x, y = moveIdxToPos(idx)
            writeFloatField(fMovesField, m, x, y, fMove)
        printFloatField(m, n, fMovesField)
    
    free(fMovesField)

cdef struct Connect6_c:
    int m
    int n
    int k
    int turn
    int winningPlayer
    signed char* board

cdef object toStringC6(Connect6_c* c6, int lastMoveX, int lastMoveY):
        mm = ['.', '░', '█']
        s = "Connect6(%i,%i), " %  (c6.m, c6.n)
        if hasEndedC6(c6) == 0:
            s += "Turn %i: %s\n" % (c6.turn, mm[getPlayerIndexOnTurnC6(c6)+1])
        elif c6.winningPlayer > -1:
            s += "Winner: %s\n" % mm[c6.winningPlayer+1]
        else:
            s += "Draw\n"

        s += "   |"
        for x in range(c6.m):
            if x < 9:
                s += " %i |" % (x+1)
            else:
                s += "%i |" % (x+1)
        
        s += "\n";
        
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        
        assert c6.n <= len(chars)
        
        for y in range(c6.n):
            for x in range(c6.m+1):
                if x-1 == lastMoveX and y == lastMoveY:
                    s += "┌─┐ "
                elif x-1 == lastMoveX and y == lastMoveY+1:
                    s += "└─┘ "
                else:
                    s += "    "
                
            s += "\n";
            s += " " + chars[y] + "  ";
            for x in range(c6.m):
                if x == lastMoveX and y == lastMoveY:
                    s += "│" + mm[readField(c6.board, c6.m, x, y)+1] + "│ ";
                else:
                    s += " " + mm[readField(c6.board, c6.m, x, y)+1] + "  ";
                
            s += "\n";
            
        for _ in range(c6.m+1):
            s += "    ";
        s += "\n";
        
        return s;

cdef Connect6_c* initC6(int m, int n):
    cdef Connect6_c* result = <Connect6_c*> malloc(sizeof(Connect6_c));
    
    result.m = m;
    result.n = n;
    result.k = 6;
    result.winningPlayer = -1;
    result.turn = 0;
    result.board = initField(m, n, -1);
    
    return result;

cdef inline void freeC6(Connect6_c* c6):
    free(c6.board);
    free(c6);

cdef object picklePackC6(Connect6_c* c):
    cdef object result = {}
    cdef object blst = []
    
    result["m"] = c.m;
    result["n"] = c.n
    result["k"] = c.k
    result["wP"] = c.winningPlayer
    result["turn"] = c.turn
    
    for i in range(c.m * c.n):
        blst.append(c.board[i])
    
    result["board"] = blst
    
    return result

cdef Connect6_c* pickleUnpackC6(object package):
    cdef Connect6_c* result = initC6(package["m"], package["n"]);
    
    result.k = package["k"]
    result.winningPlayer = package["wP"]
    result.turn = package["turn"]
    
    for i in range(result.m * result.n):
        result.board[i] = package["board"][i]
    
    return result

cdef Connect6_c* cloneC6(Connect6_c* source):
    cdef Connect6_c* result = <Connect6_c*> malloc(sizeof(Connect6_c));
    
    result.m = source.m;
    result.n = source.n;
    result.k = source.k;
    result.turn = source.turn;
    result.winningPlayer = source.winningPlayer;
    
    result.board = <signed char *> malloc(source.m * source.n * sizeof(signed char));
    memcpy(result.board, source.board, source.m * source.n * sizeof(signed char));
    
    return result;

cdef inline signed char getPlayerIndexOnTurnC6(Connect6_c* c6):
    if c6.turn == 0:
        return 1
    else:
        return ((c6.turn-1)/2) % 2

cdef inline int hasEndedC6(Connect6_c* c6):
    # games are drawn once 2/3 of the field are filled with stones
    return c6.winningPlayer != -1 or c6.turn >= (c6.m * c6.n) * 0.67; 

cdef void _searchWinnerC6(Connect6_c* c6, int lx, int ly):
    if c6.winningPlayer != -1:
        return
    
    cdef signed char p = readField(c6.board, c6.m, lx, ly);
    
    cdef int[4][2] dirs = [[1, 0], [0, 1], [1, -1], [1, 1]]
    cdef int[2] invs = [1, -1]
    
    cdef int l, x, y, xdir, ydir
    
    if p != -1:
        for d in range(4):
            l = 0;
            
            for di in range(2):
                x = lx;
                y = ly;
                
                xdir = invs[di] * dirs[d][0];
                ydir = invs[di] * dirs[d][1];
                
                while x > -1 and y > -1 and x < c6.m and y < c6.n and readField(c6.board, c6.m, x, y) == p:
                    l += 1
                    x += xdir;
                    y += ydir;
                    
                    if l - 1 >= c6.k:
                        c6.winningPlayer = p
                        return
    
cdef void placeC6(Connect6_c* c6, int x, int y):
    writeField(c6.board, c6.m, x, y, getPlayerIndexOnTurnC6(c6))
    c6.turn += 1;
    _searchWinnerC6(c6, x, y);

cdef unsigned long long seed = 1

cdef class Connect6State:
    cdef Connect6_c* c6
    cdef object legalMoves
    
    cdef readonly unsigned long long id
    cdef readonly unsigned long long lastId
    cdef readonly int lastMove
    
    cdef int hash
    cdef int hashInvalid
    
    def __init__(self):
        self.legalMoves = None
        # the caller of the constructor needs to always construct this directly
        # TODO figure out a less nasty way to handle this
        self.c6 = NULL
        self.initId()
        self.hash = 0
        self.hashInvalid = 1
    
    def initId(self):
        self.lastMove = -1
        self.lastId = 0
        self.id = 0
        self.nextId()
        self.hashInvalid = 1

    def nextId(self):
        global seed
        self.lastId = self.id
        self.id = seed
        seed += 1
        
    def __dealloc__(self):
        freeC6(self.c6)
    
    def __reduce__(self):
        return (Connect6State, (), picklePackC6(self.c6))
    
    def __setstate__(self, d):
        self.legalMoves = None
        self.c6 = pickleUnpackC6(d)
        self.initId()
    
    def __hash__(self):
        if self.hashInvalid:
            self.hash = hashC6(self.c6)
            self.hashInvalid = 0
        return self.hash
    
    def __eq__(self, Connect6State other):
        if other.c6.turn != self.c6.turn or other.c6.m != self.c6.m or other.c6.n != self.c6.n:
            return False
        return areFieldsEqual(self.c6.m, self.c6.n, self.c6.board, other.c6.board)
    
    def packageForDebug(self):
        package = {}
        package["m"] = self.c6.m
        package["n"] = self.c6.n
        package["turn"] = self.c6.turn
        package["onturn"] = getPlayerIndexOnTurnC6(self.c6)
        
        board = []
        
        for y in range(self.c6.n):
            l = []
            for x in range(self.c6.m):
                l.append(readField(self.c6.board, self.c6.m, x, y))
            board.append(l)
        
        package["board"] = board
        return package

    def getHumanMoveDescription(self, mkey):
        x, y = self.getMoveLocation(mkey)
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        return str(x+1) + "-" + chars[y] 
    
    def __str__(self):
        if self.lastMove != -1:
            lX, lY = self.getMoveLocation(self.lastMove)
        else:
            lX = -50
            lY = -50
        return toStringC6(self.c6, lX, lY)
    
    def moveProbsAndDisplay(self, mp):
        mm = ['.', '░', '█']
        
        if self.lastMove != -1:
            lastMoveX, lastMoveY = self.getMoveLocation(self.lastMove)
        else:
            lastMoveX = -50
            lastMoveY = -50
        
        m = self.c6.m
        n = self.c6.n
        c6 = self.c6
        s = "Connect6(%i,%i), " %  (c6.m, c6.n)
        if hasEndedC6(c6) == 0:
            s += "Turn %i: %s\n" % (c6.turn, mm[getPlayerIndexOnTurnC6(c6)+1])
        elif c6.winningPlayer > -1:
            s += "Winner: %s\n" % mm[c6.winningPlayer+1]
        else:
            s += "Draw\n"
        
        def getPDisplay(x, y):
            mkey = self.getMoveKey(x, y)
            m = mp[mkey]
            m *= 100
            if m > 0.1 and m < 1:
                m = 1
            stone = readField(self.c6.board, self.c6.m, x, y)+1
            if int(m) == 0 or stone > 0:
                if x == lastMoveX and y == lastMoveY:
                    return "│" + mm[stone] + "│ "
                else:
                    return " " + mm[stone] + "  "
            else:
                m = str(int(m))
                m = " " + m
                while len(m) < 4:
                    m += " "
                return m
        
        s += "   |"
        for x in range(m):
            if x < 9:
                s += " %i |" % (x+1)
            else:
                s += "%i |" % (x+1)
        
        s += "\n";
        
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        
        assert n <= len(chars)
        
        for y in range(n):
            for x in range(m+1):
                if x-1 == lastMoveX and y == lastMoveY:
                    s += "┌─┐ "
                elif x-1 == lastMoveX and y == lastMoveY+1:
                    s += "└─┘ "
                else:
                    s += "    "
            s += "\n";
            s += " " + chars[y] + "  ";
            for x in range(m):
                s += getPDisplay(x, y);
            s += "\n";
        for _ in range(m+1):
            s += "    ";
        s += "\n";
        
        return s;
    
    def getMoveLocation(self, key):
        y = int(key / self.c6.m)
        x = int(key % self.c6.m)
        return x, y
    
    def getMoveKey(self, x, y):
        return y * self.c6.m + x
    
    def getWinner(self):
        """
        return index of the winning player or -1 if a draw
        """
        return self.c6.winningPlayer
        
    def isMoveLegal(self, move):
        """
        return True if the given move is legal
        """
        cdef int x, y, key
        key = move
        x = key % self.c6.m
        y = key / self.c6.m
        return readField(self.c6.board, self.c6.m, x, y) == -1
        
    def getLegalMoves(self):
        """
        return a list of all indices of legal moves
        performance relevant. cache it hard
        """
        cdef int moveIdx
        if self.legalMoves is None:
            self.legalMoves = []
            for moveIdx in range(self.getMoveCount()):
                if self.isMoveLegal(moveIdx):
                    self.legalMoves.append(moveIdx)
        return self.legalMoves
        
    def getPlayerOnTurnIndex(self):
        """
        return index of player whose turn it is right now
        """
        return getPlayerIndexOnTurnC6(self.c6)
        
    def getTurn(self):
        """
        return the current turn, the first turn is turn 0
        """
        return self.c6.turn

    def isEarlyGame(self):
        """
        return if moves should be deterministic(False) or probabilistic (True)
        """
        cdef int c
        c = max(self.c6.m, self.c6.n)
        c += c % 2
        return self.c6.turn < c * 2
    
    def canTeachSomething(self):
        """
        returns True iff the learner may learn something from this state
        """
        return self.c6.turn > 0
    
    def getPlayerCount(self):
        """
        returns the number of players who play this game
        """
        return 2
    
    def hasDraws(self):
        return True
    
    def getMoveCount(self):
        """
        returns the number of moves a player can make, including currently illegal moves (TODO: why include that?)
        """
        return self.c6.m * self.c6.n
    
    def clone(self):
        """
        returns a deep copy of the state
        """
        c = Connect6State()
        c.c6 = cloneC6(self.c6)
        c.id = self.id
        return c
    
    def getFrameClone(self):
        """
        returns a copy that will be used as data for frames. Can probably save some memory. The returned object
        strictly only needs to implement mapPlayerIndexToTurnRel (TODO that should not be necessary...)
        and be useful to the fillNetworkInput implementation
        """
        
        c = Connect6State()
        c.c6 = cloneC6(self.c6)
        c.id = self.id
        return c
    
    def getNewGame(self):
        """
        expected to return a new state that represents a newly initialized game
        """
        c = Connect6State()
        c.c6 = initC6(self.c6.m, self.c6.n)
        return c
    
    def augmentFrame(self, frame):
        """
        given a frame (state, movedistribution, winchances) return a copy of the frame augmented for training.
        For example apply random rotations or mirror it
        """
        cdef Connect6State cstSrc = frame[0]
        cdef Connect6State cst = cstSrc.clone();
        frame = [cst, list(frame[1]), frame[2], list(frame[3])]
        
        fState = cst.c6
        fMoves = frame[1]

        augmentFieldAndMovesDistribution(fState.m, fState.n, fState.board, fMoves, 
                                         lambda idx: self.getMoveLocation(idx),
                                         lambda x,y: self.getMoveKey(x, y))

        return frame
    
    def simulate(self, move):
        """
        Do one step of the simulation given a move for the current player.
        Mutate this object.
        """
        cdef int x, y, key
        self.legalMoves = None
        key = move
        x = key % self.c6.m
        y = key / self.c6.m
        placeC6(self.c6, x, y)
        
        self.lastMove = move
        self.nextId()
        self.hashInvalid = 1
    
    def updateTensorForLastMove(self, object tensor, int batchIndex):
        assert False, "this is not used, right?"
        cdef int x, y, b
        cdef Connect6_c* c6 = self.c6
        
        x = self.lastMove % self.c6.m
        y = self.lastMove / self.c6.m
        
        b = readField(c6.board, c6.m, x, y)
        if b != -1:
            b = self.mapPlayerIndexToTurnRel(b)
        tensor[batchIndex, 0, y, x] = b
    
    def invertTensorField(self, object tensor, batchIndex):
        """
        inverts the field of the previous state in case this is the first move of the current player
        inversion means:
        -1 stays -1
        0 turns into 1
        1 turns into 0
        if this is the 2nd move of the current player, do nothing
        
        invert for turns:
        1, 3, 5, 7, 9, ....
        -> uneven turns
        """
        assert False, "this is not used, right?"
        if self.getTurn() % 2 == 1:
            mask = tensor[batchIndex] == -1
            tensor[batchIndex] += 1
            tensor[batchIndex] %= 2
            tensor[batchIndex].masked_fill_(mask, -1)
    
    # TODO this is garbage and should not be used _eq_ is implemented now
    def isEqual(self, other):
        """
        returns if this state is equal to the given other state. 
        Used to generate statistics about the diversity of encountered states
        """
        assert False, "since when am I using this?"
    
    def isTerminal(self):
        """
        return true iff this state is terminal, i.e. additional moves are not allowed
        """
        return hasEndedC6(self.c6) != 0
    
    cpdef int mapPlayerIndexToTurnRel(self, int playerIndex):
        """
        return playerIndex -> playerIndex relative to current turn 
        """
        
        # draw index is a fixed point
        if playerIndex == 2:
            return 2
        
        onTurnIdx = getPlayerIndexOnTurnC6(self.c6)
        if onTurnIdx == playerIndex:
            return 0
        else:
            return 1

def stateFormat(state):
    return str(state)

cdef void fillNetworkInput0(Connect6State state, float[:,:,:,:] tensor, int batchIndex):
    cdef int x, y, b
    cdef Connect6_c* c6 = state.c6

    cdef float evenTurn = -1
    
    if state.getTurn() % 2 == 0:
        evenTurn = 1
    
    for y in range(c6.n):
        for x in range(c6.m):
            b = readField(c6.board, c6.m, x, y)
            if b != -1:
                b = state.mapPlayerIndexToTurnRel(b)
            tensor[batchIndex,0,y,x] = b
            tensor[batchIndex,1,y,x] = evenTurn

class Connect6Init():
    def __init__(self):
        self.stateFormat = stateFormat
        
    def setConfig(self, config):
        self.config = config
        gconf = self.config["game"]
        self.m = gconf["m"]
        self.n = gconf["n"]
        
    def getStateTemplate(self):
        c = Connect6State()
        c.c6 = initC6(self.m, self.n)
        return c
    
    def getPlayerCount(self):
        return 2
    
    def hasDraws(self):
        return True
    
    def getMoveCount(self):
        return self.m * self.n
    
    def getGameDimensions(self):
        return [self.m, self.n, 2]
    
    def fillNetworkInput(self, Connect6State state, object torchTensor, int batchIndex):
        cdef float [:, :, :, :] tensor = torchTensor.numpy()
        fillNetworkInput0(state, tensor, batchIndex)

    def mkParseCommand(self):
        m = self.m
        n = self.n
        def p(cmd):
            result = -1
            try:
                ms = cmd.split("-")
                if len(ms) == 2:
                    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
                    cIdx = chars.find(ms[1]);
                    if cIdx != -1:
                        ms[1] = cIdx + 1
                    x = int(ms[0]) - 1
                    y = int(ms[1]) - 1
                    result = y * m + x
            except:
                pass
            return result
        
        return p


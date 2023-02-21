#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------
"""
COSC 4550-COSC5550 - Introduction to AI - AI-Tournament 
"""
# Tac-Tac-Tical
# This program is designed to play Tic-Tac-Tical, using lookahead and board heuristics.
# It will allow the user to play a game against the machine, or allow the machine
# to play against itself for purposes of learning to improve its play.  All 'learning'
# code has been removed from this program.
#
# Tic-Tac-Tical is a 2-player game played on a grid. Each player has the same number
# of tokens distributed on the grid in an initial configuration.  On each turn, a player
# may move one of his/her tokens one unit either horizontally or vertically (not
# diagonally) into an unoccupied square.  The objective is to be the first player to get
# three tokens in a row, either horizontally, vertically, or diagonally.
#
# The board is represented by a matrix with extra rows and columns forming a
# boundary to the playing grid. Squares in the playing grid can be occupied by
# either 'X', 'O', or 'Empty' spaces.  The extra elements are filled with 'Out of Bounds'
# squares, which makes some of the computations simpler.
#-------------------------------------------------------------------------

from __future__ import print_function
import random
from random import randrange
import copy
import numpy as np
import time
# import numba


# @numba.jit(nopython=True)
def GetNumberOfPossibleMoves(Player, Board):
    number_of_moves = 0
    for i in range(1, NumRows + 1):
        for j in range(1, NumCols + 1):
            if Board[i][j] == Player:
                # -------------------------------------------------------------
                #  Check move directions (m,n) = (-1,0), (0,-1), (0,1), (1,0)
                # -------------------------------------------------------------
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if abs(m) != abs(n):
                            if Board[i + m][j + n] == Empty:
                                number_of_moves = number_of_moves + 1
    return number_of_moves

# @numba.jit(nopython=True)
def GetMoves (Player, Board, MoveList):
#-------------------------------------------------------------------------
# Determines all legal moves for Player with current Board,
# and returns them in MoveList.
#-------------------------------------------------------------------------
    number_of_moves = 0
    for i in range(1,NumRows+1):
        for j in range(1,NumCols+1):
            if Board[i][j] == Player:
            #-------------------------------------------------------------
            #  Check move directions (m,n) = (-1,0), (0,-1), (0,1), (1,0)
            #-------------------------------------------------------------
                for m in range(-1,2):
                    for n in range(-1,2):
                        if abs(m) != abs(n):
                            if Board[i + m][j + n] == Empty:
                                # move = np.array([i, j, i + m, j + n], dtype=int)
                                MoveList[number_of_moves] = [i, j, i + m, j + n]
                                number_of_moves = number_of_moves + 1
    return MoveList


def GetHumanMove (Player, Board):
#-------------------------------------------------------------------------
# If the opponent is a human, the user is prompted to input a legal move.
# Determine the set of all legal moves, then check input move against it.
#-------------------------------------------------------------------------
    MoveList = GetMoves(Player, Board)
    Move = None

    while(True):
        FromRow, FromCol, ToRow, ToCol = map(int, \
            input('Input your move (FromRow, FromCol, ToRow, ToCol): ').split(' '))

        ValidMove = False
        if not ValidMove:
            for move in MoveList:
                if move == [FromRow, FromCol, ToRow, ToCol]:
                    ValidMove = True
                    Move = move

        if ValidMove:
            break

        print('Invalid move.  ')

    return Move


# @numba.jit(nopython=True)
def ApplyMove (Board, Move):
#-------------------------------------------------------------------------
# Perform the given move, and update Board.
#-------------------------------------------------------------------------

    FromRow = Move[0]
    FromCol = Move[1]
    ToRow = Move[2]
    ToCol = Move[3]
    newBoard = np.copy(Board)
    newBoard[ToRow][ToCol] = Board[FromRow][FromCol]
    newBoard[FromRow][FromCol] = Empty
    return newBoard


def InitBoard (Board):
#-------------------------------------------------------------------------
# Initialize the game board.
#-------------------------------------------------------------------------

    for i in range(0,BoardRows+1):
        for j in range(0,BoardCols+1):
            Board[i][j] = OutOfBounds

    for i in range(1,NumRows+1):
        for j in range(1,NumCols+1):
            Board[i][j] = Empty

    for j in range(1,NumCols+1):
        if odd(j):
            Board[1][j] = x
            Board[NumRows][j] = o
        else:
            Board[1][j] = o
            Board[NumRows][j] = x


def odd(n):
    return n%2==1


def ShowBoard (Board):
    print("")
    row_divider = "+" + "-"*(NumCols*4-1) + "+"
    print(row_divider)

    for i in range(1,NumRows+1):
        for j in range(1,NumCols+1):
            if Board[i][j] == x:
                print('| X ',end="")
            elif Board[i][j] == o:
                print('| O ',end="")
            elif Board[i][j] == Empty:
                print('|   ',end="")
        print('|')
        print(row_divider)

    print("")


# @numba.jit(nopython=True)
def Win (Player, Board):
#-------------------------------------------------------------------------
# Determines if Player has won, by finding '3 in a row'.
#-------------------------------------------------------------------------
    count = 0
    # checking all rows
    for i in range(1,NumRows+1):
        count = 0
        for j in range(1,NumCols+1):
            if Board[i][j] == Player:
                count += 1
            else:
                count = 0
            if count == 3:
                return True

    # checking all columns
    count = 0
    for j in range(1,NumCols+1):
        count = 0
        for i in range(1,NumRows+1):
            if Board[i][j] == Player:
                count += 1
            else:
                count = 0
            if count == 3:
                return True

    # checking diagonals
    for i in range(1,6):
        for j in range(1,5):
            if Board[i][j] == Player:
                if Board[i+1][j+1] == Player:
                    if Board[i+2][j+2] == Player:
                        return True
            if Board[i][j] == Player:
                if Board[i-1][j+1] == Player:
                    if Board[i-2][j+2] == Player:
                        return True

    return False

# @numba.jit(nopython=True)
def DetermineTurns(Player):
    if Player == 1:
        return -1
    else:
        return 1


def minimax_with_alphaBeta(Player, InitialBoard, depth, maxDepth, number_of_explored_states, alpha, beta):


    # check initial state of the board
    isTerminal = Win(Player, InitialBoard)
    if isTerminal:
        if Player == -1:
            return -10/depth, number_of_explored_states
        if Player == 1:
            return 10/depth, number_of_explored_states

    # maximize or minimize
    else:
        if Player == 1:
            bestScore = -infinity
            N = GetNumberOfPossibleMoves(Player, InitialBoard)
            MoveList = np.zeros((N, 4), dtype=int)
            MoveList = GetMoves(Player, InitialBoard, MoveList)
            for move in MoveList:
                newBoard = ApplyMove(InitialBoard, move)
                number_of_explored_states += 1
                if depth<=maxDepth:
                    score, number_of_explored_states = minimax_with_alphaBeta(-1, newBoard, depth+1, maxDepth, number_of_explored_states, alpha, beta)
                else:
                    score = 0
                bestScore = max(score, bestScore)
                alpha = max(bestScore, alpha)
                if alpha >= beta:
                    break

            return [bestScore, number_of_explored_states]

        elif Player == -1:
            bestScore = infinity
            N = GetNumberOfPossibleMoves(Player, InitialBoard)
            MoveList = np.zeros((N, 4), dtype=int)
            MoveList = GetMoves(Player, InitialBoard, MoveList)
            for move in MoveList:
                newBoard = ApplyMove(InitialBoard, move)
                number_of_explored_states += 1
                if depth <= maxDepth:
                    score, number_of_explored_states = minimax_with_alphaBeta(1, newBoard, depth+1, maxDepth, number_of_explored_states, alpha, beta)
                else:
                    score = 0
                bestScore = min(score, bestScore)
                beta = min(bestScore, beta)
                if alpha>=beta:
                    break

            return [bestScore, number_of_explored_states]


def GetComputerMove (Player, Board):
#-------------------------------------------------------------------------
# If the opponent is a computer, use artificial intelligence to select
# the best move.
# For this demo, a move is chosen at random from the list of legal moves.
# You need to write your own code to get the best computer move.
#-------------------------------------------------------------------------
    import numpy as np
    # import numba

    # I am defining these variables here so that they will not be mixed up with global variables
    x = -1
    o = 1
    Empty = 0
    OutOfBounds = 2
    NumRows = 5
    BoardRows = NumRows + 1
    NumCols = 4
    BoardCols = NumCols + 1
    MaxMoves = 4*NumCols
    NumInPackedBoard = 4 * (BoardRows+1) *(BoardCols+1)
    infinity = 10000  # Value of a winning board

    Board  = np.array(Board)

    # I am defining these functions here so that these specific functions will not be overwritten
    # when imported inside tournament.py
    # I have modified these functions for performance boost

    # @numba.jit(nopython=True)
    def GetNumberOfPossibleMoves(Player, Board):
        number_of_moves = 0
        for i in range(1, NumRows + 1):
            for j in range(1, NumCols + 1):
                if Board[i][j] == Player:
                    # -------------------------------------------------------------
                    #  Check move directions (m,n) = (-1,0), (0,-1), (0,1), (1,0)
                    # -------------------------------------------------------------
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            if abs(m) != abs(n):
                                if Board[i + m][j + n] == Empty:
                                    number_of_moves = number_of_moves + 1
        return number_of_moves

    # @numba.jit(nopython=True)
    def GetMoves (Player, Board, MoveList):
    #-------------------------------------------------------------------------
    # Determines all legal moves for Player with current Board,
    # and returns them in MoveList.
    #-------------------------------------------------------------------------
        number_of_moves = 0
        for i in range(1,NumRows+1):
            for j in range(1,NumCols+1):
                if Board[i][j] == Player:
                #-------------------------------------------------------------
                #  Check move directions (m,n) = (-1,0), (0,-1), (0,1), (1,0)
                #-------------------------------------------------------------
                    for m in range(-1,2):
                        for n in range(-1,2):
                            if abs(m) != abs(n):
                                if Board[i + m][j + n] == Empty:
                                    # move = np.array([i, j, i + m, j + n], dtype=int)
                                    MoveList[number_of_moves] = [i, j, i + m, j + n]
                                    number_of_moves = number_of_moves + 1
        return MoveList


    # @numba.jit(nopython=True)
    def ApplyMove (Board, Move):
    #-------------------------------------------------------------------------
    # Perform the given move, and update Board.
    #-------------------------------------------------------------------------

        FromRow = Move[0]
        FromCol = Move[1]
        ToRow = Move[2]
        ToCol = Move[3]
        newBoard = np.copy(Board)
        newBoard[ToRow][ToCol] = Board[FromRow][FromCol]
        newBoard[FromRow][FromCol] = Empty
        return newBoard


    # @numba.jit(nopython=True)
    def Win (Player, Board):
    #-------------------------------------------------------------------------
    # Determines if Player has won, by finding '3 in a row'.
    #-------------------------------------------------------------------------
        count = 0
        # checking all rows
        for i in range(1,NumRows+1):
            count = 0
            for j in range(1,NumCols+1):
                if Board[i][j] == Player:
                    count += 1
                else:
                    count = 0
                if count == 3:
                    return True

        # checking all columns
        count = 0
        for j in range(1,NumCols+1):
            count = 0
            for i in range(1,NumRows+1):
                if Board[i][j] == Player:
                    count += 1
                else:
                    count = 0
                if count == 3:
                    return True

        # checking diagonals
        for i in range(1,6):
            for j in range(1,5):
                if Board[i][j] == Player:
                    if Board[i+1][j+1] == Player:
                        if Board[i+2][j+2] == Player:
                            return True
                if Board[i][j] == Player:
                    if Board[i-1][j+1] == Player:
                        if Board[i-2][j+2] == Player:
                            return True

        return False

    # @numba.jit(nopython=True)
    def DetermineTurns(Player):
        if Player == 1:
            return -1
        else:
            return 1


    def minimax_with_alphaBeta(Player, InitialBoard, depth, maxDepth, number_of_explored_states, alpha, beta):


        # check initial state of the board
        isTerminal = Win(Player, InitialBoard)
        if isTerminal:
            if Player == -1:
                return -10/depth, number_of_explored_states
            if Player == 1:
                return 10/depth, number_of_explored_states

        # maximize or minimize
        else:
            if Player == 1:
                bestScore = -infinity
                N = GetNumberOfPossibleMoves(Player, InitialBoard)
                MoveList = np.zeros((N, 4), dtype=int)
                MoveList = GetMoves(Player, InitialBoard, MoveList)
                for move in MoveList:
                    newBoard = ApplyMove(InitialBoard, move)
                    number_of_explored_states += 1
                    if depth<=maxDepth:
                        score, number_of_explored_states = minimax_with_alphaBeta(-1, newBoard, depth+1, maxDepth, number_of_explored_states, alpha, beta)
                    else:
                        score = 0
                    bestScore = max(score, bestScore)
                    alpha = max(bestScore, alpha)
                    if alpha >= beta:
                        break

                return [bestScore, number_of_explored_states]

            elif Player == -1:
                bestScore = infinity
                N = GetNumberOfPossibleMoves(Player, InitialBoard)
                MoveList = np.zeros((N, 4), dtype=int)
                MoveList = GetMoves(Player, InitialBoard, MoveList)
                for move in MoveList:
                    newBoard = ApplyMove(InitialBoard, move)
                    number_of_explored_states += 1
                    if depth <= maxDepth:
                        score, number_of_explored_states = minimax_with_alphaBeta(1, newBoard, depth+1, maxDepth, number_of_explored_states, alpha, beta)
                    else:
                        score = 0
                    bestScore = min(score, bestScore)
                    beta = min(bestScore, beta)
                    if alpha>=beta:
                        break

                return [bestScore, number_of_explored_states]


    st = time.time()
    Board_ = np.copy(Board)
    maxDepth = 10

    N = GetNumberOfPossibleMoves(Player, Board)
    MoveList = np.zeros((N, 4), dtype=int)
    MoveList = GetMoves(Player, Board, MoveList)

    bestScores = []
    for move in MoveList:
        newBoard = ApplyMove(Board_, move)
        newBoard_ = np.copy(newBoard)
        Player_ = DetermineTurns(Player)
        move_score, number_of_explored_states = minimax_with_alphaBeta(Player_, newBoard_, depth = 1, maxDepth=maxDepth,
                                                                       number_of_explored_states = 0, alpha = -infinity,
                                                                       beta=infinity)
        bestScores.append(move_score)

    nd = time.time()
    print('nrustamo played in ', round(nd - st, 5), ' seconds and explored ', number_of_explored_states, ' states')
    bestScores = np.array(bestScores)
    if Player == 1:
        idx = np.argmax(bestScores, axis=0)
        return MoveList[idx]
    else:
        idx = np.argmin(bestScores, axis=0)
        return MoveList[idx]


if __name__ == "__main__":
#-------------------------------------------------------------------------
# A move is represented by a list of 4 elements, representing 2 pairs of
# coordinates, (FromRow, FromCol) and (ToRow, ToCol), which represent the
# positions of the piece to be moved, before and after the move.
#-------------------------------------------------------------------------
    x = -1
    o = 1
    Empty = 0
    OutOfBounds = 2
    NumRows = 5
    BoardRows = NumRows + 1
    NumCols = 4
    BoardCols = NumCols + 1
    MaxMoves = 4*NumCols
    NumInPackedBoard = 4 * (BoardRows+1) *(BoardCols+1)
    infinity = 10000  # Value of a winning board
    Board = [[0 for col in range(BoardCols+1)] for row in range(BoardRows+1)]
    Board = np.array(Board)
    print("\nThe squares of the board are numbered by row and column, with '1 1' ")
    print("in the upper left corner, '1 2' directly to the right of '1 1', etc.")
    print("")
    print("Moves are of the form 'i j m n', where (i,j) is a square occupied")
    print("by your piece, and (m,n) is the square to which you move it.")
    print("")
    print("You move the 'X' pieces.\n")

    InitBoard(Board)
    ShowBoard(Board)
    """
    MoveList = GetMoves(x,Board)
    print(MoveList)
    MoveList = GetMoves(o,Board)
    print(MoveList)   
    """
    N = GetNumberOfPossibleMoves(x, Board)
    MoveList = np.zeros((N, 4), dtype = int)
    MoveList = GetMoves(x, Board, MoveList)

    while True:
        Move = GetComputerMove(x,Board)
        Board = ApplyMove(Board,Move)
        ShowBoard(Board)
        if Win(x, Board):
            print("x won this game")
            break
    

        Move = GetComputerMove(o,Board)
        Board = ApplyMove(Board,Move)
        ShowBoard(Board)
        if Win(o, Board):
            print("o won this game")
            break            


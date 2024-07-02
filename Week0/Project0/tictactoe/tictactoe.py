"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None
BOARD_SIZE = 3


class InvalidActionError(Exception):
    def __init__(self):
        super().__init__("Invalid action")


def initial_state():
    """
    Returns starting state of the board.
    """

    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    o_count = sum([row.count(O) for row in board])
    x_count = sum([row.count(X) for row in board])

    return X if o_count >= x_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    possible = set()
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == EMPTY:
                possible.add((i, j))

    return possible


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    if action not in actions(board):
        raise InvalidActionError

    new_board = [row.copy() for row in board]
    new_board[action[0]][action[1]] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    def check(option, player):
        for cell in option:
            if cell != player:
                return False

        return True

    possible = [row for row in board] + [list(col) for col in zip(*board)]
    possible += [[board[i][i]
                 for i in range(BOARD_SIZE)]] + [[board[i][BOARD_SIZE-1-i] for i in range(BOARD_SIZE)]]

    for player in (X, O):
        for option in possible:
            if check(option, player):
                return player

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    empty_count = sum([row.count(EMPTY) for row in board])
    if empty_count != 0:
        return winner(board) is not None

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    result = winner(board)

    if result == X:
        return 1
    elif result == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    def minimize(board):
        v = math.inf
        if terminal(board):
            return utility(board), None

        for action in actions(board):
            new_val, _ = maximize(result(board, action))
            if new_val < v:
                v = new_val
                picked = action

                if v == -1:
                    return v, picked

        return v, picked

    def maximize(board):
        v = -math.inf
        if terminal(board):
            return utility(board), None

        for action in actions(board):
            new_val, _ = minimize(result(board, action))
            if new_val > v:
                v = new_val
                picked = action

                if v == 1:
                    return v, picked

        return v, picked

    if player(board) == X:
        _, picked = maximize(board)
    else:
        _, picked = minimize(board)

    return picked

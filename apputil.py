import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(current_board):
    """
    This function update_board takes the current state of a Conway's Game of Life 
    board as input and returns the updated state of the board after applying the 
    rules of the game. It performs just one iteration of the game , and takes in a 
    numpy array that represents the current board where the cells are live (1) or 
    dead (0). The function calculates the number of live neighbors for each cell and 
    applies the rules of Conway's Game of Life to determine the next state of each cell.
    Lastly, it returns the updated board as a numpy array.
    """
    # Pads the current board with a border of dead cells
    padded_board = np.pad(current_board, pad_width=1, mode='constant', constant_values=0)

    # Calculates the number of live neighbors for each cell
    neighbors = (
        padded_board[0:-2, 0:-2] + padded_board[0:-2, 1:-1] + padded_board[0:-2, 2:] +
        padded_board[1:-1, 0:-2] + padded_board[1:-1, 2:] + padded_board[2:, 0:-2] + padded_board[2:, 1:-1] + padded_board[2:, 2:]
    )

    # Identify the live cells
    live = (current_board == 1)

    # Create an updated board initialized to all dead cells
    updated_board = np.zeros_like(current_board)

    # The live cell stays live if it has two or three live neighbors
    updated_board[np.where((live) & ((neighbors == 2) | (neighbors == 3)))] = 1

    # The dead cell then becomes live if it has exactly three live neighbors
    updated_board[np.where((~live) & (neighbors == 3))] = 1

    # Returns the updated board
    return updated_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)
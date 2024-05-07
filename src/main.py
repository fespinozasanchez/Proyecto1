# Proyecto#1: Robotica_INFO1167
# Felipe Espinoza - Oscar Uribe

import pygame
import pygame_menu
import numpy as np
import random
import matplotlib.pyplot as plt


def get_map() -> list:
    """
    The function `get_map` returns a list of lists representing a map with certain values.
    :return: The function `get_map()` returns a list of lists representing a 2D map. Each inner list
    corresponds to a row in the map, and the numbers within the inner lists represent different
    locations or tiles on the map. The numbers -2 indicate empty spaces or obstacles on the map.
    """
    return [
        [0,  1,  2,  3,  4,  5,  6, -2,  -2],
        [7, -2, -2,  8, -2, -2,  9, -2,  -2],
        [10, 11, -2, 12, -2, -2, 13, -2, -2],
        [14, 15, 16, 17, 18, 19, 20, 21, 22],
        [23, 24, -2, 25, 26, -2, 27, -2, -2],
        [28, -2, -2, 29, -2, -2, 30, -2, -2],
        [31, 32, 33, 34, -2, -2, -2, -2, -2]
    ]


def norma_sp(aA, aB, nError=0) -> bool:
    """
    The function `norma_sp` compares two arrays `aA` and `aB` element-wise and returns True if the
    maximum absolute difference between corresponding elements is greater than a specified error
    threshold `nError`.
    
    :param aA: The parameter `aA` represents a list of numerical values
    :param aB: It seems like you were about to provide some information about the `aB` parameter, but
    the message got cut off. Could you please provide more details or let me know how I can assist you
    further?
    :param nError: The `nError` parameter in the `norma_sp` function represents the maximum allowable
    difference between the elements of two arrays `aA` and `aB`. If the maximum difference between
    corresponding elements of `aA` and `aB` is greater than `nError`, the function, defaults to 0
    (optional)
    :return: The function `norma_sp` is returning a boolean value. It calculates the maximum absolute
    difference between corresponding elements of two input lists `aA` and `aB`, and compares it to a
    specified error threshold `nError`. If the maximum absolute difference is greater than the error
    threshold, the function returns `True`, indicating that the condition is met. Otherwise, it returns
    `False`.
    """
    return (max([abs(aA[i]-aB[i]) for i in range(len(aA))]) > nError)


def nReward(state: int, mState: int) -> int:
    """
    The function `nReward` returns different reward values based on the input `state` compared to
    `mState` and specific conditions.
    
    :param state: The `state` parameter in the `nReward` function represents the current state of a
    system or environment. The function evaluates the `state` value against certain conditions and
    returns a reward value based on those conditions
    :type state: int
    :param mState: mState is a parameter representing a specific state in a system or environment. In
    the given function nReward, if the current state matches the mState parameter, a reward of 100 is
    returned
    :type mState: int
    :return: The function `nReward` returns an integer value based on the input `state`. The return
    value depends on the conditions specified in the function:
    - If `state` is equal to `mState`, it returns 100.
    - If `state` is equal to 14 or 17, it returns -100.
    - If `state` is equal to -2, it returns -50
    """
    if state == mState:
        return 100
    elif state == 14 or state == 17:
        return -100
    elif state == -2:
        return -50
    else:
        return -1

# -----------------mTransition--------------------


def pos_to_index(row: int, col: int, nCols: int) -> int:
    """
    The function `pos_to_index` calculates the index of a 2D array element given its row and column
    position and the number of columns in the array.
    
    :param row: The `row` parameter represents the row number in a grid
    :type row: int
    :param col: The `col` parameter in the `pos_to_index` function represents the column number of a 2D
    grid. It is used to calculate the index of a specific cell in a 1D list representation of the grid
    :type col: int
    :param nCols: The `nCols` parameter represents the number of columns in a grid or matrix. It is used
    in the `pos_to_index` function to calculate the index of a specific cell in a 2D grid based on its
    row and column coordinates
    :type nCols: int
    :return: The function `pos_to_index` takes in three parameters: `row`, `col`, and `nCols`. It
    calculates the index of a 2D array element based on the row and column provided, using the formula
    `row * nCols + col`. The calculated index is then returned by the function.
    """
    return row * nCols + col


def index_to_pos(index: int, nCols: int) -> int:
    """
    The function `index_to_pos` takes an index and the number of columns in a grid and returns the
    corresponding row and column position.
    
    :param index: The `index` parameter represents the position in a one-dimensional array, and `nCols`
    represents the number of columns in a two-dimensional grid
    :type index: int
    :param nCols: The `nCols` parameter represents the number of columns in a grid or matrix. It is used
    in the `index_to_pos` function to calculate the row and column position of a given index within the
    grid
    :type nCols: int
    :return: The function `index_to_pos` returns a tuple containing the row and column position
    corresponding to the given index within a grid with a specified number of columns.
    """
    return (index // nCols, index % nCols)


def mTransition(aMap: list, action: str, nStates: int, nRows: int, nCols: int, p_success: float = 0.9, p_side: float = 0.05) -> np.ndarray:
    """
    The function `mTransition` calculates the transition matrix for a given action in a grid environment
    with specified probabilities of success and failure.
    
    :param aMap: It seems like the description of the `aMap` parameter is missing. Could you please
    provide more information about what `aMap` represents or what kind of data it contains?
    :type aMap: list
    :param action: The `action` parameter in the `mTransition` function represents the direction in
    which the transition is being made. It can take values such as 'N' for North, 'S' for South, 'E' for
    East, and 'W' for West. This parameter determines the movement direction
    :type action: str
    :param nStates: The `nStates` parameter represents the total number of states in your system. It is
    used to initialize a square matrix `T` with dimensions `nStates x nStates` to represent the
    transition probabilities between states
    :type nStates: int
    :param nRows: The `nRows` parameter in the `mTransition` function represents the number of rows in
    the grid or map where the agent is moving. It is used to determine the size of the grid and to
    iterate over each row when calculating the transition probabilities for the given action
    :type nRows: int
    :param nCols: The `nCols` parameter in the `mTransition` function represents the number of columns
    in the grid or map where the agent is moving. It is used to determine the size of the grid and
    calculate the transition probabilities for each possible action in the environment
    :type nCols: int
    :param p_success: The parameter `p_success` in the `mTransition` function represents the probability
    of successfully moving to the main position (main_pos) when taking a specific action (North, South,
    East, West) in a grid environment. This probability is used to calculate the transition
    probabilities in the transition matrix `
    :type p_success: float
    :param p_side: The `p_side` parameter in the `mTransition` function represents the probability of
    transitioning to one of the side positions (side1_pos and side2_pos) when taking a specific action
    (North, South, East, West) in a grid environment. This probability is used when calculating the
    transition
    :type p_side: float
    :return: The function `mTransition` is returning a numpy array `T` which represents the transition
    matrix for a given action in a grid environment. The transition matrix `T` contains the
    probabilities of transitioning from one state to another state based on the specified action and
    probabilities of success and side movements.
    """

    T = np.zeros((nStates, nStates))  # Matriz de transicion initializada en 0

    for row in range(nRows):
        for col in range(nCols):
            # Muralla o bloque invisible
            if aMap[row][col] == -2 or aMap[row][col] == 14 or aMap[row][col] == 17:
                continue
            current_index = pos_to_index(row, col, nCols)

            # Determinar las posiciones objetivo para cada accion
            if action == 'N':
                main_pos = (row - 1, col)
                side1_pos = (row, col - 1)
                side2_pos = (row, col + 1)
            elif action == 'S':
                main_pos = (row + 1, col)
                side1_pos = (row, col - 1)
                side2_pos = (row, col + 1)
            elif action == 'E':
                main_pos = (row, col + 1)
                side1_pos = (row - 1, col)
                side2_pos = (row + 1, col)
            elif action == 'W':
                main_pos = (row, col - 1)
                side1_pos = (row - 1, col)
                side2_pos = (row + 1, col)

            # Aplicar las probabilidades
            for pos, prob in [(main_pos, p_success), (side1_pos, p_side), (side2_pos, p_side)]:
                r, c = pos
                if r < 0 or r >= nRows or c < 0 or c >= nCols or aMap[r][c] < 0:
                    # Se queda en el mismo lugar
                    T[current_index, current_index] += prob
                else:
                    target_index = pos_to_index(r, c, nCols)
                    T[current_index, target_index] += prob

    return T


# ---------------ALGORITMOS-------------------

def ValueIteration(nIteration: int, vFunction: list, nStates: int, nActions: int, aTransition: list, QValue: list, aPoliticas: list, nErr: float, sMeta: int, ld: float = 1) -> np.ndarray:
    """
    This Python function implements the Value Iteration algorithm for solving a Markov Decision Process
    to find optimal policies.
    
    :param nIteration: The `nIteration` parameter in the `ValueIteration` function represents the number
    of iterations to perform in the value iteration algorithm. It determines how many times the
    algorithm will update the value function and policy estimates before terminating and returning the
    final results
    :type nIteration: int
    :param vFunction: The `vFunction` parameter is a list that stores the value function for each
    iteration during the value iteration process. It contains the value function for each state in the
    Markov Decision Process (MDP) at each iteration
    :type vFunction: list
    :param nStates: The `nStates` parameter in the `ValueIteration` function represents the total number
    of states in the Markov Decision Process (MDP) being analyzed. It is used to determine the size of
    various data structures and to iterate over all states during the value iteration process
    :type nStates: int
    :param nActions: The `nActions` parameter in the `ValueIteration` function represents the number of
    possible actions that can be taken in each state of the Markov Decision Process (MDP). It is used to
    determine the range of the action loop in the function where the algorithm calculates the Q-values
    for each state
    :type nActions: int
    :param aTransition: The `aTransition` parameter in the `ValueIteration` function seems to represent
    the transition probabilities from one state to another given an action. It is a 3-dimensional list
    where `aTransition[a][s][i]` represents the probability of transitioning from state `s` to state `i
    :type aTransition: list
    :param QValue: The `QValue` parameter in the `ValueIteration` function seems to be a list that
    stores the Q-values for each state-action pair in a Markov Decision Process (MDP). The Q-value
    represents the expected cumulative reward of taking action `a` in state `s` and following a
    :type QValue: list
    :param aPoliticas: The `aPoliticas` parameter in the `ValueIteration` function seems to represent
    the policy for each state in the Markov Decision Process (MDP). It is a list that stores the chosen
    action for each state based on the Q-values calculated during the value iteration process
    :type aPoliticas: list
    :param nErr: The `nErr` parameter in the `ValueIteration` function represents the threshold for the
    convergence criteria. It is used to determine when the algorithm should stop iterating based on the
    difference between the value functions of consecutive iterations. The function will continue
    iterating until the norm of the difference between the value functions of
    :type nErr: float
    :param sMeta: The parameter `sMeta` in the `ValueIteration` function represents the goal state or
    target state in the Markov Decision Process (MDP) for which the algorithm is trying to find an
    optimal policy. It is the state at which the agent aims to arrive or the state that signifies the
    completion
    :type sMeta: int
    :param ld: The parameter `ld` in the `ValueIteration` function seems to represent the discount
    factor used in the Bellman equation for reinforcement learning. It is typically denoted as the Greek
    letter lambda (λ) and is used to discount future rewards in the calculation of the Q-values,
    defaults to 1
    :type ld: float (optional)
    :return: The function `ValueIteration` returns the number of iterations `nK` and the policy
    `aPoliticas` reshaped as a 7x9 array.
    """
    nK = 1
    while nK < nIteration:
        vFunction.append(np.zeros((1, nStates)))

        for s in range(0, nStates):
            for a in range(0, nActions):
                if s in [14, 17, -2]:
                    continue
                aAux = [aTransition[a][s][i] * vFunction[nK-1][0][i]
                        for i in range(0, nStates)]
                QValue[s][a] = nReward(s, sMeta) + ld*sum(aAux)
            vFunction[nK][0][s] = max(QValue[s][:])
            aPoliticas[0][s] = np.argmax(QValue[s][:])

        if not norma_sp(vFunction[nK][0][:], vFunction[nK-1][0][:], nErr):
            break
        nK += 1

    # print(f"Iteracion: {nK} - Politica:\n {aPoliticas[0][:].reshape(7, 9)}")
    return nK, aPoliticas[0][:].reshape(7, 9)


def GaussSeidel(nIteration: int, nStates: int, nActions: int, aTransition: list, aPoliticas: list,  sMeta: int, threshold: float = 0.01, ld: float = 1) -> tuple:
    """
    The function `GaussSeidel` implements the Gauss-Seidel algorithm for policy iteration in a Markov
    Decision Process.
    
    :param nIteration: The `nIteration` parameter in the `GaussSeidel` function represents the maximum
    number of iterations that will be performed during the Gauss-Seidel iteration process. It controls
    how many times the algorithm will update the values of the states based on the Bellman equation
    before checking for convergence
    :type nIteration: int
    :param nStates: The `nStates` parameter in the `GaussSeidel` function represents the total number of
    states in the system or environment. It is used to determine the size of the value function array
    `V` and to iterate over all states during the value iteration process
    :type nStates: int
    :param nActions: The `nActions` parameter in the `GaussSeidel` function represents the number of
    possible actions that can be taken in the given problem or environment. It is used to determine the
    range for iterating over actions when calculating the value of each state based on the transition
    probabilities and rewards associated with those
    :type nActions: int
    :param aTransition: It seems like the description of the `aTransition` parameter got cut off. Could
    you please provide more information or complete the description of the `aTransition` parameter so
    that I can assist you further with the `GaussSeidel` function?
    :type aTransition: list
    :param aPoliticas: The `aPoliticas` parameter in the `GaussSeidel` function seems to represent a
    list that stores the best action for each state in a given policy. It is updated during the
    iteration process based on the calculated values for each state
    :type aPoliticas: list
    :param sMeta: The parameter `sMeta` in the provided function `GaussSeidel` appears to represent the
    goal state or target state in a Markov Decision Process (MDP) setting. It is used in the function to
    calculate the reward for transitioning from a current state to the goal state. The `
    :type sMeta: int
    :param threshold: The `threshold` parameter in the `GaussSeidel` function represents the convergence
    threshold. It is a floating-point value that determines the level of convergence required for the
    algorithm to stop iterating. The algorithm will continue iterating until the maximum change in the
    value of states between two consecutive iterations is less than
    :type threshold: float
    :param ld: The parameter `ld` in the GaussSeidel function represents the discount factor used in the
    Bellman equation for reinforcement learning. It is typically denoted as the Greek letter lambda (λ)
    and is used to discount future rewards in the calculation of the state values. A discount factor of
    1 means, defaults to 1
    :type ld: float (optional)
    :return: The function `GaussSeidel` is returning a tuple containing two values: the number of
    iterations `nK` and the reshaped policy `aPoliticas[0][:].reshape(7, 9)`.
    """
    nK = 1
    V = np.zeros(nStates)  # Inicializar los valores de los estados a cero

    while nK < nIteration:
        delta = 0  # Para verificar la convergencia

        for s in range(nStates):
            v_old = V[s]
            max_value = float('-inf')
            best_action = None

            for a in range(nActions):
                # Sumatoria sobre los estados siguientes
                sum_value = sum(aTransition[a][s][s_prime] * (
                    nReward(s, sMeta) + ld * V[s_prime]) for s_prime in range(nStates))

                # Buscar la accion que maximiza el valor del estado
                if sum_value > max_value:
                    max_value = sum_value
                    best_action = a

            # Actualizar el valor del estado con la informacion mas reciente
            V[s] = max_value

            if best_action is not None:
                # Guardar la mejor accion para el estado actual
                aPoliticas[0][s] = best_action

            # Actualizar delta para verificar la convergencia
            delta = max(delta, abs(v_old - V[s]))

        # Comprobar si hemos alcanzado la convergencia
        if delta < threshold:
            print(f'Convergencia alcanzada en la iteracion {nK}')
            break

        nK += 1

    # print(f"Iteracion: {nK} - Politica:\n {aPoliticas[0][:].reshape(7, 9)}")
    return nK, aPoliticas[0][:].reshape(7, 9)


def RelativeValueIteration(nIteration: int, vFunction: list, nStates: int, nActions: int, aTransition: list, QValue: list, aPoliticas: list, nErr: float, sMeta: int, ld: float = 1) -> np.ndarray:
    """
    The function `RelativeValueIteration` implements the relative value iteration algorithm to solve a
    reinforcement learning problem.
    
    :param nIteration: The `nIteration` parameter in the `RelativeValueIteration` function represents
    the maximum number of iterations that will be performed during the relative value iteration
    algorithm. This parameter controls how many times the algorithm will update the value function and
    policies based on the state-action transitions and rewards in the environment
    :type nIteration: int
    :param vFunction: The `vFunction` parameter is a list containing the value function for each state
    at each iteration of the algorithm. It is updated iteratively to approximate the optimal value
    function for the given Markov Decision Process (MDP)
    :type vFunction: list
    :param nStates: The `nStates` parameter typically refers to the number of states in a Markov
    Decision Process (MDP) or a similar environment. It represents the total number of possible states
    that the agent can be in. In the context of the `RelativeValueIteration` function you provided,
    `nStates
    :type nStates: int
    :param nActions: The `nActions` parameter in the `RelativeValueIteration` function represents the
    number of possible actions that can be taken in each state of the Markov Decision Process (MDP). It
    is used to determine the range of the action loop in the algorithm to calculate the Q-values for
    each state-action
    :type nActions: int
    :param aTransition: The `aTransition` parameter in the `RelativeValueIteration` function seems to
    represent the transition probabilities from each state to another state given an action. It is a
    3-dimensional list where the first dimension represents the action, the second dimension represents
    the current state, and the third dimension represents the next
    :type aTransition: list
    :param QValue: QValue is a list that stores the Q-values for each state-action pair in your
    reinforcement learning algorithm. It is typically represented as a 2D array where the rows
    correspond to states and the columns correspond to actions. Each element in the array represents the
    expected cumulative reward of taking a specific action in
    :type QValue: list
    :param aPoliticas: The `aPoliticas` parameter in the `RelativeValueIteration` function seems to
    represent the policy for each state in the Markov Decision Process (MDP). It is a list that stores
    the chosen action for each state based on the calculated Q-values during the value iteration process
    :type aPoliticas: list
    :param nErr: The `nErr` parameter in the `RelativeValueIteration` function represents the threshold
    for the convergence criteria. It is used to determine when the algorithm should stop iterating based
    on the norm of the difference between the current value function and the previous value function
    :type nErr: float
    :param sMeta: The `sMeta` parameter in the `RelativeValueIteration` function appears to represent
    the goal state or target state in the context of the value iteration algorithm. It is used in the
    calculation of the Q-values and rewards within the iteration process to determine the optimal policy
    for reaching this specific state
    :type sMeta: int
    :param ld: In the provided function, the parameter `ld` represents the discount factor used in the
    Bellman equation for calculating the Q-values. It is multiplied by the sum of the discounted future
    rewards when updating the Q-values in the value iteration algorithm, defaults to 1
    :type ld: float (optional)
    :return: The function `RelativeValueIteration` is returning two values: `nK`, which represents the
    number of iterations performed, and `aPoliticas[0][:].reshape(7, 9)`, which is the reshaped policy
    array.
    """
    nK = 1
    s_ref = 3

    while nK < nIteration:
        vFunction.append(np.zeros((1, nStates)))
        for s in range(0, nStates):
            if s in {14, 17, -2}:
                continue
            for a in range(0, nActions):
                aAux = [aTransition[a][s][i] * vFunction[nK-1][0][i]
                        for i in range(0, nStates)]
                QValue[s][a] = (nReward(s, sMeta) + ld *
                                sum(aAux)) - vFunction[nK-1][0][s_ref]
            vFunction[nK][0][s] = max(QValue[s][:])
            aPoliticas[0][s] = np.argmax(QValue[s][:])

        if not norma_sp(vFunction[nK][0][:] - vFunction[nK-1][0][s_ref],
                        vFunction[nK-1][0][:] - vFunction[nK-1][0][s_ref], nErr):
            break
        nK += 1

    # print(f"Iteracion: {nK} - Politica:\n {aPoliticas[0][:].reshape(7, 9)}")
    return nK, aPoliticas[0][:].reshape(7, 9)


def QValueIteration(nIteration: int, vFunction: list, nStates: int, nActions: int, aTransition: list, QValue: list, aPoliticas: list, nErr: float, sMeta: int, ld: float = 1) -> np.ndarray:
    """
    This Python function implements Q-value iteration for reinforcement learning to update value
    functions and policies based on state-action transitions and rewards.
    
    :param nIteration: The `nIteration` parameter in the `QValueIteration` function represents the
    number of iterations or steps that the algorithm will take to update the value function and policy.
    It determines how many times the algorithm will loop through the states and actions to improve the
    estimates of the Q-values and the value function
    :type nIteration: int
    :param vFunction: The `vFunction` parameter seems to represent the value function in a reinforcement
    learning setting. It is a list containing arrays where each array represents the value function for
    a specific iteration during the Q-value iteration process
    :type vFunction: list
    :param nStates: The `nStates` parameter typically refers to the number of states in a Markov
    Decision Process (MDP) or a similar environment. In the context of the provided function
    `QValueIteration`, `nStates` would represent the total number of states in the environment being
    considered for value iteration
    :type nStates: int
    :param nActions: The `nActions` parameter represents the number of possible actions that can be
    taken in the environment. In the context of the Q-value iteration function you provided, it is used
    to determine the range for the action loop when updating the Q-values for each state
    :type nActions: int
    :param aTransition: The `aTransition` parameter in the `QValueIteration` function seems to represent
    the transition probabilities between states given an action. It is a list that likely contains
    transition probabilities for each action and state pair. The structure of this list could be
    something like `aTransition[action][state][next_state
    :type aTransition: list
    :param QValue: The `QValue` parameter in the `QValueIteration` function seems to be a list that
    holds the Q-values for each state-action pair in a reinforcement learning setting. The Q-values
    represent the expected cumulative rewards of taking a particular action in a specific state and
    following a certain policy thereafter
    :type QValue: list
    :param aPoliticas: The `aPoliticas` parameter in the `QValueIteration` function seems to represent
    the policy for each state in the Markov Decision Process. It is a list that stores the chosen action
    for each state based on the Q-values calculated during the value iteration process
    :type aPoliticas: list
    :param nErr: The parameter `nErr` in the function `QValueIteration` represents the threshold value
    for the convergence criteria in the value iteration algorithm. The iteration process will stop when
    the L1 norm of the difference between the value functions of consecutive iterations falls below this
    threshold (`nErr`)
    :type nErr: float
    :param sMeta: The `sMeta` parameter in the `QValueIteration` function appears to represent the goal
    state or target state in the Markov Decision Process (MDP) being solved. It is used in the
    calculation of rewards within the function. If you have any specific questions or need further
    clarification on how
    :type sMeta: int
    :param ld: The parameter `ld` in the function `QValueIteration` represents the discount factor used
    in the Bellman equation for reinforcement learning. It determines the importance of future rewards
    in the agent's decision-making process. A discount factor of 1 means that future rewards are
    considered equally important as immediate rewards,, defaults to 1
    :type ld: float (optional)
    :return: The function `QValueIteration` returns the number of iterations `nK` and the policy as a
    reshaped 2D array `aPoliticas[0]` with dimensions 7x9.
    """
    nK = 1
    while nK < nIteration:
        vFunction.append(np.zeros((1, nStates)))
        for s in range(nStates):
            if s in [14, 17, -2]:
                continue
            for a in range(nActions):
                QValue[s][a] = sum(aTransition[a][s][i] * (nReward(s, sMeta) +
                                   ld * vFunction[nK-1][0][i]) for i in range(nStates))
            vFunction[nK][0][s] = max(QValue[s])
            aPoliticas[0][s] = np.argmax(QValue[s])

        if np.linalg.norm(vFunction[nK][0] - vFunction[nK-1][0], ord=1) < nErr:
            break
        nK += 1

    # print(f"Iteracion: {nK} - Politica:\n {aPoliticas[0].reshape(7, 9)}")
    return nK, aPoliticas[0].reshape(7, 9)

# ---------------Interfaz grafica con pygame-------------------


def Interface(aMapa: list, nStates: int, nActions: int, Ld: float, sMeta: int, aQ: np.ndarray, aP: np.ndarray, aE: list, aT: list) -> None:
    """
    The `Interface` function in Python initializes a Pygame window for a maze project, allows selection
    of algorithms for maze solving, and includes functions for running simulations and displaying
    histograms of convergence iterations.
    
    :param aMapa: The `aMapa` parameter is a list that represents the maze map. Each element in the list
    corresponds to a row in the maze, and the values within each row represent different types of cells
    in the maze. Here is an example of how the maze map might look like:
    :type aMapa: list
    :param nStates: The `nStates` parameter in the `Interface` function represents the number of states
    in the system or environment being modeled. It is an integer value that indicates the total number
    of states that the system can be in during the simulation or computation. This parameter is used in
    various algorithms and calculations within the
    :type nStates: int
    :param nActions: The `nActions` parameter in the `Interface` function represents the number of
    possible actions that can be taken in the environment or system being modeled. In reinforcement
    learning or decision-making problems, this parameter typically defines the set of actions that an
    agent can choose from at each state
    :type nActions: int
    :param Ld: The parameter `Ld` in the `Interface` function seems to represent the discount factor
    used in certain algorithms for value iteration. This discount factor is typically denoted by the
    symbol "γ" (gamma) in reinforcement learning literature. It is used to discount future rewards in
    the reinforcement learning process
    :type Ld: float
    :param sMeta: The parameter `sMeta` in the `Interface` function represents the goal state in a maze.
    It is an integer value that indicates the position of the goal state within the maze. The goal of
    the maze-solving algorithms implemented in the function is to navigate the maze from the starting
    position to this goal
    :type sMeta: int
    :param aQ: The parameter `aQ` appears to be an `np.ndarray` that is used in the `Interface` function
    you provided. It is likely used as part of the algorithm implementations within the function
    :type aQ: np.ndarray
    :param aP: The parameter `aP` in the `Interface` function seems to be related to transition
    probabilities in a Markov Decision Process (MDP). In MDPs, `aP` typically represents the transition
    probability matrix, where `aP[i][j][k]` denotes the probability of
    :type aP: np.ndarray
    :param aE: The parameter `aE` seems to be used in the `Interface` function you provided. However,
    the specific purpose or content of the `aE` parameter is not explicitly defined in the code snippet
    you shared
    :type aE: list
    :param aT: The parameter `aT` seems to be used in the function `Interface` as an input argument.
    However, the specific purpose or content of the `aT` parameter is not explicitly defined in the
    provided code snippet
    :type aT: list
    """

    pygame.init()
    clock = pygame.time.Clock()

    surface = pygame.display.set_mode((900, 700))
    pygame.display.set_caption('Proyecto 1: Laberinto')
    font_path = "src/font/gunshipcondital.ttf"
    mapa = np.array(aMapa)
    selected_policy = 0

    def load_images():
        images = {}
        for i in range(63):
            # Ruta correcta necesaria
            images[i+1] = pygame.image.load(f'src/img/fondo/{i}.png')
        images['walle'] = pygame.image.load('src/img/walle.png')
        return images

    def draw_labyrinth(surface, map_array, images):
        # Draw maze and background images
        for i in range(map_array.shape[0]):
            for j in range(map_array.shape[1]):
                image_index = i * map_array.shape[1] + j + 1
                if image_index <= 63:
                    image = pygame.transform.scale(
                        images[image_index], (100, 100))
                    surface.blit(image, (j * 100, i * 100))
                elif map_array[i, j] == -2:
                    pygame.draw.rect(surface, (255, 255, 255),
                                     (j * 100, i * 100, 100, 100))

    def set_algorithm(value, algorithm):
        nonlocal selected_policy
        selected_policy = algorithm

    def start_labyrinth(iteraciones, success=90):
        font = pygame.font.Font(font_path, 54)
        text = font.render("Cargando...", True, (255, 255, 255))
        text_rect = text.get_rect(center=(450, 350))
        surface.fill((0, 0, 0))
        surface.blit(text, text_rect)
        pygame.display.flip()

        images = load_images()
        draw_labyrinth(surface, mapa, images)

        # Probabilidad de error dado el porcentaje de exito
        pError = (100 - success) / 100

        # Run the selected algorithm
        nonlocal selected_policy
        if selected_policy == 0:
            converged, policy = ValueIteration(
                iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta)
        elif selected_policy == 1:
            converged, policy = RelativeValueIteration(
                iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta)
        elif selected_policy == 2:
            converged, policy = GaussSeidel(
                iteraciones, nStates, nActions, aT, aP, sMeta)
        elif selected_policy == 3:
            converged, policy = ValueIteration(
                iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta, Ld)
        elif selected_policy == 4:
            converged, policy = RelativeValueIteration(
                iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta, Ld)
        elif selected_policy == 5:
            converged, policy = QValueIteration(
                iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta)
        elif selected_policy == 6:
            converged, policy = QValueIteration(
                iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta, Ld)

        # Wait the convergence
        if policy is None:
            return

        # Random start position
        random_pos = [[0, 0], [1, 4], [3, 8], [6, 6]]
        current_pos = random.choice(random_pos)
        nCols = 9
        nRows = 7

        # Show the optimal policy
        print("\n Política optima (representada como matriz):")
        print(f"{policy}\n")

        while current_pos != [5, 3]:  # Continuar hasta alcanzar el estado objetivo
            action = policy[current_pos[0], current_pos[1]]
            arrow = "↑" if action == 0 else "↓" if action == 1 else "→" if action == 2 else "←"

            next_pos = list(current_pos)
            if action == 0:  # Up
                next_pos[0] -= 1
            elif action == 1:  # Down
                next_pos[0] += 1
            elif action == 2:  # Right
                next_pos[1] += 1
            elif action == 3:  # Left
                next_pos[1] -= 1

            # Verify if the next position is valid
            if (0 <= next_pos[0] < nRows) and (0 <= next_pos[1] < nCols) and (mapa[next_pos[0], next_pos[1]] != -1):
                current_pos = next_pos
                print(f"Movimiento a {current_pos}. {arrow}")
            else:
                print(f"Intento de movimiento invalido a {next_pos}. Manteniendo posicion en {current_pos}.")
  

            # Draw the maze and the robot
            draw_labyrinth(surface, mapa, images)
            walle = pygame.transform.scale(images['walle'], (100, 100))
            surface.blit(walle, (current_pos[1] * 100, current_pos[0] * 100))
            pygame.display.flip()
            pygame.time.delay(500)

        pygame.time.delay(2000)
        menu.mainloop(surface)

    def histogram():

        pError = [0.02, 0.05, 0.1, 0.3, 0.5, 0.8]
        pError = np.array([98, 95, 90, 70, 50, 20])/100
        results = []
        iterations = 100

        # Loading screen
        font = pygame.font.Font(font_path, 54)
        loading_text = font.render("Cargando...", True, (255, 255, 255))
        loading_text_rect = loading_text.get_rect(center=(450, 350))
        surface.fill((0, 0, 0))
        surface.blit(loading_text, loading_text_rect)
        pygame.display.flip()

        # 'probability': probabilidad de error a exito

        for probabilidad in pError:
            results.append({
                'algorithm': 'V-Iter',
                'probability': 100 - probabilidad*100,
                'convergence': ValueIteration(iterations, aE, nStates, nActions, aT, aQ, aP, probabilidad, sMeta)[0],
            })
            results.append({
                'algorithm': 'R-V-Iter',
                'probability': 100 - probabilidad*100,
                'convergence': RelativeValueIteration(iterations, aE, nStates, nActions, aT, aQ, aP, probabilidad, sMeta)[0],
            })
            results.append({
                'algorithm': 'G-Seidel',
                'probability': 100 - probabilidad*100,
                'convergence': GaussSeidel(iterations, nStates, nActions, aT, aP, sMeta)[0],
            })
            results.append({
                'algorithm': 'V-Iter 0.98',
                'probability': 100 - probabilidad*100,
                'convergence': ValueIteration(iterations, aE, nStates, nActions, aT, aQ, aP, probabilidad, sMeta, 0.98)[0],
            })
            results.append({
                'algorithm': 'R-V-Iter 0.98',
                'probability': 100 - probabilidad*100,
                'convergence': RelativeValueIteration(iterations, aE, nStates, nActions, aT, aQ, aP, probabilidad, sMeta, 0.98)[0],
            })
            results.append({
                'algorithm': 'Q-V-Iter-C',
                'probability': 100 - probabilidad*100,
                'convergence': QValueIteration(iterations, aE, nStates, nActions, aT, aQ, aP, probabilidad, sMeta)[0],
            })
            results.append({
                'algorithm': 'Q-V-Iter 0.98',
                'probability': 100 - probabilidad*100,
                'convergence': QValueIteration(iterations, aE, nStates, nActions, aT, aQ, aP, probabilidad, sMeta, 0.98)[0],
            })

        # Print results
        for result in results:
            print(f"{result['algorithm']} - Probabilidad de exito: {result['probability']} - Iteraciones: {result['convergence']}")

        # Plot results
        fig, ax = plt.subplots()
        for algorithm in ['V-Iter', 'R-V-Iter', 'G-Seidel', 'V-Iter 0.98', 'R-V-Iter 0.98', 'Q-V-Iter-C', 'Q-V-Iter 0.98']:
            data = [result['convergence']
                    for result in results if result['algorithm'] == algorithm]
            ax.plot(pError, data, label=algorithm)

        ax.set(xlabel='Probabilidad de error', ylabel='Iteraciones',
               title='Iteraciones para converger')
        ax.grid()
        ax.legend()
        plt.show()

    menu = pygame_menu.Menu(
        'Opciones', 900, 700, theme=pygame_menu.themes.THEME_SOLARIZED)

    # Menu configuration
    menu.add.label("Proyecto 1: Laberinto", max_char=-
                   1, font_size=40, font_color=(0, 0, 0))
    menu.add.label("Integrantes: Felipe Espinoza - Oscar Uribe",
                   max_char=-1, font_size=20, font_color=(0, 0, 0))
    menu.add.label("", max_char=-1, font_size=10)

    algoritmos = [("Value Iteration", 0),
                  ("Relative Value Iteration", 1),
                  ("Gauss-Siedel Value Iteration", 2),
                  ("Value Iteration con Factor de Descuento del 0.98", 3),
                  ("Relative Value Iteration con Factor de Descuento del 0.98", 4),
                  ("Q-Value Iteration Clasico", 5),
                  ("Q-Value Iteration con Factor de Descuento del 0.98", 6)]

    menu.add.selector('', algoritmos, onchange=set_algorithm, font_size=25)

    menu.add.label("", max_char=-1, font_size=10)
    menu.add.range_slider('Numero de Iteraciones:', 300, (100, 500), 1,
                          rangeslider_id='itera',
                          value_format=lambda x: str(int(x)),
                          font_size=25, font_color=(0, 0, 0))

    menu.add.range_slider('Probabilidad de exito:', 90, (50, 100), 1,
                          rangeslider_id='success', value_format=lambda x: str(int(x)),
                          font_size=25, font_color=(0, 0, 0))

    menu.add.label("", max_char=-1, font_size=10)

    # Start Simulation button
    menu.add.button(
        'Iniciar',
        lambda: start_labyrinth(menu.get_input_data(
        )['itera'],  menu.get_input_data()['success']),
        font_size=20,
        border_width=1,
        font_color=(0, 0, 0),
        border_color='white',
        padding=10,
        shadow_width=2)

    menu.add.label("", max_char=-1, font_size=3)

    # Histogram button
    menu.add.button(
        'Histogramas',
        lambda: histogram(),
        font_size=20,
        border_width=1,
        font_color=(0, 0, 0),
        border_color='white',
        padding=10,
        shadow_width=2)

    # Main loop
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit()

        if menu.is_enabled():
            menu.update(events)
            menu.draw(surface)

        pygame.display.flip()
        clock.tick(30)
# ------------------------------------------------------------


def main() -> None:
    """
    This Python function initializes and sets up the necessary matrices and parameters for a
    reinforcement learning algorithm to find optimal policies in a grid world environment.
    """
    aMapa = get_map()
    nRows = 7                                 # Cantidad de filas
    nCols = 9                                 # Cantidad de columnas
    nStates = nCols * nRows                   # Cantidad de estados
    nActions = 4                              # Posibles acciones
    aActions = ['N', 'S', 'E', 'W']           # Acciones
    Ld = 0.9                                  # Factor de descuento
    sMeta = 48                                # Estado Meta

    aQ = np.zeros((nStates, nActions))        # Matriz de Q-Valores
    aP = np.zeros((1, nStates))               # Matriz de Politicas
    aE = [np.ones((1, nStates))]              # Matriz de funciones de valor
    aT = []                                   # Matriz de matrices de transicion

    # Matriz transicion de cada accion [N, S, E, W]
    for a in aActions:
        aT.append(mTransition(aMapa, a, nStates, nRows, nCols))

    Interface(aMapa, nStates, nActions, Ld, sMeta, aQ, aP, aE, aT)


if __name__ == "__main__":
    main()

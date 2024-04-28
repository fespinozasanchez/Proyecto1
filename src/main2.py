#%%
import numpy as np
# Esto solo es para imprimir matrices completas en consola
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def map():
    return [
        # 00  01  02  03  04  05  06  07  08
        [1,  1,  1,  1,  1,  1,  1, -2, -2],
        [1, -2, -2,  1, -2, -2,  1, -2, -2],
        [1,  1, -2,  1, -2, -2,  1, -2, -2],
        [-1, 1,  1, -1,  1,  1,  1,  1,  1],
        [1,  1,  0,  1,  1, -2,  1, -2, -2],
        [1, -2, -2,  3, -2, -2,  1, -2, -2],
        [1,  1,  1,  1, -2, -2, -2, -2, -2]
    ]


def norma_sp(aA, aB, nError=0):
    return (max([abs(aA[i]-aB[i]) for i in range(len(aA))]) > nError)


def nReward(state: int, action) -> int:
    if state == 3:
        return 100  # Meta
    else:
        return -5


def ValueIteration(nIteration: int, vFunction: list, nStates: int, nActions: int, aTransition: list, QValue: list, aPoliticas: list, nErr: float, ld: float = 1):
    nK = 0
    vFunction.append(np.zeros(nStates))  # Initial Value Function as zeros
    aPoliticas.append(np.zeros(nStates, dtype=int))  # Initial policy as zeros

    while True:
        vFunction.append(np.zeros(nStates))
        # Append new policy array for each iteration
        aPoliticas.append(np.zeros(nStates, dtype=int))
        for s in range(nStates):
            for a in range(nActions):
                QValue[s][a] = nReward(
                    s, a) + ld * sum([aTransition[a][s][i] * vFunction[nK][i] for i in range(nStates)])
            vFunction[nK + 1][s] = max(QValue[s])
            aPoliticas[nK][s] = int(np.argmax(QValue[s]))

        if not norma_sp(vFunction[nK + 1], vFunction[nK], nErr):
            break
        nK += 1

    print("Optimal policies calculated.")
    print("-" * 24)
    print(aPoliticas[nK])
    print("-" * 24)
    print(aPoliticas[nK].reshape(7, 9))
    politica = []
    for i in aPoliticas[nK]:
        if i == 0:
            politica.append('N')
        elif i == 1:
            politica.append('S')
        elif i == 2:
            politica.append('E')
        elif i == 3:
            politica.append('W')
    print("Política óptima: ", politica)
    return aPoliticas[nK]


# -----------------------Creación matrices de transición probabilidad para cada acción-----------------------------
#


def pos_to_index(row, col, nCols):
    return row * nCols + col


def index_to_pos(index, nCols):
    return (index // nCols, index % nCols)


def build_transition_matrix(aMap, action, p_success=0.9, p_side=0.05):
    """ Con esta función tomamos en cuenta las probabilidades de éxito y de ir a los lados  90% , 5% y 5% respectivamente 
        Ademas se toma en cuenta las murallas y bloques invisibles.
    """
    nRows, nCols = len(aMap), len(aMap[0])
    nStates = nRows * nCols
    T = np.zeros((nStates, nStates))  # Matriz de transición initializada en 0

    for row in range(nRows):
        for col in range(nCols):
            if aMap[row][col] < 0:  # Muralla o bloque invisible
                continue
            current_index = pos_to_index(row, col, nCols)

            # Determinar las posiciones objetivo para cada acción
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

# ----------------------------------------------------------------------------------------------------------------

def simulate_agent(start_row, start_col, policies, aMap, max_steps=100):
    nCols = len(aMap[0])
    position = (start_row, start_col)
    steps = 0

    while steps < max_steps:
        current_index = pos_to_index(position[0], position[1], nCols)
        action = policies[current_index]  # Acción según la política
        
        # Mover según la acción
        if action == 0:  # Norte
            next_position = (position[0] - 1, position[1])
        elif action == 1:  # Sur
            next_position = (position[0] + 1, position[1])
        elif action == 2:  # Este
            next_position = (position[0], position[1] + 1)
        elif action == 3:  # Oeste
            next_position = (position[0], position[1] - 1)

        # Chequear si la nueva posición es válida
        if (0 <= next_position[0] < len(aMap) and
            0 <= next_position[1] < len(aMap[0]) and
            aMap[next_position[0]][next_position[1]] >= 0):
            position = next_position
        else:
            break  # Si choca con un obstáculo, termina la simulación

        # Verificar si ha llegado a la meta
        if position == (5, 3):
            return True, steps
        
        steps += 1

    return False, steps  # Retornar False si no llega a la meta en los pasos dados




def main():
    ld = 0.9                             # Factor de descuento
    nErr = 0.0001 * (1-ld)/2*ld          # Error
    nCols = 9                            # Número de columnas
    nRows = 7                            # Número de filas
    nStates = nCols * nRows              # Número de estados
    nActions = 4                         # Número de acciones
    nE = 0.0001                          # Cota de error
    actions = ['N', 'S', 'E', 'W']       # Acciones
    aQ = np.zeros((nStates, nActions))   # Matriz de Q(s,a) Value
    aP = []          # Matriz de Política *
    aT = []                              # Matrices de transición [N, S, E, W]
    aE = []                              # Value Function
    # iK = 1                               # Número de iteracion actual
    goal = [5, 3]                        # Posición objetivo
    aMap = map()                         # Mapa

    # Prob Trans. Accion N  = (0: go North)
    T_N = build_transition_matrix(aMap, 'N')

    # Prob Trans. Accion S  = (1: go South)
    T_S = build_transition_matrix(aMap, 'S')

    # Prob Trans. Accion E  = (2: go East)
    T_E = build_transition_matrix(aMap, 'E')

    # Prob Trans. Accion O  = (3: go West)
    T_W = build_transition_matrix(aMap, 'W')

    # Load Distr. Prob dada la Accion
    aT.append(T_N)
    aT.append(T_S)
    aT.append(T_E)
    aT.append(T_W)

    aPoliticas = ValueIteration(100, aE, nStates, nActions, aT, aQ, aP, nE, ld)
    # Simular desde varios puntos de inicio
    starting_points = [(0, 0), (4, 2), (3, 1)]  # Ejemplos de puntos de inicio
    results = {sp: simulate_agent(sp[0], sp[1], aPoliticas.flatten(), map()) for sp in starting_points}

    print(results)

if __name__ == "__main__":
    main()

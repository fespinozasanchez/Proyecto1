#%%
#Proyecto#1: Robótica_INFO1167
#Felipe Espinoza - Oscar Uribe

import numpy as np
import json

# Esto solo es para imprimir matrices completas en consola 
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def norma_sp(aA, aB, nError=0):
    return (max([abs(aA[i]-aB[i]) for i in range(len(aA))]))


def nReward(state, goal, aMap):
    row, col = state
    if state == goal:
        return 100  #Meta
    elif aMap[row][col] == -1:
        return -10  # Obstáculo
    else:
        return -1  # Camino


def ValueIterationClassic(nIterations, nStates, T_N, T_S, T_E, T_W, aMap, goal, ld=0.9, nErr=1e-4):
    values = np.zeros(nStates)  # Inicializa los valores de los estados
    nCols = len(aMap[0])
    
    for _ in range(nIterations):
        new_values = np.copy(values)
        for state in range(nStates):
            row, col = index_to_pos(state, nCols)
            if aMap[row][col] < 0:
                continue  # No calcular valores para obstáculos
            
            # Calcular el valor máximo de tomar cualquier acción desde este estado
            max_value = -float('inf')
            for action, T in [('N', T_N), ('S', T_S), ('E', T_E), ('W', T_W)]:
                value = sum(T[state, s] * (nReward(index_to_pos(s, nCols), goal, aMap) + ld * values[s]) for s in range(nStates))
                if value > max_value:
                    max_value = value
            
            new_values[state] = max_value
        
        # Comprobar convergencia
        if np.max(np.abs(new_values - values)) < nErr:
            print("Convergencia alcanzada.")
            break
        
        values = new_values
    
    # Imprime los valores finales para inspección
    print("Valores finales de los estados:")
    print(values.reshape(len(aMap), nCols))
    return values

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
                    T[current_index, current_index] += prob  # Se queda en el mismo lugar
                else:
                    target_index = pos_to_index(r, c, nCols)
                    T[current_index, target_index] += prob
    
    return T

# ----------------------------------------------------------------------------------------------------------------


ld = 0.9
nErr = 0.0001 * (1-ld) / (2 * ld)
nCols = 9
nRows = 7
ik = [0] * (nCols * nRows)  # Inicializa 'ik' correctamente
nStates = nCols * nRows
goal = pos_to_index(5, 3, nCols)  # Conversión de la posición objetivo a índice
aMap = [
    [1,  1,  1,  1,  1,  1,  1, -2, -2],
    [1, -2, -2,  1, -2, -2,  1, -2, -2],
    [1,  1, -2,  1, -2, -2,  1, -2, -2],
    [-1, 1,  1, -1,  1,  1,  1,  1,  1],
    [1,  1,  0,  1,  1, -2,  1, -2, -2],
    [1, -2, -2,  3, -2, -2,  1, -2, -2],
    [1,  1,  1,  1, -2, -2, -2, -2, -2]
]

# Construir matrices de transición
T_N = build_transition_matrix(aMap, 'N')
T_S = build_transition_matrix(aMap, 'S')
T_E = build_transition_matrix(aMap, 'E')
T_W = build_transition_matrix(aMap, 'W')

# Ejecutar ValueIterationClassic
vic = ValueIterationClassic(1000, nStates, T_N, T_S, T_E, T_W, aMap, goal, ld, nErr)











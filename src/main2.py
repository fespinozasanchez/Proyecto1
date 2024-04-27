import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def norma_sp(aA, aB, nError=0):
    return (max([abs(aA[i]-aB[i]) for i in range(len(aA))]))


def nReward(s, a):
    pass


def ValueIteration(nIteration, nStates, T1, T2, T3, T4, ik, aA, nErr, ld=1):
    pass

# -----------------------Creación matrices de transición probabilidad para cada acción-----------------------------
# 
def pos_to_index(row, col, nCols):
    return row * nCols + col

def index_to_pos(index, nCols):
    return (index // nCols, index % nCols)

def build_transition_matrix(aMap, action, p_success=0.9, p_side=0.05):
    nRows, nCols = len(aMap), len(aMap[0])
    nStates = nRows * nCols
    T = np.zeros((nStates, nStates))  # Matriz de transición
    
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

def main():
    ld = 0.9
    nErr = 0.0001 * (1-ld)/2*ld
    nCols = 9
    nRows = 7
    ik = []
    nStates = nCols * nRows
    goal = [5, 3]
    aRewards = np.zeros((nRows, nCols))
    actions = {'N': 0, 'S': 1, 'E': 2, 'W': 3}
    aA = [0, 0, 0, 0]
    aMap = [
        # 00  01  02  03  04  05  06  07  08
        [1,  1,  1,  1,  1,  1,  1, -2, -2],
        [1, -2, -2,  1, -2, -2,  1, -2, -2],
        [1,  1, -2,  1, -2, -2,  1, -2, -2],
        [-1, 1,  1, -1,  1,  1,  1,  1,  1],
        [1,  1,  0,  1,  1, -2,  1, -2, -2],
        [1, -2, -2,  3, -2, -2,  1, -2, -2],
        [1,  1,  1,  1, -2, -2, -2, -2, -2]
    ]

    # Prob Trans. Accion N  = (0: go North)
    T_N = build_transition_matrix(aMap, 'N')

    # Prob Trans. Accion S  = (1: go South)
    T_S = build_transition_matrix(aMap, 'S')
    
    # Prob Trans. Accion E  = (2: go East)
    T_E = build_transition_matrix(aMap, 'E')
    
    # Prob Trans. Accion O  = (3: go West)
    T_W = build_transition_matrix(aMap, 'W')



    print(len(T_N))

if __name__ == "__main__":
    main()

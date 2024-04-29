#%%
#Proyecto#1: Robótica_INFO1167
#Felipe Espinoza - Oscar Uribe

import numpy as np
import random
# Esto solo es para imprimir matrices completas en consola
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def getMap():
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
    vFunction.append(np.zeros(nStates))  # Función de valor inicial como ceros
    aPoliticas.append(np.zeros(nStates, dtype=int))  # Política inicial como ceros

    while True:
        vFunction.append(np.zeros(nStates))
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

    # Convertir índices de acción en direcciones
    return aPoliticas[nK]

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
    print("Value Iteration: ", politica)
    return aPoliticas[nK]

def RelativeValueIteration(aMap, aT, nStates, nActions, nErr, goal):
    J = np.zeros(nStates)  # Inicializar la función de valor
    rho = 0  # Valor en el estado distinguido
    distinguished_state = pos_to_index(goal[0], goal[1], len(aMap[0])) 

    while True:
        J_new = np.zeros(nStates)
        for state in range(nStates):
            if index_to_pos(state, len(aMap[0])) in [(row, col) for row in range(len(aMap)) for col in range(len(aMap[0])) if aMap[row][col] < 0]:
                continue  # Saltar estados no permitidos
            
            max_value = float('-inf')
            for action in range(nActions):
                sum_prob = 0
                for next_state in range(nStates):
                    sum_prob += aT[action][state, next_state] * J[next_state]
                max_value = max(max_value, nReward(state, action) + sum_prob)
            
            J_new[state] = max_value

        # Normalizar usando el estado distinguido
        rho_new = J_new[distinguished_state]
        J_new -= rho_new
        
        # Verificar la convergencia
        if norma_sp(J, J_new, nErr):
            break
        
        J = J_new
        rho = rho_new
    
    # Extraer la política óptima
    policy = np.zeros(nStates, dtype=int)
    for state in range(nStates):
        best_action = None
        max_value = float('-inf')
        for action in range(nActions):
            value = nReward(state, action) + sum(aT[action][state, next_state] * J[next_state] for next_state in range(nStates))
            if value > max_value:
                max_value = value
                best_action = action
        policy[state] = best_action
    
    return policy



# -----------------------Creación matrices de transición probabilidad para cada acción-----------------------------
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
# ----------------------------------------Interfaz grafica con pygame---------------------------------------------
import pygame 
import pygame_menu

def Interface(Politicas):
    pygame.init()
    surface = pygame.display.set_mode((1000, 800))
    pygame.display.set_caption('Proyecto 1: Laberinto')
    selected_policy = None
    
    def set_algorithm(value, algorithm):
        nonlocal selected_policy  # Indica que selected_policy es no local
        selected_policy = Politicas[algorithm]
    
    def start_labyrinth():
        print("Iniciando laberinto")
        if selected_policy is not None:
            print("Con la política del algoritmo seleccionado:", selected_policy)
        else:
            print("No se ha seleccionado ninguna política")
    

    menu = pygame_menu.Menu('Opciones', 1000, 800, theme=pygame_menu.themes.THEME_SOLARIZED) # Main menu

    # Menu items
    menu.add.label("Proyecto 1: Laberinto", max_char=-1, font_size=35, font_color=(0,0,0))
    menu.add.label("Integrantes: Felipe Espinoza, Oscar Uribe", max_char=-1, font_size=25, font_color=(0,0,0))
    menu.add.label("", max_char=-1, font_size=45)
    
    algoritmos =[("Value Iteration", 0), 
                ("Relative Value Iteration", 1), 
                ("Gauss-Siedel Value Iteration", 2), 
                ("Value Iteration con Factor de Descuento del 0.98", 3), 
                ("Relative Value Iteration con Factor de Descuento del 0.98", 4),
                ("Q-Value Iteration Clásico", 5), 
                ("Q-Value Iteration con Factor de Descuento del 0.98", 6)]

    menu.add.selector('', algoritmos, onchange=set_algorithm)
    menu.add.button('Iniciar', start_labyrinth)
    menu.add.button('Salir', pygame_menu.events.EXIT)

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
# ----------------------------------------------------------------------------------------------------------------


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
    aP = []                              # Matriz de Política *
    aT = []                              # Matrices de transición [N, S, E, W]
    aE = []                              # Value Function
    iK = 1                               # Número de iteracion actual
    goal = [5, 3]                        # Posición objetivo
    aMap = getMap()                         # Mapa

    # Prob Trans. Accion N  = (0: go North)
    T_N = build_transition_matrix(aMap, 'N')

    # Prob Trans. Accion S  = (1: go South)
    T_S = build_transition_matrix(aMap, 'S')

    # Prob Trans. Accion E  = (2: go East)
    T_E = build_transition_matrix(aMap, 'E')

    # Prob Trans. Accion O  = (3: go West)
    T_W = build_transition_matrix(aMap, 'W')

    # Load Distr. Prob dada la Accion
    aT.append(T_N), aT.append(T_S), aT.append(T_E), aT.append(T_W)

    aPoliticas = ValueIteration(100, aE, nStates, nActions, aT, aQ, aP, nErr, ld)
    bPoliticas = RelativeValueIteration(aMap, aT, nStates, nActions, nErr, goal)
    cPoliticas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] #Sin implementar
    dPoliticas = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] #Sin implementar
    ePoliticas = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3] #Sin implementar
    fPoliticas = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4] #Sin implementar
    gPoliticas = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5] #Sin implementar

    
    
    Politicas = {0: aPoliticas, 1: bPoliticas, 2: cPoliticas, 3: dPoliticas, 4: ePoliticas, 5: fPoliticas, 6: gPoliticas}
    Interface(Politicas)


if __name__ == "__main__":
    main()














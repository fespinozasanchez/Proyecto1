#%%
#Proyecto#1: Robótica_INFO1167
#Felipe Espinoza - Oscar Uribe

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

    # Lista de algoritmos como tuplas
    algoritmos = [("Value Iteration", 0), 
                  ("Relative Value Iteration", 1), 
                  ("Gauss-Siedel Value Iteration", 2), 
                  ("Value Iteration con Factor de Descuento del 0.98", 3), 
                  ("Relative Value Iteration con Factor de Descuento del 0.98", 4),
                  ("Q-Value Iteration Clásico", 5), 
                  ("Q-Value Iteration con Factor de Descuento del 0.98", 6)]
    

    # Creación del menú
    menu = pygame_menu.Menu('Opciones', 1000, 800, theme=pygame_menu.themes.THEME_SOLARIZED)

    #Nombre proyecto y autores en la parte superior de la pantalla
    menu.add.label("Proyecto 1: Laberinto", max_char=-1, font_size=35, font_color=(0,0,0))
    menu.add.label("Integrantes: Felipe Espinoza, Oscar Uribe", max_char=-1, font_size=25, font_color=(0,0,0))
    menu.add.label("", max_char=-1, font_size=45)

    menu.add.selector('', algoritmos, onchange=set_algorithm)
    menu.add.button('Iniciar', start_labyrinth)
    menu.add.button('Salir', pygame_menu.events.EXIT)


    # Bucle principal del menú
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
    # iK = 1                             # Número de iteracion actual
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
    bPoliticas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cPoliticas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dPoliticas = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    ePoliticas = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    fPoliticas = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    gPoliticas = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    
    
    Politicas = {0: aPoliticas, 1: bPoliticas, 2: cPoliticas, 3: dPoliticas, 4: ePoliticas, 5: fPoliticas, 6: gPoliticas}
    Interface(Politicas)


if __name__ == "__main__":
    main()














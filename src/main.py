#%%
#Proyecto#1: Robótica_INFO1167
#Felipe Espinoza - Oscar Uribe

import numpy as np
import random

def getMap():
    return [
        # 00  01  02  03  04  05  06  07  08
        [1,  1,  1,  1,  1,  1,  1, -2, -2],
        [1, -2, -2,  1, -2, -2,  1, -2, -2],
        [1,  1, -2,  1, -2, -2,  1, -2, -2],
        [-1, 1,  1, -1,  1,  1,  1,  1,  1],
        [1,  1, -2,  1,  1, -2,  1, -2, -2],
        [1, -2, -2,  3, -2, -2,  1, -2, -2],
        [1,  1,  1,  1, -2, -2, -2, -2, -2]
    ]


def norma_sp(aA, aB, nError=0):
    return (max([abs(aA[i]-aB[i]) for i in range(len(aA))]) > nError)


def nReward(state: int, action) -> int:
    if state == 3:
        return 100  # Meta
    else:
        return -1


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
    #return aPoliticas[nK]

    politica = []
    for i in aPoliticas[nK]:
        if i == 0:
            politica.append('↑')
        elif i == 1:
            politica.append('↓')
        elif i == 2:
            politica.append('→')
        elif i == 3:
            politica.append('←')
    return aPoliticas[nK]

def RelativeValueIteration(aMap, aT, nStates, nActions, nErr, goal):
    pass


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

def Interface(Politicas, aMapa):
    pygame.init()
    surface = pygame.display.set_mode((900, 700))
    pygame.display.set_caption('Proyecto 1: Laberinto')
    selected_policy = Politicas[0]  # Política inicial predeterminada
    font_path = "src/font/gunshipcondital.ttf"
    clock = pygame.time.Clock()
    mapa = np.array(aMapa)

    def load_images():
        # Carga las imágenes del fondo
        images = {}
        for i in range(63):
            images[i+1] = pygame.image.load(f'src/img/fondo/{i}.png')  # Ruta correcta necesaria
        images['walle'] = pygame.image.load('src/img/walle.png')
        return images
    
    def draw_labyrinth(surface, map_array, images):
        # Dibuja el laberinto en la superficie
        for i in range(map_array.shape[0]):
            for j in range(map_array.shape[1]):
                image_index = i * map_array.shape[1] + j + 1
                if image_index <= 63:
                    image = pygame.transform.scale(images[image_index], (100, 100))
                    surface.blit(image, (j * 100, i * 100))
                elif map_array[i, j] == -2:
                    pygame.draw.rect(surface, (255, 255, 255), (j * 100, i * 100, 100, 100))

    def set_algorithm(value, algorithm):
        nonlocal selected_policy
        selected_policy = Politicas[algorithm]  # Actualiza la política seleccionada



    def start_labyrinth():
        print("Iniciando laberinto con la política del algoritmo seleccionado:", selected_policy)

        images = load_images()
        draw_labyrinth(surface, mapa, images)
        pygame.display.flip() 

        current_pos = [0, 0]
        for action in selected_policy:
            if action == 0:  # Arriba
                current_pos[0] -= 1
            elif action == 1:  # Abajo
                current_pos[0] += 1
            elif action == 2:  # Derecha
                current_pos[1] += 1
            elif action == 3:  # Izquierda
                current_pos[1] -= 1

            if current_pos[0] < 0 or current_pos[0] >= mapa.shape[0] or current_pos[1] < 0 or current_pos[1] >= mapa.shape[1] or mapa[current_pos[0], current_pos[1]] == -2:
                break

            surface.fill((0, 0, 0))
            draw_labyrinth(surface, mapa, images)
            walle = pygame.transform.scale(images['walle'], (100, 100))
            surface.blit(walle, (current_pos[1] * 100, current_pos[0] * 100))
            pygame.display.flip()
            pygame.time.delay(500)

        pygame.time.delay(2000)
        pygame.quit()
        exit()

    menu = pygame_menu.Menu('Opciones', 900, 700, theme=pygame_menu.themes.THEME_SOLARIZED) # Main menu

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
    Interface(Politicas, aMap)


if __name__ == "__main__":
    main()














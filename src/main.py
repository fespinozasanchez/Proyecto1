# %%
# Proyecto#1: Robótica_INFO1167
# Felipe Espinoza - Oscar Uribe

import pygame
import pygame_menu
import numpy as np
import random


def map() -> list:
    return [
        [0,  1,  2,  3,  4,  5,  6, -2,  -2],
        [7, -2, -2,  8, -2, -2,  9, -2,  -2],
        [10, 11, -2, 12, -2, -2, 13, -2, -2],
        [14, 15, 16, 17, 18, 19, 20, 21, 22],
        [23, 24, -2, 25, 26, -2, 27, -2, -2],
        [28, -2, -2, 29, -2, -2, 30, -2, -2],
        [31, 32, 33, 34, -2, -2, -2, -2, -2]
        # CON 0.9# [1. 3. 2. 2. 2. 2. 1. 0. 0.
        #  1. 0. 0. 1. 0. 0. 1. 0. 0.
        #  2. 1. 0. 1. 0. 0. 1. 0. 0.
        #  0. 1. 3. 0. 1. 3. 3. 3. 3.
        #  1. 3. 0. 1. 3. 0. 0. 0. 0.
        #  1. 0. 0. 2. 0. 0. 0. 0. 0.
        #  2. 2. 2. 0. 0. 0. 0. 0. 0.]

        # CON 1 # [2. 2. 2. 2. 2. 2. 1. 0. 0.
        #  0. 0. 0. 0. 0. 0. 1. 0. 0.
        #  0. 1. 0. 0. 0. 0. 1. 0. 0.
        #  0. 1. 3. 0. 1. 3. 3. 3. 3.
        #  1. 3. 0. 1. 3. 0. 0. 0. 0.
        #  1. 0. 0. 2. 0. 0. 0. 0. 0.
        #  2. 2. 2. 0. 0. 0. 0. 0. 0.]
    ]


def norma_sp(aA, aB, nError=0):
    return (max([abs(aA[i]-aB[i]) for i in range(len(aA))]) > nError)


def nReward(state: int, mState: int) -> int:
    if state == mState:
        return 100
    elif state == 14 or state == 17:
        return -100
    elif state == -2:
        return -100
    else:
        return -50

# ----------------------------------------cREACIÓN MATRIZ TRANSISICION---------------------------------------------


def pos_to_index(row, col, nCols):
    return row * nCols + col


def index_to_pos(index, nCols):
    return (index // nCols, index % nCols)


def mTransition(aMap, action, nStates, nRows, nCols, p_success=0.9, p_side=0.05):
    """ Con esta función tomamos en cuenta las probabilidades de éxito y de ir a los lados  90% , 5% y 5% respectivamente 
        Ademas se toma en cuenta las murallas y bloques invisibles.
    """
    T = np.zeros((nStates, nStates))  # Matriz de transición initializada en 0

    for row in range(nRows):
        for col in range(nCols):
            # Muralla o bloque invisible
            if aMap[row][col] == -2 or aMap[row][col] == 14 or aMap[row][col] == 17:
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

# ----------------------------------------ALGORITMOS---------------------------------------------


def ValueIteration(nIteration: int, vFunction: list, nStates: int, nActions: int, aTransition: list, QValue: list, aPoliticas: list, nErr: float, sMeta: int, ld: float = 1):
    nK = 1
    while nK < nIteration:
        vFunction.append(np.zeros((1, nStates)))

        for s in range(0, nStates):
            for a in range(0, nActions):
                if s == 14 or s == 17 or s == -2:
                    continue
                aAux = [aTransition[a][s][i]*vFunction[nK-1][0][i]
                        for i in range(0, nStates)]
                QValue[s][a] = nReward(s, sMeta) + ld*sum(aAux)
            vFunction[nK][0][s] = max(QValue[s][:])
            aPoliticas[0][s] = np.argmax(QValue[s][:])
        if not norma_sp(vFunction[nK][0][:], vFunction[nK-1][0][:], nErr):
            break
        nK += 1
    print(f"Iteración: {nK} - Politica:\n {aPoliticas[0][:].reshape(7, 9)}")
    return aPoliticas[0][:].reshape(7, 9)


def GaussSeidel() -> list:
    pass


def RelativeValueIteration() -> list:
    pass


def QValueIteration() -> list:
    pass

# ----------------------------------------Interfaz grafica con pygame---------------------------------------------


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
            # Ruta correcta necesaria
            images[i+1] = pygame.image.load(f'src/img/fondo/{i}.png')
        images['walle'] = pygame.image.load('src/img/walle.png')
        return images

    def draw_labyrinth(surface, map_array, images):
        # Dibuja el laberinto en la superficie
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
        # Actualiza la política seleccionada
        selected_policy = Politicas[algorithm]


    def start_labyrinth():
        print("Iniciando laberinto con la política del algoritmo seleccionado:", selected_policy)

        # Loading screen
        font = pygame.font.Font(font_path, 50)
        text = font.render("Cargando...", True, (255, 255, 255))
        surface.fill((0, 0, 0))
        surface.blit(text, (300, 300))
        pygame.display.flip()
        pygame.time.delay(500)
        

        images = load_images()
        draw_labyrinth(surface, mapa, images)
        pygame.display.flip()

        # Asumimos que el robot comienza en la posición (0, 0)
        current_pos = [1, 4]
        nCols = 9
        nRows = 7

        while current_pos != [5, 3]:  # Continuar hasta alcanzar el estado objetivo
            action = selected_policy[current_pos[0], current_pos[1]]

            next_pos = list(current_pos)
            if action == 0:  # Arriba
                next_pos[0] -= 1
            elif action == 1:  # Abajo
                next_pos[0] += 1
            elif action == 2:  # Derecha
                next_pos[1] += 1
            elif action == 3:  # Izquierda
                next_pos[1] -= 1

            # Verificar si la nueva posición es válida
            if 0 <= next_pos[0] < nRows and 0 <= next_pos[1] < nCols and mapa[next_pos[0], next_pos[1]] != -1:
                current_pos = next_pos
            else:
                print(f"Intento de movimiento inválido a {next_pos}. Manteniendo posición en {current_pos}.")

            # Dibujar estado actual del laberinto y posición del robot
            surface.fill((0, 0, 0))
            draw_labyrinth(surface, mapa, images)
            walle = pygame.transform.scale(images['walle'], (100, 100))
            surface.blit(walle, (current_pos[1] * 100, current_pos[0] * 100))
            pygame.display.flip()
            pygame.time.delay(500)

        pygame.time.delay(2000)
        # Vuelve al menú principal
        menu.mainloop(surface)

    menu = pygame_menu.Menu(
        'Opciones', 900, 700, theme=pygame_menu.themes.THEME_SOLARIZED)  # Main menu

    # Menu items
    menu.add.label("Proyecto 1: Laberinto", max_char=-
                   1, font_size=35, font_color=(0, 0, 0))
    menu.add.label("Integrantes: Felipe Espinoza, Oscar Uribe",
                   max_char=-1, font_size=25, font_color=(0, 0, 0))
    menu.add.label("", max_char=-1, font_size=45)

    algoritmos = [("Value Iteration", 0),
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
    nCols = 9                            # Número de columnas
    nRows = 7                            # Número de filas
    nStates = nCols * nRows              # Número de estados
    nActions = 4                         # Número de acciones
    Ld = 0.9                             # Factor de descuento
    nError = 0.0001 * (1-Ld)/2*Ld        # Error
    sMeta = 48                           # Estado Meta
    aMap = map()                         # Mapa
    aActions = ['N', 'S', 'E', 'W']      # Acciones
    aTn = mTransition(aMap, aActions[0], nStates, nRows, nCols)  # Matriz de transición para la acción Norte
    aTs = mTransition(aMap, aActions[1], nStates, nRows, nCols)  # Matriz de transición para la acción Sura
    aTe = mTransition(aMap, aActions[2], nStates, nRows, nCols)  # Matriz de transición para la acción Este
    aTw = mTransition(aMap, aActions[3], nStates, nRows, nCols)  # Matriz de transición para la acción Oeste
    aQ = np.zeros((nStates, nActions))   # Matriz de Q-Valores
    aP = np.zeros((1, nStates))          # Matriz de Políticas
    aT = []                              # Matriz de matrices de transiciones
    aE = []                              # Matriz de funciones de valor
    aE.append(np.ones((1, nStates)))     # Función de valor inicial [1,1,1,1,1....]
    aT.append(aTn); aT.append(aTs); aT.append(aTe); aT.append(aTw)

    # Value Iteration Classic
    vPoliticas = ValueIteration(1000, aE, nStates, nActions, aT, aQ, aP, nError, sMeta)
    rPoliticas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Sin implementar
    gPoliticas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Sin implementar
    vdPoliticas = ValueIteration( 1000, aE, nStates, nActions, aT, aQ, aP, nError, sMeta, Ld)
    rePoliticas = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # Sin implementar
    qvPoliticas = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # Sin implementar
    qvdPoliticas = [1, 2, 2, 2, 2, 2, 1, 1, 0, 1, 2, 1, 0, 1, 0, 1, 3, 0,
                    1, 3, 3, 3, 3, 1, 3, 1, 3, 0, 1, 0, 0, 2, 2, 2, 0]  # Sin implementar

    Politicas = {0: vPoliticas, 1: rPoliticas, 2: gPoliticas,
                 3: vdPoliticas, 4: rePoliticas, 5: qvPoliticas, 6: qvdPoliticas}
    Interface(Politicas, aMap)


if __name__ == "__main__":
    main()

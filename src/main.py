# Proyecto#1: Robótica_INFO1167
# Felipe Espinoza - Oscar Uribe

import pygame
import pygame_menu
import numpy as np
import random


def get_map() -> list:
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
    return (max([abs(aA[i]-aB[i]) for i in range(len(aA))]) > nError)


def nReward(state: int, mState: int) -> int:
    if state == mState:
        return 100
    elif state == 14 or state == 17:
        return -100
    elif state == -2:
        return -50
    else:
        return -1

# ----------------------------------------cREACIÓN MATRIZ TRANSISICION---------------------------------------------


def pos_to_index(row: int, col: int, nCols: int) -> int:
    return row * nCols + col


def index_to_pos(index: int, nCols: int) -> int:
    return (index // nCols, index % nCols)


def mTransition(aMap: list, action: str, nStates: int, nRows: int, nCols: int, p_success: float = 0.9, p_side: float = 0.05) -> np.ndarray:
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


def ValueIteration(nIteration: int, vFunction: list, nStates: int, nActions: int, aTransition: list, QValue: list, aPoliticas: list, nErr: float, sMeta: int, ld: float = 1) -> np.ndarray:
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

    print(f"Iteración: {nK} - Politica:\n {aPoliticas[0][:].reshape(7, 9)}")
    return nK, aPoliticas[0][:].reshape(7, 9)


def GaussSeidel(nIteration: int, nStates: int, nActions: int, aTransition: list, aPoliticas: list,  sMeta: int, threshold: float = 0.01, ld: float = 1) -> tuple:
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
                
                # Buscar la acción que maximiza el valor del estado
                if sum_value > max_value:
                    max_value = sum_value
                    best_action = a
            
            V[s] = max_value  # Actualizar el valor del estado con la información más reciente
            
            if best_action is not None:
                aPoliticas[0][s] = best_action  # Guardar la mejor acción para el estado actual
            
            # Actualizar delta para verificar la convergencia
            delta = max(delta, abs(v_old - V[s]))
        
        # Comprobar si hemos alcanzado la convergencia
        if delta < threshold:
            print(f'Convergencia alcanzada en la iteración {nK}')
            break
        
        nK += 1

    print(f"Iteración: {nK} - Politica:\n {aPoliticas[0][:].reshape(7, 9)}")
    return nK, aPoliticas[0][:].reshape(7, 9)


def RelativeValueIteration(nIteration: int, vFunction: list, nStates: int, nActions: int, aTransition: list, QValue: list, aPoliticas: list, nErr: float, sMeta: int, ld: float = 1) -> np.ndarray:
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

    print(f"Iteración: {nK} - Politica:\n {aPoliticas[0][:].reshape(7, 9)}")
    return nK, aPoliticas[0][:].reshape(7, 9)


def QValueIteration(nIteration: int, vFunction: list, nStates: int, nActions: int, aTransition: list, QValue: list, aPoliticas: list, nErr: float, sMeta: int, ld: float = 1) -> np.ndarray:
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

    print(f"Iteración: {nK} - Politica:\n {aPoliticas[0].reshape(7, 9)}")
    return nK, aPoliticas[0].reshape(7, 9)

# ----------------------------------------Interfaz grafica con pygame---------------------------------------------

def Interface(aMapa: list, nStates: int, nActions: int, Ld: float, sMeta: int, aQ: np.ndarray, aP: np.ndarray, aE: list, aT: list) -> None:    
    
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

        # Probabilidad de error dado el porcentaje de éxito
        pError = (100 - success) / 100

        # Run the selected algorithm
        nonlocal selected_policy
        if selected_policy == 0:
            converged, policy = ValueIteration(iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta)
        elif selected_policy == 1:
            converged, policy = RelativeValueIteration(iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta)
        elif selected_policy == 2:
            converged, policy = GaussSeidel(iteraciones, nStates, nActions, aT, aP, sMeta)
        elif selected_policy == 3:
            converged, policy = ValueIteration(iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta, Ld)
        elif selected_policy == 4:
            converged, policy = RelativeValueIteration(iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta, Ld)
        elif selected_policy == 5:
            converged, policy = QValueIteration(iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta)
        elif selected_policy == 6:
            converged, policy = QValueIteration(iteraciones, aE, nStates, nActions, aT, aQ, aP, pError, sMeta, Ld)


        # Wait the convergence
        if policy is None:
            return
                
        # Random start position
        random_pos = [[0, 0], [1, 4], [3, 8], [6, 6]]
        current_pos = random.choice(random_pos)
        nCols = 9; nRows = 7

        while current_pos != [5, 3]:  # Continuar hasta alcanzar el estado objetivo
            action = policy[current_pos[0], current_pos[1]]

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
            else:
                print(f"Intento de movimiento inválido a {next_pos}. Manteniendo posición en {current_pos}.")

            # Draw the maze and the robot
            draw_labyrinth(surface, mapa, images)
            walle = pygame.transform.scale(images['walle'], (100, 100))
            surface.blit(walle, (current_pos[1] * 100, current_pos[0] * 100))
            pygame.display.flip()
            pygame.time.delay(500)

        pygame.time.delay(2000)
        menu.mainloop(surface)

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
                  ("Q-Value Iteration Clásico", 5),
                  ("Q-Value Iteration con Factor de Descuento del 0.98", 6)]

    menu.add.selector('', algoritmos, onchange=set_algorithm, font_size=25)

    menu.add.label("", max_char=-1, font_size=10)    
    menu.add.range_slider('Número de Iteraciones:', 550, (100, 900), 1, 
                        rangeslider_id='itera',
                        value_format=lambda x: str(int(x)),
                        font_size=25, font_color=(0, 0, 0))
           
    menu.add.range_slider('Probabilidad de éxito:', 90, (50, 100), 1,
                        rangeslider_id='success',value_format=lambda x: str(int(x)),
                        font_size=25, font_color=(0, 0, 0))


    menu.add.label("", max_char=-1, font_size=10)

    # Start Simulation button
    menu.add.button(
    'Iniciar', 
    lambda: start_labyrinth(menu.get_input_data()['itera'],  menu.get_input_data()['success']), 
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
        lambda: start_labyrinth(menu.get_input_data()['itera'],  menu.get_input_data()['success']), 
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
# ----------------------------------------------------------------------------------------------------------------


def main() -> None:
    aMapa = get_map()
    nRows = 7                                 # Cantidad de filas
    nCols = 9                                 # Cantidad de columnas
    nStates = nCols * nRows                   # Cantidad de estados
    nActions = 4                              # Posibles acciones
    aActions = ['N', 'S', 'E', 'W']           # Acciones
    Ld = 0.9                                  # Factor de descuento
    sMeta = 48                                # Estado Meta
    
    aQ = np.zeros((nStates, nActions))        # Matriz de Q-Valores
    aP = np.zeros((1, nStates))               # Matriz de Políticas
    aE = [np.ones((1, nStates))]              # Matriz de funciones de valor
    aT = []                                   # Matriz de matrices de transición

    # Matriz transicion de cada acción [N, S, E, W]
    for a in aActions:
        aT.append(mTransition(aMapa, a, nStates, nRows, nCols))

    
    Interface(aMapa, nStates, nActions, Ld, sMeta, aQ, aP, aE, aT)
    
if __name__ == "__main__":
    main()

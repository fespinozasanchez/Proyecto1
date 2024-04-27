#%%
import numpy as np
import json

# Matriz del mapa (Laberinto del robot) donde 1 es camino y -1 es obstáculo
Mapa = [
    [1,  1,  1,  1,  1,  1,  1, -1, -1],
    [1, -1, -1,  1, -1, -1,  1, -1, -1],
    [1,  1, -1,  1, -1, -1,  1, -1, -1],
    [-1, 1,  1, -1,  1,  1,  1,  1,  1],
    [1,  1, -1,  1,  1, -1,  1, -1, -1],
    [1, -1, -1, -1, -1, -1,  1, -1, -1],
    [1,  1,  1,  1, -1, -1, -1, -1, -1]
]

estado_array = np.array(Mapa)
num_filas, num_cols = estado_array.shape
n_estados = num_filas * num_cols

# Inicialización de matrices de transición
T = {dir: np.zeros((n_estados, n_estados)) for dir in ['norte', 'sur', 'este', 'oeste']}

# Probabilidad de éxito y error
prob_exito = 0.9
prob_error = 0.1

# Rellenar matriz de recompensas y transiciones
for i in range(num_filas):
    for j in range(num_cols):
        estado_actual = i * num_cols + j
        if estado_array[i, j] == -1:
            continue  # No acciones desde un obstáculo
        acciones_posibles = []
        for dir, vec in zip(['norte', 'sur', 'este', 'oeste'], [(-1, 0), (1, 0), (0, 1), (0, -1)]):
            ni, nj = i + vec[0], j + vec[1]
            if 0 <= ni < num_filas and 0 <= nj < num_cols and estado_array[ni, nj] != -1:
                estado_siguiente = ni * num_cols + nj
            else:
                estado_siguiente = estado_actual  # Se queda en el mismo lugar si el movimiento es inválido
            acciones_posibles.append((dir, estado_siguiente))

        # Aplicar probabilidad de éxito y error correctamente
        for dir, estado_siguiente in acciones_posibles:
            T[dir][estado_actual, estado_siguiente] += prob_exito
            for other_dir, other_estado in acciones_posibles:
                if other_dir != dir:
                    T[other_dir][estado_actual, estado_siguiente] += prob_error / 3  # Dividir error entre las otras tres direcciones

        # Normalizar filas para que la suma sea 1
        for dir in T:
            suma = np.sum(T[dir][estado_actual, :])
            if suma != 0:
                T[dir][estado_actual, :] /= suma

# Convertir las matrices numpy a listas y guardarlas en un archivo JSON
matrices_dict = {dir: matriz.tolist() for dir, matriz in T.items()}
with open('matrices_transicion.json', 'w') as file:
    json.dump(matrices_dict, file, indent=4)


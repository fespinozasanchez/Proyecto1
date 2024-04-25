#%%
import numpy as np

# Matriz de estados como la proporcionada en la tabla
estado_tabla = [
    [0, 1, 2, 3, 4, 5, 6, -1, -1],
    [7, -1, -1, 8, -1, -1, 9, -1, -1],
    [10, 11, -1, 12, -1, -1, 13, -1, -1],
    [-1, 15, 16, -1, 18, 19, 20, 21, 22],
    [23, 24, -1, 25, 26, -1, 27, -1, -1],
    [28, -1, -1, -1, -1, -1, 30, -1, -1],
    [31, 32, 33, 34, -1, -1, -1, -1, -1]
]

estado_array = np.array(estado_tabla)
num_filas, num_cols = estado_array.shape
n_estados = 35

# Inicialización de matrices de transición
T = {dir: np.zeros((n_estados, n_estados)) for dir in ['norte', 'sur', 'este', 'oeste']}
prob_exito = 0.9
prob_fracaso = 0.1

# Función para determinar si una posición es válida y no es una muralla
def es_valido(fila, col):
    return 0 <= fila < num_filas and 0 <= col < num_cols and estado_array[fila, col] != -1

# Rellenar las matrices de transición
for fila in range(num_filas):
    for col in range(num_cols):
        estado_actual = estado_array[fila, col]
        if estado_actual == -1:
            continue
        
        # Direcciones posibles con manejo de bordes y murallas
        direcciones = {'norte': (fila - 1, col), 'sur': (fila + 1, col),
                       'este': (fila, col + 1), 'oeste': (fila, col - 1)}
        
        for direccion, (n_fila, n_col) in direcciones.items():
            if es_valido(n_fila, n_col):
                T[direccion][estado_actual, estado_array[n_fila, n_col]] = prob_exito
            T[direccion][estado_actual, estado_actual] += prob_fracaso

# Asegurar que todos los movimientos en el estado meta (29) se quedan en 29
for matriz in T.values():
    matriz[29, :] = 0
    matriz[29, 29] = 1.0

# %%

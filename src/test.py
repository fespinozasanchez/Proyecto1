#%%
import numpy as np
import pygame
import time  # Importar time para retrasos

# Inicialización de Pygame
pygame.init()
screen_size = (720, 540)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Robot Maze Simulation")
clock = pygame.time.Clock()

# Definición de colores
colors = {
    'free_path': (200, 200, 200),
    'wall': (50, 50, 50),
    'goal': (0, 255, 0),
    'robot': (255, 0, 0),
    'invisible_block': (100, 100, 100)
}

# Obtener el mapa
def get_map():
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, -2, -2],
        [1, -2, -2, 1, -2, -2, 1, -2, -2],
        [1, 1, -2, 1, -2, -2, 1, -2, -2],
        [-1, 1, 1, -1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, -2, 1, -2, -2],
        [1, -2, -2, 3, -2, -2, 1, -2, -2],
        [1, 1, 1, 1, -2, -2, -2, -2, -2]
    ])

# Dibujar el mapa
def draw_map(map_array, robot_position):
    rows, cols = map_array.shape
    cell_size = (screen_size[0] // cols, screen_size[1] // rows)
    for row in range(rows):
        for col in range(cols):
            cell_type = map_array[row][col]
            color = colors['free_path'] if cell_type == 1 else colors['wall']
            if cell_type == -2:
                color = colors['invisible_block']
            elif cell_type == 3:
                color = colors['goal']
            elif cell_type == -1:
                color = colors['wall']
            pygame.draw.rect(screen, color, (col * cell_size[0], row * cell_size[1], cell_size[0], cell_size[1]))

    # Dibujar el robot
    pygame.draw.rect(screen, colors['robot'], (robot_position[1] * cell_size[0], robot_position[0] * cell_size[1], cell_size[0], cell_size[1]))

# Posición inicial del robot
robot_position = [4, 2]

# Simulación del movimiento basado en alguna política (simplificado)
# Movimientos: 'N', 'S', 'E', 'W'
movements = ['N', 'E', 'E', 'S', 'S', 'W', 'N', 'E', 'N', 'N']

# Bucle principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill((0, 0, 0))
    draw_map(get_map(), robot_position)

    # Aplicar movimiento
    if movements:
        move = movements.pop(0)
        if move == 'N' and robot_position[0] > 0:
            robot_position[0] -= 1
        elif move == 'S' and robot_position[0] < get_map().shape[0] - 1:
            robot_position[0] += 1
        elif move == 'E' and robot_position[1] < get_map().shape[1] - 1:
            robot_position[1] += 1
        elif move == 'W' and robot_position[1] > 0:
            robot_position[1] -= 1
        
        time.sleep(0.5)  # Retraso para visualizar el movimiento

    pygame.display.flip()
    clock.tick(60)  # Actualización de la pantalla a 60 fps

pygame.quit()

#Proyecto#1: Robótica_INFO1167
#Felipe Espinoza - Oscar Uribe
#%%
class RobotNavigation:
    def __init__(self, map_size, success_probability=0.90, discount_factor=0.98, actions=['N', 'S', 'E', 'W']):
        self.map_size = map_size  # Dimensiones del mapa
        self.success_probability = success_probability  # Probabilidad de éxito en la ejecución de la acción
        self.discount_factor = discount_factor  # Factor de descuento
        self.actions = actions # Acciones posibles para el robot
        self.state_values = [[0 for _ in range(map_size[1])] for _ in range(map_size[0])]  # Valores iniciales para cada estado

    def value_iteration_classic(self):
        # Implementa Value Iteration Clásico
        threshold = 0.001
        delta = threshold
        while delta >= threshold:
            delta = 0
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    v = self.state_values[i][j]
                    max_value = float('-inf')
                    for action in self.actions:
                        reward = -1  # Costo estándar de moverse
                        if action == 'N' and i > 0:
                            new_value = reward + self.discount_factor * self.state_values[i-1][j]
                        elif action == 'S' and i < self.map_size[0] - 1:
                            new_value = reward + self.discount_factor * self.state_values[i+1][j]
                        elif action == 'E' and j < self.map_size[1] - 1:
                            new_value = reward + self.discount_factor * self.state_values[i][j+1]
                        elif action == 'W' and j > 0:
                            new_value = reward + self.discount_factor * self.state_values[i][j-1]
                        else:
                            new_value = reward + self.discount_factor * self.state_values[i][j]
                        if new_value > max_value:
                            max_value = new_value
                    self.state_values[i][j] = max_value
                    delta = max(delta, abs(v - self.state_values[i][j]))
        return self.state_values

    # def relative_value_iteration(self):
    #     # Implementa Relative Value Iteration
    #     return

    # def gauss_siedel_value_iteration(self):
    #     # Implementa Gauss-Siedel Value Iteration
    #     return

    # def value_iteration_discounted(self):
    #     # Implementa Value Iteration Clásico con Factor de Descuento
    #     return

    # def relative_value_iteration_discounted(self):
    #     # Implementa Relative Value Iteration con Factor de Descuento
    #     return

    # def q_value_iteration_classic(self):
    #     # Implementa Q-Value Iteration Clásico
    #     return

    # def q_value_iteration_discounted(self):
    #     # Implementa Q-Value Iteration con Factor de Descuento
    #     return
    
# Uso de la clase para simulación
if __name__ == "__main__":
    map_size = (7, 5)
    robot_nav = RobotNavigation(map_size)
    values = robot_nav.value_iteration_classic()
    print("Values after value iteration:", values)

    # robot_nav = RobotNavigation(map_size)
    # robot_nav.value_iteration_classic()
    # robot_nav.relative_value_iteration()
    # robot_nav.gauss_siedel_value_iteration()
    # robot_nav.value_iteration_discounted()
    # robot_nav.relative_value_iteration_discounted()
    # robot_nav.q_value_iteration_classic()
    # robot_nav.q_value_iteration_discounted()
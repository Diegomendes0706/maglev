import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros do sistema
m = 1200  # Massa do trem (kg)
g = 9.81  # Aceleração gravitacional (m/s^2)
mu_0 = 4 * np.pi * 1e-7  # Permeabilidade do vácuo (H/m)
N = 1000  # Número de espiras
A_magnet = 0.05  # Área da seção transversal do ímã (m^2)
B = 1.0  # Densidade do fluxo magnético (T)
L_motor = 1000  # Comprimento do motor linear (m)
lambda_motor = 2.0  # Comprimento de onda do campo magnético (m)
rho = 1.225  # Densidade do ar (kg/m^3)
C_d = 0.3  # Coeficiente de arrasto
A = 10  # Área frontal do trem (m^2)
k_p = 1000  # Ganho proporcional do controle
k_d = 100  # Ganho derivativo do controle
z_ref = 0.1  # Altura de referência (m)
I0 = 1000  # Corrente inicial (A)
omega = 2 * np.pi / 30  # Frequência angular (rad/s)


# Equações diferenciais do sistema
def maglev_dynamics(t, y):
    x, v, z, I = y

    # Força eletromagnética de levitação
    F_lev = (mu_0 * N ** 2 * A_magnet * I ** 2) / (2 * z ** 2)

    # Resistência do ar
    F_resist = 0.5 * rho * C_d * A * v ** 2

    # Aceleração na direção do movimento
    F_prop = (B * I * L_motor * np.sin(omega * t)) / (2 * np.pi * lambda_motor)
    a_x = (F_prop - F_resist) / m

    # Aceleração na direção vertical
    a_z = (F_lev - m * g) / m

    # Controle da corrente para manter a altura desejada
    dI_dt = -k_p * (z - z_ref) - k_d * a_z

    return [v, a_x, a_z, dI_dt]


# Condições iniciais
x0 = 0.0  # Posição inicial (m)
v0 = 0.0  # Velocidade inicial (m/s)
z0 = 0.1  # Altura inicial (m)
y0 = [x0, v0, z0, I0]

# Resolvendo a equação diferencial
t_span = (0, 120)  # Simulação de 120 segundos
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(maglev_dynamics, t_span, y0, t_eval=t_eval)

# Plotando os resultados
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(sol.t, sol.y[0], label='Posição (x)')
plt.ylabel('Posição (m)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sol.t, sol.y[1], label='Velocidade (v)', color='g')
plt.ylabel('Velocidade (m/s)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(sol.t, sol.y[2], label='Altura (z)', color='r')
plt.xlabel('Tempo (s)')
plt.ylabel('Altura (m)')
plt.legend()

plt.tight_layout()
plt.show()

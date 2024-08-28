import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema Maglev
mu0 = 4 * np.pi * 1e-7  # Permeabilidade do vácuo (H/m)
N = 5000 # Número de espiras
A = 0.01  # Área da seção transversal do ímã (m^2)
m = 1000  # Massa do trem (kg)
g = 9.81  # Aceleração devido à gravidade (m/s^2)

# Ganhos do controlador PD
kp = 4500  # Ganho proporcional
kd = 200  # Ganho derivativo

# Parâmetros de simulação
z_ref = 0.1  # Altura de referência (10 cm)
t_end = 10  # Tempo final da simulação (s)
dt = 0.001  # Passo de tempo (s)


# Função que descreve o sistema de equações diferenciais
def maglev_system(t, y, I):
    z, dz = y
    F_lev = mu0 * N ** 2 * A * I ** 2 / (2 * z ** 2)
    F_g = m * g
    dzdt = (F_lev - F_g) / m
    return np.array([dz, dzdt])


# Função para simular o sistema usando Runge-Kutta de quarta ordem
def simulate_maglev_rk4(t_end, dt):
    t = np.arange(0, t_end, dt)
    z = np.zeros_like(t)  # Altura
    dz = np.zeros_like(t)  # Velocidade
    I = np.zeros_like(t)  # Corrente

    z[0] = 0.05  # Condição inicial da altura (m)
    dz[0] = 0.0  # Condição inicial da velocidade (m/s)

    for i in range(1, len(t)):
        error = z_ref - z[i - 1]
        derror = -dz[i - 1]
        I[i - 1] = kp * error + kd * derror

        if I[i - 1] < 0:
            I[i - 1] = 0

        y = np.array([z[i - 1], dz[i - 1]])
        k1 = maglev_system(t[i - 1], y, I[i - 1])
        k2 = maglev_system(t[i - 1] + 0.5 * dt, y + 0.5 * dt * k1, I[i - 1])
        k3 = maglev_system(t[i - 1] + 0.5 * dt, y + 0.5 * dt * k2, I[i - 1])
        k4 = maglev_system(t[i - 1] + dt, y + dt * k3, I[i - 1])

        y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        z[i] = y_next[0]
        dz[i] = y_next[1]

    return t, z


# Simular o sistema usando RK4
t, z = simulate_maglev_rk4(t_end, dt)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(t, z, label='Altura do Trem (z)')
plt.axhline(y=z_ref, color='r', linestyle='--', label='Altura de Referência (0.1 m)')
plt.xlabel('Tempo (s)')
plt.ylabel('Altura (m)')
plt.title('Simulação da Altura do Trem Maglev com Controle PD usando RK4')
plt.legend()
plt.grid(True)
plt.show()

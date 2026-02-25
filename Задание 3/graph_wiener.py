import numpy as np
import matplotlib.pyplot as plt

# параметры
T = 1.0
dt = 1e-4
M = 1000  # число траекторий

N = int(T / dt)  # число шагов
t = np.linspace(0.0, T, N + 1)

# приращения: dB ~ N(0, dt)
dB = np.sqrt(dt) * np.random.randn(M, N)

# B0 = 0, дальше накопление суммы
B = np.zeros((M, N + 1))
B[:, 1:] = np.cumsum(dB, axis=1)

# график
plt.figure(figsize=(12, 6))
for i in range(M):
    plt.plot(t, B[i], linewidth=0.3)

plt.title(f"{M} trajectories of Wiener process, T={T}, dt={dt}")
plt.xlabel("t")
plt.ylabel("B(t)")
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

#Parametros globales
num_celulas = 50
limx = 10.0
limy = 10.0
radio = 0.5
dt = 0.05
m = 1.0
sigma_pos = 0.0

# Parámetros libres 
V0 = 1.0
u0 = 1.0
sigma_angulo = 1.0
w = 1.0

# Se crean los arreglos para los datos
pos_x = np.random.uniform(0, limx, num_celulas)
pos_y = np.random.uniform(0, limy, num_celulas)
angulos = np.where(np.arange(num_celulas) < num_celulas/2, 0.0, np.pi)

fuerza_x = np.zeros(num_celulas)
fuerza_y = np.zeros(num_celulas)
alineacion = np.zeros(num_celulas)
vel_x = np.zeros(num_celulas)
vel_y = np.zeros(num_celulas)
posx_real = np.copy(pos_x)
posy_real = np.copy(pos_y)

# Se configuran los graficos
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.40) 

ax.set_xlim(0, limx)
ax.set_ylim(0, limy)
ax.set_aspect('equal')
ax.set_title("Animación: Células en Contraflujo", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

scatter = ax.scatter(pos_x, pos_y, s=400, c='blue', edgecolors='black', zorder=3)

# Sliders
ax_v0 = plt.axes([0.25, 0.27, 0.60, 0.03])
ax_u0 = plt.axes([0.25, 0.20, 0.60, 0.03])
ax_ruido = plt.axes([0.25, 0.13, 0.60, 0.03])
ax_w = plt.axes([0.25, 0.06, 0.60, 0.03])

slider_v0 = Slider(ax_v0, 'Velocidad ($V_0$)', 0.0, 5.0, valinit=V0)
slider_u0 = Slider(ax_u0, 'Repulsión ($u_0$)', 0.0, 5.0, valinit=u0)
slider_ruido = Slider(ax_ruido, r'Ruido Ang. ($\sigma$)', 0.0, 5.0, valinit=sigma_angulo)
slider_w = Slider(ax_w, 'Alineación ($w$)', 0.0, 5.0, valinit=w)

def actualizar_parametros(val):
    global V0, u0, sigma_angulo, w
    V0 = slider_v0.val
    u0 = slider_u0.val
    sigma_angulo = slider_ruido.val
    w = slider_w.val

slider_v0.on_changed(actualizar_parametros)
slider_u0.on_changed(actualizar_parametros)
slider_ruido.on_changed(actualizar_parametros)
slider_w.on_changed(actualizar_parametros)

#Mismas funciones del otro codigo
def fuerzas(): 
    global pos_x, pos_y, fuerza_x, fuerza_y, alineacion
    fuerza_x.fill(0)
    fuerza_y.fill(0)
    alineacion.fill(0)

    size_grilla = 2*radio 

    columnas = int(limx / size_grilla)
    filas = int(limy / size_grilla)

    for i in range(num_celulas):
        grid_x1 = int(pos_x[i] / size_grilla)
        grid_y1 = int(pos_y[i] / size_grilla)
        for j in range(num_celulas):
            if i != j:
                grid_x2 = int(pos_x[j] / size_grilla)
                grid_y2 = int(pos_y[j] / size_grilla)

                dist_grillax = abs(grid_x1 - grid_x2)
                dist_grillay = abs(grid_y1 - grid_y2)

                if dist_grillax > columnas / 2:
                    dist_grillax = columnas - dist_grillax
                if dist_grillay > filas / 2:
                    dist_grillay = filas - dist_grillay

                if dist_grillax > 1 or dist_grillay > 1:  
                    continue

                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]

                if dx > limx / 2:
                    dx -= limx
                elif dx < -limx / 2:
                    dx += limx
                
                if dy > limy / 2:
                    dy -= limy
                elif dy < -limy / 2:
                    dy += limy

                distancia = np.sqrt(dx**2 + dy**2)
                if distancia < 2*radio and distancia > 0:
                    fuerza_x[i] += (2*radio - distancia) * dx / distancia
                    fuerza_y[i] += (2*radio - distancia) * dy / distancia
            
                    alineacion[i] += w * np.sin(m * (angulos[j] - angulos[i]))

def actu_angulo():
    global angulos
    for i in range(num_celulas):
        gauss_angulo = np.random.normal(0, 1)
        angulos[i] = angulos[i] + (alineacion[i] * dt) + sigma_angulo * gauss_angulo * np.sqrt(dt) 
    angulos = angulos % (2 * np.pi)

def actu_pos_angulo():
    global pos_x, pos_y, posx_real, posy_real, vel_x, vel_y
    for i in range(num_celulas):
        gauss_x = np.random.normal(0, 1)
        gauss_y = np.random.normal(0, 1)

        direc_x = np.cos(angulos[i])
        direc_y = np.sin(angulos[i])
        
        dx = u0 * dt * fuerza_x[i] + sigma_pos * np.sqrt(dt) * gauss_x + V0 * direc_x * dt
        dy = u0 * dt * fuerza_y[i] + sigma_pos * np.sqrt(dt) * gauss_y + V0 * direc_y * dt
        
        vel_x[i] = dx / dt
        vel_y[i] = dy / dt
        
        posx_real[i] = posx_real[i] + dx
        posy_real[i] = posy_real[i] + dy
        
        pos_x[i] = pos_x[i] + dx
        pos_y[i] = pos_y[i] + dy
        
        pos_x[i] = pos_x[i] % limx
        pos_y[i] = pos_y[i] % limy

#Animacion
def update_frame(frame):
    fuerzas()
    actu_angulo()
    actu_pos_angulo()
    
    # Se actualiza el grafico
    scatter.set_offsets(np.column_stack((pos_x, pos_y)))
    return scatter,

# Se anima
ani = FuncAnimation(fig, update_frame, frames=1000, interval=40, blit=False)

plt.show()
"""
Qiu_elec446_a1_f25.py
Author: Yusen Qiu

Adapted from:
Example control_approx_linearization.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# Need to be in .venv with numpy, matplotlib, and scipy

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mobotpy.models import FourWheelSteered
from mobotpy.integration import rk_four

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 20.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# Pre-compute the desired trajectory


# initial condition and trajectory parameters

x_init = 0
y_init = 0
vel_desired = 5.55556
theta_desired = np.pi/4
phi_desired = 0
basetoWheel = 1.5

x_d = np.zeros((4, N))
u_d = np.zeros((2, N))
for k in range(0, N):
    x_d[0, k] = x_init + (vel_desired * t[k] * np.cos(theta_desired))
    x_d[1, k] = y_init + (vel_desired * t[k] * np.sin(theta_desired))
    x_d[2, k] = theta_desired
    x_d[3, k] = 0
    u_d[0, k] = vel_desired
    u_d[1, k] = 0

# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL_T = 1.0
ELL_W = basetoWheel * 2
# Create a vehicle object of type DiffDrive
vehicle = FourWheelSteered(ELL_W, ELL_T)

# %%
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(4)
x_init[0] = 0
x_init[1] = 0
x_init[2] = (3*np.pi)/4
x_init[3] = 0

#Locality Counter-example
# x_init = np.zeros(4)
# x_init[0] = -5000
# x_init[1] = 20
# x_init[2] = (3*np.pi)/4
# x_init[3] = (np.pi)/4



# Setup some arrays
x = np.zeros((4, N))
u = np.zeros((2, N))
x[:, 0] = x_init


for k in range(1, N):

    # Simulate the differential drive vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Compute the approximate linearization
    A = np.array(
        [
            [0, 0, -vel_desired * np.cos(phi_desired) * np.sin(theta_desired), -vel_desired * np.sin(phi_desired) * np.cos(theta_desired)],
            [0, 0, vel_desired * np.cos(phi_desired) * np.cos(theta_desired), -vel_desired * np.sin(phi_desired)* np.sin(theta_desired)],
            [0, 0, 0, (1/basetoWheel) * vel_desired * np.cos(phi_desired)],
            [0, 0, 0, 0],
        ]
    )
    B = np.array([[np.cos(phi_desired) * np.cos(theta_desired), 0], 
                  [np.cos(phi_desired) * np.sin(theta_desired), 0], 
                  [(1/basetoWheel) * np.sin(phi_desired), 0], 
                  [0, 1]])

    # Compute the gain matrix to place poles of (A - BK) at p
    p = np.array([-0.8, -3.2, -0.87, -2.8])
    K = signal.place_poles(A, B, p)

    # Compute the controls (v, omega) and convert to wheel speeds (v_L, v_R)
    u[:, k] = -K.gain_matrix @ (x[:, k - 1] - x_d[:, k - 1]) + u_d[:, k]

# %% 
# MAKE PLOTS

# Change some plot settings (optional)
plt.rc("text", usetex=False)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(10.4)
ax1a = plt.subplot(611)
plt.plot(t, x_d[0, :], "C1--")
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(612)
plt.plot(t, x_d[1, :], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(613)
plt.plot(t, x_d[2, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(614)
plt.plot(t, x_d[3, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[3, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel(r"$\phi$ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1e = plt.subplot(615)
plt.step(t, u[0, :], "C2", where="post", label="$v_F$")
plt.grid(color="0.95")
plt.ylabel(r"$v_F$ [m/s]")
plt.xlabel(r"t [s]")
ax1f = plt.subplot(616)
plt.step(t, u[1, :], "C3", where="post", label="$v_2$")
plt.grid(color="0.95")
plt.ylabel(r"$\frac{d\phi}{dt}$ [rad/s]")
plt.xlabel(r"t [s]")

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = vehicle.draw(x[:,0])
plt.fill(X_BL, Y_BL, "k")
plt.fill(X_BR, Y_BR, "k")
plt.fill(X_FL, Y_FL, "k")
plt.fill(X_FR, Y_FR, "k")
plt.fill(X_BD, Y_BD, "C2", alpha=0.8, label="Start")
X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = vehicle.draw(x[:,N-1])
plt.fill(X_BL, Y_BL, "k")
plt.fill(X_BR, Y_BR, "k")
plt.fill(X_FL, Y_FL, "k")
plt.fill(X_FR, Y_FR, "k")
plt.fill(X_BD, Y_BD, "C3", alpha=0.8, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig2.pdf")

# Show the plots to the screen
plt.show()

# %%
# MAKE AN ANIMATION

# Create the animation
ani = vehicle.animate(x, T)

# Create and save the animation
# ani = vehicle.animate_trajectory(
#     x, x_d, T, True, "../agv-book/gifs/ch4/control_approx_linearization.gif"
# )

# Show animation in HTML output if you are using IPython or Jupyter notebooks
from IPython.display import display

plt.rc("animation", html="jshtml")
display(ani)
plt.close()

# %%

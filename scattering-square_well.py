import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import sys


# Input parameter
try:
    k0b = float(sys.argv[1])
except IndexError:
    k0b = 5*np.pi/2+0.1


# Parameters
b = 1.0
k0 = k0b / b  # k0 = (2*m*V0 / hbar**2)**0.5
V0 = k0**2 / 2.0
a = b - np.tan(k0b) / k0

# Scattering solution
u_lin = lambda r: k0*np.cos(k0)*(r-a)
u_s = lambda r: np.where(r > b, u_lin(r), np.sin(k0*r))

# Bound solution
interval = np.round((k0b-1e-11) / np.pi)
if interval > 0:
    k_b = opt.root_scalar(lambda k: np.sqrt(2*V0-k**2) + k / np.tan(k), method='bisect',
                          bracket=(np.pi*(interval-0.5)+1e-6, min(np.pi*interval, (2*V0)**0.5)-1e-6)).root
else:
    k_b = 0
u_inside = lambda r: np.sin(k_b*r)
u_outside = lambda r: np.exp(-(2.0*V0-k_b**2)**0.5*(r-b)) * np.sin(k_b*b)
u_b = lambda r: np.where(r > b, u_outside(r), u_inside(r))

# Bounded State Energy Levels
ene = []
for i in range(1, int(interval)+1):
    k = opt.root_scalar(lambda k: np.sqrt(2*V0-k**2) + k / np.tan(k), method='bisect',
                        bracket=(np.pi*(i-0.5)+1e-6, min(np.pi*i, (2*V0)**0.5)-1e-6)).root
    ene.append(0.5*k**2-V0)
print("> Energy levels:")
print(''.join([f"> E_{i} / V0 = {e / V0}\n" for i, e in enumerate(ene)])) 

# Plotting
x = np.linspace(0, 4.0, 1000*min(int(k0b/np.pi), 1))
xh = np.linspace(-1.0, 2.0)

plt.hlines(0, -0.5, 4.0)
plt.vlines(0, -1.5, 3.0)
plt.vlines(b, -1, 0)
plt.hlines(-1, 0, b)
for i, e in enumerate(ene):
    plt.hlines(e/V0, 0, b, colors='red')

plt.plot(xh, u_lin(xh), color='gray', linestyle=':')
plt.plot(x, u_s(x), color='blue', linestyle='-')
plt.plot(x, u_b(x), color='green', linestyle='--')

plt.grid()
plt.xlim(-0.5, 4.0)
plt.ylim(-1.5, 3.0)
plt.show()

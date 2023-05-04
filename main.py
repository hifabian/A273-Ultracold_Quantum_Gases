import numpy as np
import matplotlib.pyplot as plt


## Parameters
# Spatial discretization
Ns = 2**8
a = 12
xi, xf = (-2*a**0.5, 2*a**0.5)
# Temporal discretization
nsteps = 10000
dt = 0.0001
# Phyiscal parameters
N = 1e3
g = 1.0
Omega = 0.0

# Trap
def Vfunc(x):
    x2 = np.sum([xd**2 for xd in x], axis=0)
    return -a*x2 + x2**2


def create_grid(xi, xf, Ns):
    x, dx = np.linspace(xi, xf, Ns, retstep=True)
    p = 2*np.pi*np.fft.fftfreq(Ns, d=dx)
    dp = p[1]-p[0]
    x = np.meshgrid(x, x)
    p = np.meshgrid(p, p)
    return x, dx, p, dp


def H(K, V, g, Omega, x, p, psi):
    Hpsi = np.fft.ifftn(K*np.fft.fftn(psi))  # Kinetic
    Hpsi += (V+g*np.abs(psi)**2)*psi  # Trap
    Hpsi += np.fft.ifftn(Omega*x[1]*p[0]*np.fft.fftn(psi, axes=(1,)), axes=(1,))
    Hpsi += np.fft.ifftn(-Omega*x[0]*p[1]*np.fft.fftn(psi, axes=(0,)), axes=(0,))
    return  Hpsi


def energy(K, V, g, Omega, x, p, psi, dx):
    return dx*np.real(np.vdot(psi, H(K, V, g, Omega, x, p, psi)))


def solve(K, V, g, Omega, x, dx, p, dp, psi0, dt, nsteps):
    t = np.zeros(nsteps)
    e = np.zeros(nsteps)
#    psi = np.zeros((nsteps,)+psi0.shape, dtype=complex)
    Up = np.exp(-0.5*dt*K)
    Uxpy = np.exp(-0.5*dt*Omega*x[1]*p[0])
    Uypx = np.exp(dt*Omega*x[0]*p[1])
    psi = np.copy(psi0)
    e[0] = energy(K, V, g, Omega, x, p, psi, dx)
    for i in range(1, nsteps):
        psi_old = psi
        # Trap+Non-Linearity (half)
        Ux = np.exp(-0.5*dt*(V+g*np.abs(psi)**2))
        psi *= Ux
        # Kinetic Energy (half)
        psi = np.fft.fftn(psi)
        psi *= Up
        psi = np.fft.ifftn(psi, axes=(0,))
        # Rotation (xpy) (half)
        psi *= Uxpy
        psi = np.fft.ifftn(psi, axes=(1,))
        # Rotation (ypx) (full)
        psi = np.fft.fftn(psi, axes=(0,))
        psi *= Uypx
        psi = np.fft.ifftn(psi, axes=(0,))
        # Rotation (xpy) (half)
        psi = np.fft.fftn(psi, axes=(1,))
        psi *= Uxpy
        # Kinetic Energy (half)
        psi = np.fft.fftn(psi, axes=(0,))
        psi *= Up
        psi = np.fft.ifftn(psi)
        # Trap+Non-Linearity (half)
        psi *= Ux #np.exp(-0.5*dt*(V+g*np.abs(psi)**2))
        # Normalization
        psi /= np.trapz(np.trapz(np.abs(psi)**2, dx=dx), dx=dx)**0.5
        e[i] = energy(K, V, g, Omega, x, p, psi, dx)
        t[i] = t[i-1]+dt

    return psi, e, t


def get_flux(psi):
    return 0.5j*(psi*np.gradient(np.conjugate(psi))-np.conjugate(psi)*np.gradient(psi))


if __name__ == '__main__':
    x, dx, p, dp = create_grid(xi, xf, Ns)
    psi0 = np.exp(-0.5*(x[0]**2+x[1]**2))  # initial guess
    #psi0 *= x[0]**10+x[1]**10
    #psi0 *= np.random.rand(Ns, Ns)
    psi0 /= (np.real(np.vdot(psi0, psi0))*dx*dx)**0.5
    V = Vfunc(x)
    K = 0.5*np.sum([pd**2 for pd in p], axis=0)

    psi, e, t = solve(K, V, N*g, Omega, x, dx, p, dp, psi0, dt, nsteps)

    print(f"ENERGY = {e[-1]}")
    print(f"ERROR (WFC)  = {(dx*np.sum(np.abs(H(K, V, g, Omega, x, p, psi)-e[-1]*psi)**2)/e[-1]**2)**0.5}")
    print(f"ERROR (Norm) = {dx*np.sum(np.abs(np.abs(H(K, V, g, Omega, x, p, psi))**2-np.abs(e[-1]*psi)**2))/e[-1]**2}")

    plt.plot(t, e)
    plt.show()
    plt.imshow(np.abs(psi)**2)
    plt.show()
    plt.imshow(np.abs(H(K, V, g, Omega, x, p, psi))**2)
    plt.show()
    plt.imshow(np.angle(psi))
    plt.show()
    #plt.streamplot(*x, *np.real(get_flux(psi)))
    #plt.show()
    #plt.streamplot(*x, *np.gradient(np.angle(psi)))
    #plt.show()
#%%
from numpy import *
from numpy.fft import  fft, ifft
from numpy.linalg import norm
from matplotlib.pyplot import *

def simulation(potential, initialvalue):
    scale = pi

    # Time interval, step size
    tend = 5.0*sqrt(ieps)
    nl = 1
    n = int(tend * 10**nl)
    print("Doing %d timesteps" % n)
    h = tend/n
    ih = 1j*h

    # place for norms
    normsol = zeros(n, dtype=floating)
    cinE = zeros(n)
    potE = zeros(n)

    # Space discretization
    l = 12
    N = 2**l
    
    # Laplacian
    A = 0.5 * eps * (concatenate((arange(0,N//2,1),arange(-N//2,0,1))))**2

    #  Potential
    x = scale * arange(-1, 1, 2.0/N)
    V = ieps * potential(x)

    v = initialvalue(x)
    u = fft(v)
    normsol[0] = sqrt(2*pi) * norm(u)/N
    print("norm: %f" % normsol[0])

    # Exponentials
    cinE[0] = real(eps * dot(conj(u), (A*u))/N**2 * (2.0*pi))
    potE[0] = real(eps * dot(conj(u), fft(V*v)/N**2) * (2.0*pi))
    print("E kin: %f" % cinE[0])
    print("E pot: %f" % potE[0])
    print("E tot: %f" % (cinE[0]+potE[0]))

    plot(x, v, V, 0, h)
    # Time stepping

    for k in range(1, n):
        print("Timestep: %d" % k)

        # Propagate
        v = ifft(u) * exp(- 0.5 * ih * V) # First half step
        u = fft(v) * exp(- ih * A) # Full step
        v = ifft(u) * exp(- 0.5 * ih * V) # Second half step

        # Compute norm
        u = fft(v)
        normsol[k] = sqrt(2*pi) * norm(u)/N
        print("norm: %f" % normsol[k])

        # Compute energies
        cinE[k] = real(eps * dot(conj(u), A*u)/N**2 * (2.0*pi))
        potE[k] = real(eps * dot(conj(u), fft(V*v)/N**2) * (2.0*pi))

        print("E kin: %f" % cinE[k])
        print("E pot: %f" % potE[k])
        print("E tot: %f" % (cinE[k]+potE[k]))
  
        #plot(x, v, V, k, h)

    fig = figure()
    ax = fig.gca()
    ax.plot(h*arange(n), normsol)
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Norm $\| u \|$")
    #fig.savefig("norm.png")
    close(fig)

    fig = figure()
    ax = fig.gca()
    ax.plot(h*arange(n), cinE, label=r"$E_{kin}$")
    ax.plot(h*arange(n), potE, label=r"$E_{pot}$")
    ax.plot(h*arange(n), cinE+potE, label=r"$E_{tot}$")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Energy $E$")
    ax.set_title("Energies for step potential and anfangswert g0")
    #fig.savefig("energies_step_g0.png")
    show()
    plot(x, v, V, k, h)
    close(fig)


def plot(x, v, V, k, h, ymin=-3, ymax=3):
    fig = figure(figsize=(10,10))
    ax = fig.gca()
    ax.fill_between(x, V*eps, ymin, color="k", alpha=0.2)
    ax.plot(x, V*eps,"k-", label=r"$V(x)$")
    ax.plot(x, real(v), "b-", label=r"$\Re u(t)$",linewidth = 1)
    ax.plot(x, imag(v), "g-", label=r"$\Im u(t)$",linewidth = 1)
    ax.plot(x, abs(v), "r-", label=r"$|u(t)|$")
    ax.legend(loc='lower right')
    ax.set_xlabel(r"$x$")
    ax.set_title("Time t=%f" % (k*h))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ymin, ymax)
    #fig.savefig("solution_at_timestep %04d.png" % k)
    show()
    close(fig)




if __name__ == "__main__":
    # Parameter
    eps = 0.01
    ieps = 1.0/eps

    # Potentials
    def morse(x, V0=8, beta=0.25):
        x = x + 1.0
        V = V0*(exp(-2.0*beta*x) - 2.0*exp(-beta*x))
        return V0 + V

    def harmonic(x):
        V = 0.5*(x**2)
        return V
    

    def sombrero(x, k=1.0, r0=1.0):
        return 0.5 * k * (sqrt(x**2) - r0)**2
    
    def step(x, V0=1):
        return np.where((x >= 0) & (x <= 1), V0, 0)



    #  Initial value
    g0 = lambda x: (ieps/pi)**(0.25) * exp(-(0.5*ieps)*(x+0.5)**2) * exp(-1j*x*ieps)
    g1 = lambda x: (ieps/pi)**(0.25) * exp(-(0.5*ieps)*x**2)

    simulation(harmonic, g0)

# %%

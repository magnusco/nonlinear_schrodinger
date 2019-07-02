# import matplotlib
# matplotlib.use('Agg')

import time
import numpy as np
import pickle
import math as m
from scipy.sparse import linalg
from scipy import sparse
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


class NSE:

    def __init__(self, interval, lmbda, M, N, T, analytic=None, initial=None, filename="saved_solution"):
        self.space = interval
        self.lmbda = lmbda
        self.M = M
        self.N = N
        self.T = T
        self.h = (interval[1] - interval[0]) / M
        self.k = T / N
        self.r = self.k / self.h**2 + 0.0j
        self.xi = np.linspace(interval[0], interval[1], self.M + 1)
        self.ti = np.linspace(0, T, self.N + 1)
        self.sol = np.zeros((self.N + 1, self.M + 1), dtype=np.complex)
        self.filename = filename
        self.exact = np.zeros((self.N + 1, self.M + 1), dtype=np.complex)
        self.analytic_expr = analytic
        self.initial_expr = initial


    # Functions:
    # ------------------------------------------------------------------------------------------------------------------
    def write_sol_to_file(self):
        with open(self.filename, 'wb') as f:
            pickle.dump([self.sol, self.xi, self.ti, self.M, self.N, self.T], f)
        return 0

    def read_sol_from_file(self, name):
        with open(self.filename, 'rb') as f:
            sol, xi, ti, M, N, T = pickle.load(f)
        return sol, xi, ti, M, N, T

    def set_new_params(self, M=None, N=None, T=None):
        if M != None:
            self.M = M
        if N != None:
            self.N = N
        if T != None:
            self.T = T
        self.h = (self.space[1] - self.space[0]) / self.M
        self.k = self.T / self.N
        self.r = self.k / self.h ** 2 + 0.0j
        self.xi = np.linspace(self.space[0], self.space[1], self.M + 1)
        self.ti = np.linspace(0, self.T, self.N + 1)
        self.sol = np.zeros((self.N + 1, self.M + 1), dtype=np.complex)
        self.exact = np.zeros((self.N + 1, self.M + 1), dtype=np.complex)
        return 0

    def tridiag(self, a, b, c, n):
        e = np.ones(n)
        return a * np.diag(e[1:], -1) + b * np.diag(e) + c * np.diag(e[1:], 1)

    def initial(self, x):
        if self.initial_expr != None:
            return self.initial_expr(x, self.lmbda)
        else:
            return self.analytic(x, np.zeros(np.shape(x)))

    def analytic(self, x, t):
        if self.analytic_expr != None:
            return self.analytic_expr(x, t, self.lmbda)
        else:
            return 0

    def calculate_analytic(self):
        X, T = np.meshgrid(self.xi, self.ti)
        self.exact = self.analytic(X, T)
        return 0

    def plot_solution(self, sol, xi, ti, title=None):
        fig = plt.figure(2)
        plt.clf()
        ax = fig.gca(projection='3d')
        X, T = np.meshgrid(xi, ti)
        ax.plot_wireframe(X, T, sol)
        ax.plot_surface(X, T, sol, cmap=plt.get_cmap('coolwarm'))
        ax.view_init(azim=40)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(title)
        plt.show()

    def plot_analytic(self):
        self.plot_solution(np.real(self.exact), self.xi, self.ti, "Real Part of Analytic Solution")
        self.plot_solution(np.imag(self.exact), self.xi, self.ti, "Imaginary Part of Analytic Solution")
        self.plot_solution(np.absolute(self.exact), self.xi, self.ti, "Absolute Value of Analytic Solution")
        return 0

    def plot(self):
        self.plot_solution(np.real(self.sol), self.xi, self.ti, "Real Part of Numerical Solution")
        self.plot_solution(np.imag(self.sol), self.xi, self.ti, "Imaginary value of solution")
        self.plot_solution(np.absolute(self.sol), self.xi, self.ti, "Absolute Value of Numerical Solution")
        return 0

    def plot_2D_final(self):
        plt.plot(self.xi, np.real(self.sol[-1]), 'b', label="approximation")
        plt.plot(self.xi, np.real(self.exact[-1]), 'r', label="exact")
        plt.grid(True)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$U_{m}^{N+1}$")
        plt.legend()
        plt.show()

    def animate_solution(self, exact=False):
        fig = plt.figure()
        ax = plt.axes(xlim=(self.space[0], self.space[1]), ylim=(-1, 3))
        if exact == True:
            line_1, = ax.plot([], [], 'b', lw=2, label="numerical")
            line_2, = ax.plot([], [], 'r', lw=2, label="exact")
            def init():
                line_1.set_data([], [])
                line_2.set_data([], [])
                return line_1, line_2,
            def animate(i):
                line_1.set_data(self.xi, np.absolute(self.sol[i]))
                line_2.set_data(self.xi, np.absolute(self.exact[i]))
                return line_1, line_2,

            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=self.N, interval=10, blit=True)
        else:
            line, = ax.plot([], [], 'r', lw=2, label="numerical")
            def init():
                line.set_data([], [])
                return line,
            def animate(i):
                line.set_data(self.xi, np.absolute(self.sol[i]))
                return line,
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=self.N, interval=10, blit=True)

        plt.xlabel(r"$x$")
        plt.ylabel(r"$|U|$")
        plt.legend()
        plt.show()
        return 0

    # Numerical methods:
    # ------------------------------------------------------------------------------------------------------------------

    # Very unstable.
    def forward_euler(self, iterations=None):
        if iterations == None:
            iterations = self.N
        self.sol[0, :] = self.initial(self.xi)
        B_const = self.tridiag(1, -2, 1, self.M)
        B_const[0, -1] = 1
        B_const[-1, 0] = 1
        B_const = 1.0j * self.r * B_const
        for i in range(0, iterations):
            abs = np.absolute(self.sol[i, 0:-1])
            B = - 1.0j * self.k * self.lmbda * np.diag(abs**2 + 0.0j, 0)
            B += B_const + np.identity(self.M)
            self.sol[i + 1, 0:-1] = np.matmul(B, self.sol[i, 0:-1])
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    # Should work.
    def central_time(self):
        self.sol[0, :] = self.initial(self.xi)
        self.cn_implicit_builtin_solver(1)
        B_const = self.tridiag(1, -2, 1, self.M)
        B_const[0, -1] = 1
        B_const[-1, 0] = 1
        B_const = 1.0j * 2 * self.r * B_const
        for i in range(1, self.N):
            abs = np.absolute(self.sol[i, 0:-1])
            B = - 1.0j * 2 * self.k * self.lmbda * np.diag(abs**2 + 0.0j, 0)
            B += B_const
            self.sol[i + 1, 0:-1] = np.matmul(B, self.sol[i, 0:-1]) + self.sol[i - 1, 0:-1]
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    # Works.
    # Need even number of space steps.
    def hopscotch(self, iterations=None):
        if iterations == None:
            iterations = self.N

        self.sol[0, :] = self.initial(self.xi)
        B_const = self.tridiag(1.0j * self.r, 1 - 2 * 1.0j * self.r, 1.0j * self.r, self.M)
        B_const[0, -1] = 1.0j * self.r
        B_const[-1, 0] = 1.0j * self.r
        A_const = self.tridiag(1.0j * self.r, 0, 1.0j * self.r, self.M)
        A_const[0, -1] = 1.0j * self.r
        A_const[-1, 0] = 1.0j * self.r

        for i in range(0, iterations):
            iter_remainder = i % 2
            inv_iter_remainder = np.logical_not(iter_remainder).astype(int)

            abs_prev = np.absolute(self.sol[i, 0:-1])
            B = np.diag(abs_prev[0:-1]**2 + 0.0j, -1) + np.diag(abs_prev[1:]**2 + 0.0j, 1)
            B[0, -1] = abs_prev[-1]**2
            B[-1, 0] = abs_prev[0]**2
            B *= (- 0.5) * 1.0j * self.lmbda * self.k
            B = B[iter_remainder::2] + B_const[iter_remainder::2]
            self.sol[i + 1, iter_remainder:-1:2] = np.matmul(B, self.sol[i, 0:-1])

            abs_next = np.absolute(self.sol[i + 1, 0:-1])
            A = np.diag(abs_next[0:-1]**2 + 0.0j, -1) + np.diag(abs_next[1:]**2 + 0.0j, 1)
            A[0, -1] = abs_next[-1]**2
            A[-1, 0] = abs_next[0]**2
            A *= (- 0.5) * 1.0j * self.lmbda * self.k
            A = A[inv_iter_remainder::2] + A_const[inv_iter_remainder::2]
            self.sol[i + 1, inv_iter_remainder:-1:2] = \
                (np.matmul(A, self.sol[i + 1, 0:-1]) + self.sol[i, inv_iter_remainder:-1:2]) / (1 + 2 * 1.0j * self.r)

        self.sol[:, -1] = self.sol[:, 0]
        return 0

    # Works
    def cn_implicit_builtin_solver(self, iterations=None):
        if iterations == None:
            iterations = self.N
        self.sol[0, :] = self.initial(self.xi)
        A_const = self.tridiag(self.r / 2, 1.0j - self.r, self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A_const[0, -1] = self.r / 2
        A_const[-1, 0] = self.r / 2
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(0, iterations):
            abs_prev = np.absolute(self.sol[i, 0:-1])
            B = 0.5 * self.lmbda * self.k * np.diag(abs_prev**2 + 0.0j, 0)
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])

            def F(U):
                folded_U = U[0:self.M] + 1.0j * U[self.M:]
                abs_next = np.absolute(folded_U)
                A = -0.5 * self.lmbda * self.k * np.diag(abs_next**2 + 0.0j, 0)
                A += A_const
                a = np.matmul(A, folded_U)
                return np.concatenate((np.real(a - b), np.imag(a - b)), axis=0)

            unfolded_U = np.concatenate((np.real(self.sol[i, 0:-1]), np.imag(self.sol[i, 0:-1])), axis=0)
            opt = optimize.newton_krylov(F, unfolded_U, f_tol=1e-7)
            self.sol[i + 1, 0:-1] = opt[0: self.M] + 1.0j * np.real(opt[self.M:])
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    # Works.
    def cn_liearized_1(self):
        self.sol[0, :] = self.initial(self.xi)
        self.cn_implicit_builtin_solver(1)
        A_const = self.tridiag(self.r / 2, 1.0j - self.r, self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A_const[0, -1] = self.r / 2
        A_const[-1, 0] = self.r / 2
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(1, self.N):
            abs = np.absolute(self.sol[i, 0:-1])
            A = np.diag(3 * abs**2 - 2 * abs * np.absolute(self.sol[i - 1, 0:-1]) + 0.0j, 0)
            B = np.diag(abs**2 + 0.0j, 0)
            A *= -0.5 * self.lmbda * self.k
            B *= 0.5 * self.lmbda * self.k
            A += A_const
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])
            As = sparse.csr_matrix(A)
            self.sol[i + 1, 0:-1] = linalg.spsolve(As, b)
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    # Works.
    def cn_liearized_2(self):
        self.sol[0, :] = self.initial(self.xi)
        self.cn_implicit_builtin_solver(1)
        A = self.tridiag(self.r / 2, 1.0j - self.r, self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A[0, -1] = self.r / 2
        A[-1, 0] = self.r / 2
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2
        As = sparse.csr_matrix(A)

        for i in range(1, self.N):
            abs = np.absolute(self.sol[i, 0:-1])
            B = np.diag(5 * abs**2 + 0.0j - 2 * abs * np.absolute(self.sol[i - 1, 0:-1]) + 0.0j, 0)
            B *= 0.5 * self.lmbda * self.k
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])
            b -= 0.5 * self.lmbda * self.k * np.multiply(abs**2, self.sol[i - 1, 0:-1])
            self.sol[i + 1, 0:-1] = linalg.spsolve(As, b)
        self.sol[:, -1] = self.sol[:, 0]
        return 0

# Some functions for assessment of the numerical schemes:
# ----------------------------------------------------------------------------------------------------------------------


def convergence_order_space(analytic, lmbda):
    M = 10
    iter = 6

    central_time = NSE([-np.pi, np.pi], lmbda, M, 10000, 5, analytic=analytic)
    hopscotch = NSE([-np.pi, np.pi], lmbda, M, 50000, 5, analytic=analytic)
    cn_implicit_builtin_solver = NSE([-np.pi, np.pi], lmbda, M, 5000, 5, analytic=analytic)
    cn_linearized_1 = NSE([-np.pi, np.pi], lmbda, M, 5000, 5, analytic=analytic)
    cn_linearized_2 = NSE([-np.pi, np.pi], lmbda, M, 5000, 5, analytic=analytic)

    method_collection = [central_time, hopscotch, cn_implicit_builtin_solver, cn_linearized_1, cn_linearized_2]
    method_colors = ['b', 'r', 'g', 'cyan', 'deeppink']
    method_lables = ["Central time", "Hopscotch", "Crank-Nicolson", "Linearized CN, real", "Linearized CN, complex"]
    number_of_methods = 5

    for method in method_collection:
        method.calculate_analytic()

    errors_space = np.zeros((number_of_methods, iter))
    h = np.zeros((number_of_methods, iter))

    for i in range(0, iter):
        print(i)
        central_time.central_time()
        hopscotch.hopscotch()
        cn_implicit_builtin_solver.cn_implicit_builtin_solver()
        cn_linearized_1.cn_liearized_1()
        cn_linearized_2.cn_liearized_2()
        M *= 2
        for j in range(0, number_of_methods):
            method_collection[j].calculate_analytic()
            errors_space[j, i] = np.max(np.absolute(method_collection[j].exact - method_collection[j].sol))
            h[j, i] = method_collection[j].h
            method_collection[j].set_new_params(M)

    for j in range(0, number_of_methods):
        plt.plot(h[j], errors_space[j], method_colors[j], label=method_lables[j])

    order_2_h = np.logspace(-2, 0, 100)
    order_2_e = order_2_h**2
    plt.plot(order_2_h, order_2_e, 'k', label=r"$\mathcal{O}(h^{2})$")

    plt.yscale('log')
    plt.grid(True)
    plt.title("Convergence in Space")
    plt.xscale('log')
    plt.xlabel(r"$h$")
    plt.ylabel(r"$||e_{h}||_{max}$")
    plt.legend()
    plt.savefig('convergence_space.png')
    return 0

def convergence_order_time(analytic, lmbda):
    M = 500
    iter = 6

    hopscotch = NSE([-np.pi, np.pi], lmbda, M, 2500, 5, analytic=analytic)
    cn_implicit_builtin_solver = NSE([-np.pi, np.pi], lmbda, M, 50, 5, analytic=analytic)
    cn_linearized_1 = NSE([-np.pi, np.pi], lmbda, M, 50, 5, analytic=analytic)
    cn_linearized_2 = NSE([-np.pi, np.pi], lmbda, M, 60, 5, analytic=analytic)

    method_collection = [hopscotch, cn_implicit_builtin_solver, cn_linearized_1, cn_linearized_2]
    method_colors = ['b', 'r', 'g', 'cyan']
    method_lables = ["Hopscotch", "Crank-Nicolson", "Linearized CN, real", "Linearized CN, complex"]
    number_of_methods = 4

    errors_time = np.zeros((number_of_methods, iter))
    k = np.zeros((number_of_methods, iter))

    for method in method_collection:
        method.calculate_analytic()

    for i in range(0, iter):
        print(i)
        hopscotch.hopscotch()
        cn_implicit_builtin_solver.cn_implicit_builtin_solver()
        cn_linearized_1.cn_liearized_1()
        cn_linearized_2.cn_liearized_2()
        for j in range(0, number_of_methods):
            method_collection[j].calculate_analytic()
            errors_time[j, i] = np.max(np.absolute(method_collection[j].exact - method_collection[j].sol))
            method_collection[j].set_new_params(N=(method_collection[j].N * 2))
            k[j, i] = method_collection[j].k

    for j in range(0, number_of_methods):
        plt.plot(k[j], errors_time[j], method_colors[j], label=method_lables[j])

    order_2_k = np.logspace(-2, 0, 100)
    order_2_e = order_2_k**2
    plt.plot(order_2_k, order_2_e, 'k', label=r"$\mathcal{O}(k^{2})$")

    plt.yscale('log')
    plt.grid(True)
    plt.title("Convergence in Time")
    plt.xscale('log')
    plt.xlabel(r"$k$")
    plt.ylabel(r"$||e_{k}||_{max}$")
    plt.legend()
    plt.savefig('convergence_time.png')
    return 0


def run_time(analytic, lmbda):

    central_time = NSE([-np.pi, np.pi], lmbda, 140, 10000, 5, analytic=analytic)
    hopscotch = NSE([-np.pi, np.pi], lmbda, 140, 4000, 5, analytic=analytic)
    cn_implicit_builtin_solver = NSE([-np.pi, np.pi], lmbda, 126, 600, 5, analytic=analytic)
    cn_linearized_1 = NSE([-np.pi, np.pi], lmbda, 130, 600, 5, analytic=analytic)
    cn_linearized_2 = NSE([-np.pi, np.pi], lmbda, 200, 1000, 5, analytic=analytic)

    method_collection = [central_time, hopscotch, cn_implicit_builtin_solver, cn_linearized_1, cn_linearized_2]
    method_colors = ['b', 'r', 'g', 'cyan', 'deeppink']
    method_lables = ["Central time", "Hopscotch", "Crank-Nicolson", "Linearized CN, real", "Linearized CN, complex"]
    number_of_methods = 5

    for method in method_collection:
        method.calculate_analytic()

    times = np.zeros(number_of_methods)
    errors = np.zeros(number_of_methods)

    start_time = time.time()
    central_time.central_time()
    times[0] = time.time() - start_time
    errors[0] = np.max(np.absolute(central_time.exact - central_time.sol))

    start_time = time.time()
    hopscotch.hopscotch()
    times[1] = time.time() - start_time
    errors[1] = np.max(np.absolute(hopscotch.exact - hopscotch.sol))

    start_time = time.time()
    cn_implicit_builtin_solver.cn_implicit_builtin_solver()
    times[2] = time.time() - start_time
    errors[2] = np.max(np.absolute(cn_implicit_builtin_solver.exact - cn_implicit_builtin_solver.sol))

    start_time = time.time()
    cn_linearized_1.cn_liearized_1()
    times[3] = time.time() - start_time
    errors[3] = np.max(np.absolute(cn_linearized_1.exact - cn_linearized_1.sol))

    start_time = time.time()
    cn_linearized_2.cn_liearized_2()
    times[4] = time.time() - start_time
    errors[4] = np.max(np.absolute(cn_linearized_2.exact - cn_linearized_2.sol))

    print("Running-times:")
    print(times)
    print()
    print("Corresponding errors:")
    print(errors)
    return 0


def conservation_of_mass(analytic, lmbda):
    central_time = NSE([-np.pi, np.pi], lmbda, 130, 10000, 5, analytic=analytic)
    hopscotch = NSE([-np.pi, np.pi], lmbda, 200, 4000, 5, analytic=analytic)
    cn_implicit_builtin_solver = NSE([-np.pi, np.pi], lmbda, 200, 600, 5, analytic=analytic)
    cn_linearized_1 = NSE([-np.pi, np.pi], lmbda, 200, 600, 5, analytic=analytic)
    cn_linearized_2 = NSE([-np.pi, np.pi], lmbda, 300, 1000, 5, analytic=analytic)

    method_collection = [central_time, hopscotch, cn_implicit_builtin_solver, cn_linearized_1, cn_linearized_2]
    method_lables = ["Central time", "Hopscotch", "Crank-Nicolson", "Linearized CN, real", "Linearized CN, complex"]
    number_of_methods = 5

    central_time.central_time()
    hopscotch.hopscotch()
    cn_implicit_builtin_solver.cn_implicit_builtin_solver()
    cn_linearized_1.cn_liearized_1()
    cn_linearized_2.cn_liearized_2()

    for j in range(0, number_of_methods):
        method_collection[j].calculate_analytic()
        mass = np.zeros(method_collection[j].N + 1)
        mass_a = np.zeros(method_collection[j].N + 1)
        times = method_collection[j].ti
        h = method_collection[j].h
        for i in range(0, method_collection[j].N + 1):
            mass[i] = h * np.linalg.norm(method_collection[j].sol[i], 2)
            mass_a[i] = h * np.linalg.norm(method_collection[j].exact[i], 2)
        plt.plot(times[::10], mass_a[::10] - mass[::10], 'k')
        plt.ticklabel_format(style="sci", scilimits=(0, 0))
        plt.ylabel(r"$h( \sum |u|^2 - \sum |U|^2)$")
        plt.xlabel(r"$t$")
        plt.grid(True)
        plt.show()
    return 0
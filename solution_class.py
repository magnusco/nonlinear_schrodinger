import numpy as np
import pickle
import math as m
from scipy.sparse import linalg
from scipy import sparse
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


class Nonlin_Schrodinger_Solver:

    def __init__(self, interval, lmbda, M, N, T, filename):
        self.space = interval
        self.lmbda = lmbda
        self.M = M
        self.N = N
        self.T = T
        self.h = (interval[1] - interval[0]) / M
        self.k = T / N + 0.0j
        self.r = self.k / self.h**2 + 0.0j
        self.xi = np.linspace(interval[0], interval[1], self.M + 1)
        self.ti = np.linspace(0, T, self.N + 1)
        self.sol = np.zeros((self.N + 1, self.M + 1), dtype=np.complex)
        self.filename = filename
        self.deviation_from_analytic = 0
        self.exact = np.zeros((self.N + 1, self.M + 1), dtype=np.complex)

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

    def tridiag(self, a, b, c, n):
        e = np.ones(n)
        return a * np.diag(e[1:], -1) + b * np.diag(e) + c * np.diag(e[1:], 1)

    def f(self, x):
        return self.analytic_1(x, np.zeros(np.shape(x)))

    def analytic_1(self, x, t):
        """t += 1
        c = 1
        y = (x**2) / (4 * t) + 2 * c**2 * np.log(t)
        return (c) / (np.sqrt(t)) * np.exp(1.0j * y)"""

        # return np.exp(1.0j * (x - t * (self.lmbda + 1)))

        t -= self.T / 2
        a = 1
        b = 1
        theta = a**2 * b * np.sqrt(2 - b**2) * t
        c = (2 * b**2 * np.cosh(theta) + 2 * 1.0j * b * np.sqrt(2 - b**2) * np.sinh(theta)) / \
            (2 * np.cosh(theta) - np.sqrt(2) * np.sqrt(2 - b**2) * np.cos(a * b * x)) - 1
        return c * a * np.exp(1.0j * a**2 * t)

    def calculate_analytic(self):
        X, T = np.meshgrid(self.xi, self.ti)
        self.exact = self.analytic_1(X, T)
        return 0

    def find_deviation_from_analytic(self):
        self.deviation_from_analytic = np.max(np.absolute(self.sol - self.exact))
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
        #self.plot_solution(np.imag(sol), self.xi, self.ti, "Imaginary Part of Analytic Solution")
        return 0

    def plot(self):
        self.plot_solution(np.real(self.sol), self.xi, self.ti, "Real Part of Numerical Solution")
        #self.plot_solution(np.imag(self.sol), self.xi, self.ti, "Imaginary value of solution")
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
        ax = plt.axes(xlim=(self.space[0], self.space[1]), ylim=(-2, 2))
        if exact == True:
            line_1, = ax.plot([], [], 'b', lw=2, label="numerical")
            line_2, = ax.plot([], [], 'r', lw=2, label="exact")
            def init():
                line_1.set_data([], [])
                line_2.set_data([], [])
                return line_1, line_2,
            def animate(i):
                line_1.set_data(self.xi, np.real(self.sol[i]))
                line_2.set_data(self.xi, np.real(self.exact[i]))
                return line_1, line_2,

            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=self.N, interval=10, blit=True)
        else:
            line, = ax.plot([], [], 'r', lw=2, label="numerical")
            def init():
                line.set_data([], [])
                return line,
            def animate(i):
                line.set_data(self.xi, self.sol[i])
                return line,
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=self.N, interval=10, blit=True)

        plt.xlabel(r"$x$")
        plt.ylabel(r"$Re(U)$")
        plt.legend()
        plt.show()
        return 0

    # Numerical methods:
    # ------------------------------------------------------------------------------------------------------------------

    # Very unstable.
    def forward_euler(self, iterations=None):
        if iterations == None:
            iterations = self.N
        self.sol[0, :] = self.f(self.xi)
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
        self.sol[0, :] = self.f(self.xi)
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

    # Works, migth be a subtle error in the boundary stuff.
    # Need even number of space steps.
    def hopscotch(self, iterations=None):
        if iterations == None:
            iterations = self.N

        self.sol[0, :] = self.f(self.xi)
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
            self.sol[i + 1, inv_iter_remainder:-1:2] = (np.matmul(A, self.sol[i + 1, 0:-1]) + self.sol[i, inv_iter_remainder:-1:2]) / (1 + 2 * 1.0j * self.r)

        self.sol[:, -1] = self.sol[:, 0]
        return 0

    # Works
    def cn_implicit_builtin_solver(self, iterations=None):
        if iterations == None:
            iterations = self.N
        self.sol[0, :] = self.f(self.xi)
        A_const = self.tridiag(self.r / 2, 1.0j - self.r, self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A_const[0, -1] = self.r / 2
        A_const[-1, 0] = self.r / 2
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(0, iterations):
            abs_prev = np.absolute(self.sol[i, 0:-1])
            B = 0.5 * self.lmbda * self.k * np.diag(abs_prev**2, 0)
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])

            def F(U):
                folded_U = U[0:self.M] + 1.0j * U[self.M:]
                abs_next = np.absolute(folded_U)
                A = -0.5 * self.lmbda * self.k * np.diag(abs_next**2, 0)
                A += A_const
                a = np.matmul(A, folded_U)
                return np.concatenate((np.real(a - b), np.imag(a - b)), axis=0)

            unfolded_U = np.concatenate((np.real(self.sol[i, 0:-1]), np.imag(self.sol[i, 0:-1])), axis=0)
            opt = optimize.newton_krylov(F, unfolded_U)
            self.sol[i + 1, 0:-1] = opt[0: self.M] + 1.0j * np.real(opt[self.M:])
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    # Works.
    def cn_liearized_1(self):
        self.sol[0, :] = self.f(self.xi)
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
        self.sol[0, :] = self.f(self.xi)
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


if __name__ == '__main__':
    # forward_euler = Nonlin_Schrodinger_Solver([-np.pi, np.pi], 2, 10, 10000, 5, 'forward_euler.pkl')
    # forward_euler.forward_euler()
    # forward_euler.calculate_analytic()
    # forward_euler.plot()
    # forward_euler.plot_analytic()

    # central_time = Nonlin_Schrodinger_Solver([-np.pi, np.pi], 1, 10, 100, 5, 'central_time.pkl')
    # central_time.central_time()
    # central_time.calculate_analytic()
    # central_time.plot()
    # central_time.plot_analytic()

    # hopscotch = Nonlin_Schrodinger_Solver([-np.pi, np.pi], 2, 200, 200, 5, 'hopscotch.pkl')
    # hopscotch.hopscotch()
    # hopscotch.calculate_analytic()
    # hopscotch.plot()
    # hopscotch.plot_analytic()

    cn_implicit_builtin_solver = Nonlin_Schrodinger_Solver([-np.pi, np.pi], -1, 100, 100, 5, 'cn_implicit.pkl')
    cn_implicit_builtin_solver.cn_implicit_builtin_solver()
    cn_implicit_builtin_solver.calculate_analytic()
    cn_implicit_builtin_solver.plot()
    cn_implicit_builtin_solver.plot_analytic()
    cn_implicit_builtin_solver.animate_solution(True)

    # cn_linearized_1 = Nonlin_Schrodinger_Solver([-np.pi, np.pi], 2, 200, 200, 5, 'cn_linearized.pkl')
    # cn_linearized_1.cn_liearized_1()
    # cn_linearized_1.calculate_analytic()
    # cn_linearized_1.plot()
    # cn_linearized_1.plot_analytic()
    # cn_linearized_1.animate_solution(True)

    # cn_linearized_2 = Nonlin_Schrodinger_Solver([-np.pi, np.pi], -2, 200, 200, 5, 'cn_linearized.pkl')
    # cn_linearized_2.cn_liearized_2()
    # cn_linearized_2.calculate_analytic()
    # cn_linearized_2.plot()
    # cn_linearized_2.plot_analytic()
    # cn_linearized_2.animate_solution(True)


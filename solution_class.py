import numpy as np
import pickle
from scipy.sparse import linalg
from scipy import sparse
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Nonlin_Scrodinger_Solver:

    def __init__(self, interval, lmbda, M, N, T, filename):
        self.space = interval
        self.lmbda = lmbda
        self.M = M
        self.N = N
        self.T = T
        self.h = (interval[1] - interval[0]) / M
        self.k = T / N
        self.r = self.k / self.h**2
        self.xi = np.linspace(interval[0], interval[1], self.M + 1)
        self.ti = np.linspace(0, T, self.N + 1)
        self.sol = np.zeros((self.N + 1, self.M + 1), dtype=np.complex)
        self.filename = filename
        self.deviation_from_analytic = 0

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

    def f1(self, x):
        # return np.sin(x)
        return np.exp(1.0j * x)

    def analytic_1(self, x, t):
        X, T = np.meshgrid(x, t)
        return np.exp(1.0j * (X - (self.lmbda) * T))

    def find_deviation_from_analytic(self):
        U_exact = self.analytic_1(self.xi, self.ti)
        self.deviation_from_analytic = np.max(np.absolute(self.sol - U_exact))
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
        sol = self.analytic_1(self.xi, self.ti)
        self.plot_solution(np.real(sol), self.xi, self.ti, "Real Part of Analytic Solution")
        self.plot_solution(np.imag(sol), self.xi, self.ti, "Imaginary Part of Analytic Solution")
        return 0

    def cn_explicit_average(self):
        self.sol[0, :] = self.f1(self.xi)
        A = self.tridiag(self.r / 2, 1.0j - self.r, self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A[0, -1] = self.r / 2
        A[-1, 0] = self.r / 2
        As = sparse.csr_matrix(A)
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(0, self.N):
            abs = np.absolute(self.sol[i, 0:-1])
            B = np.diag(abs[0:-1]**2 * 0.0j, -1) + np.diag(abs[1:]**2 * 0.0j, 1)
            B[0, -1] = abs[-1]**2
            B[-1, 0] = abs[0]**2
            B *= self.lmbda * self.k * 0.5
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])
            self.sol[i + 1, 0:-1] = linalg.spsolve(As, b)
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    def cn_implicit_builtin_solver(self):
        self.sol[0, :] = self.f1(self.xi)
        A_const = self.tridiag(self.r / 2, 1.0j - self.r, self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A_const[0, -1] = self.r / 2
        A_const[-1, 0] = self.r / 2
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(0, self.N):
            abs_prev = np.absolute(self.sol[i, 0:-1])
            B = 0.5 * self.lmbda * self.k * np.diag(abs_prev**2 * 0.0j, 0)
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])

            def F(U_next):
                abs_next = np.absolute(U_next)
                A = 0.5 * self.lmbda * self.k * np.diag(abs_next ** 2 * 0.0j, 0)
                A += A_const
                a = np.matmul(A, U_next)
                return a - b

            self.sol[i + 1, 0:-1] = optimize.newton_krylov(F, self.sol[i, 0:-1])
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    def cn_liearized_1(self):
        self.sol[0, :] = self.f1(self.xi)
        A_const = self.tridiag(self.r / 2, 1.0j - self.r, self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A_const[0, -1] = self.r / 2
        A_const[-1, 0] = self.r / 2
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(1, self.N):
            abs = np.absolute(self.sol[i, 0:-1])
            A = np.diag(3 * abs**2 * 0.0j - 2 * abs * np.absolute(self.sol[i - 1, 0:-1] * 1.0j), 0)
            B = np.diag(abs**2 * 0.0j, 0)
            A *= 0.5 * self.lmbda * self.k
            B *= 0.5 * self.lmbda * self.k
            A += A_const
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])
            As = sparse.csr_matrix(A)
            self.sol[i + 1, 0:-1] = linalg.spsolve(As, b)
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    def cn_liearized_2(self):
        self.sol[0, :] = self.f1(self.xi)
        A = self.tridiag(self.r / 2, 1.0j - self.r, self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A[0, -1] = self.r / 2
        A[-1, 0] = self.r / 2
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2
        As = sparse.csr_matrix(A)

        for i in range(1, self.N):
            abs = np.absolute(self.sol[i, 0:-1])
            B = np.diag(5 * abs**2 * 0.0j - 2 * abs * np.absolute(self.sol[i - 1, 0:-1]) * 0.0j, 0)
            B *= 0.5 * self.lmbda * self.k
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])
            b -= 0.5 * self.lmbda * self.k * np.multiply(abs**2, self.sol[i - 1, 0:-1])
            self.sol[i + 1, 0:-1] = linalg.spsolve(As, b)
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    def plot(self):
        self.plot_solution(np.real(self.sol), self.xi, self.ti, "Real value of solution")
        self.plot_solution(np.imag(self.sol), self.xi, self.ti, "Imaginary value of solution")
        return 0


if __name__ == '__main__':
    # cn_explicit_average = Nonlin_Scrodinger_Solver([-np.pi, np.pi], 2, 400, 400, 5, 'cn_explicit_average.pkl')
    # cn_explicit_average.cn_explicit_average()
    # cn_explicit_average.plot()

    cn_implicit_builtin_solver = Nonlin_Scrodinger_Solver([-np.pi, np.pi], 1, 200, 200, 5, 'cn_implicit.pkl')
    cn_implicit_builtin_solver.cn_implicit_builtin_solver()
    cn_implicit_builtin_solver.find_deviation_from_analytic()
    print(cn_implicit_builtin_solver.deviation_from_analytic)
    #cn_implicit_builtin_solver.plot()

    # cn_linearized_1 = Nonlin_Scrodinger_Solver([-np.pi, np.pi], 2, 200, 200, 5, 'cn_linearized.pkl')
    # cn_linearized_1.cn_liearized_1()
    # cn_linearized_1.plot()

    # cn_linearized_2 = Nonlin_Scrodinger_Solver([-np.pi, np.pi], 2, 200, 200, 5, 'cn_linearized.pkl')
    # cn_linearized_2.cn_liearized_2()
    # cn_linearized_2.plot()




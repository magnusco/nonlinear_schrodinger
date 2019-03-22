import numpy as np
import pickle
from scipy.sparse import linalg
from scipy import sparse
import matplotlib.pyplot as plt
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
        return np.sin(x)

    def analytic_1(self, x, t):
        X, T = np.meshgrid(x, t)
        a = (np.cos(X) + 1.0j * np.sqrt(2) * np.sinh(T - np.pi)) / (np.sqrt(2) * np.cosh(T - np.pi) - np.cos(X))
        return a * np.exp(1.0j * T)

    def plot_solution(self, sol, xi, ti, title):
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

    def cn_explicit_average(self):
        self.sol[0, :] = self.f1(self.xi)
        A = self.tridiag(self.r / 2, 1.0j - self.r, + self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A[0, -1] = self.r / 2
        A[-1, 0] = self.r / 2
        As = sparse.csr_matrix(A)
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(0, self.N):
            abs = np.absolute(self.sol[i, 0:-1])**2
            B = np.diag(abs[0:-1] * 0.0j, -1) + np.diag(abs[1:] * 0.0j, 1)
            B[0, -1] = abs[-1]
            B[-1, 0] = abs[0]
            B *= self.lmbda * self.k * 0.5
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])
            self.sol[i + 1, 0:-1] = linalg.spsolve(As, b)
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    def cn_implicit(self):
        self.sol[0, :] = self.f1(self.xi)
        A_const = self.tridiag(self.r / 2, 1.0j - self.r, + self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A_const[0, -1] = self.r / 2
        A_const[-1, 0] = self.r / 2
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(0, self.N):
            abs = np.absolute(self.sol[i, 0:-1])**2
            B = np.diag(abs * 0.0j, 0) * self.lmbda * self.k * 0.5
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])
        return 0

    def cn_liearized(self):
        self.sol[0, :] = self.f1(self.xi)
        A = self.tridiag(self.r / 2, 1.0j - self.r, + self.r / 2, self.M)
        B_const = self.tridiag(- self.r / 2, 1.0j + self.r, - self.r / 2, self.M)
        A[0, -1] = self.r / 2
        A[-1, 0] = self.r / 2
        As = sparse.csr_matrix(A)
        B_const[0, -1] = - self.r / 2
        B_const[-1, 0] = - self.r / 2

        for i in range(0, self.N):
            abs = np.absolute(self.sol[i, 0:-1])**2
            B = (3 / 2) * np.diag(abs * 0.0j, 0) - np.diag(np.absolute(self.sol[i, 0:-1] * np.absolute(self.sol[i-1, 0:-1])), 0)
            B *= self.lmbda * self.k
            B += B_const
            b = np.matmul(B, self.sol[i, 0:-1])
            self.sol[i + 1, 0:-1] = linalg.spsolve(As, b)
        self.sol[:, -1] = self.sol[:, 0]
        return 0

    def plot(self):
        self.plot_solution(np.real(self.sol), self.xi, self.ti, "Real value of solution")
        self.plot_solution(np.imag(self.sol), self.xi, self.ti, "Imaginary value of solution")
        return 0


if __name__ == '__main__':
    cn_explicit_average = Nonlin_Scrodinger_Solver([-np.pi, np.pi], 2, 200, 200, 5, 'cn_explicit_average.pkl')
    cn_explicit_average.cn_explicit_average()
    cn_explicit_average.plot()

    # cn_implicit = Nonlin_Scrodinger_Solver([-np.pi, np.pi], 2, 200, 200, 5, 'cn_implicit.pkl')
    # cn_implicit.cn_implicit()
    # cn_implicit.plot()

    cn_linearized = Nonlin_Scrodinger_Solver([-np.pi, np.pi], 2, 200, 200, 5, 'cn_linearized.pkl')
    cn_linearized.cn_liearized()
    cn_linearized.plot()




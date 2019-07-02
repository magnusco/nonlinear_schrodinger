if __name__ == '__main__':


    def exact_1(x, t, lmbda):
        return np.exp(1.0j * (x - t * (lmbda + 1)))

    def exact_2(x, t, lmbda):
        T = 5
        t -= T / 2
        a = 1
        b = 1
        theta = a**2 * b * np.sqrt(2 - b**2) * t
        c = (2 * b**2 * np.cosh(theta) + 2 * 1.0j * b * np.sqrt(2 - b**2) * np.sinh(theta)) / \
            (2 * np.cosh(theta) - np.sqrt(2) * np.sqrt(2 - b**2) * np.cos(a * b * x)) - 1
        return c * a * np.exp(1.0j * a**2 * t)


    # forward_euler = NSE([-np.pi, np.pi], -1, 100, 10000, 5, analytic=exact_2, filename='forward_euler.pkl')
    # forward_euler.forward_euler()
    # forward_euler.calculate_analytic()
    # forward_euler.plot()
    # forward_euler.plot_analytic()

    # central_time = NSE([-np.pi, np.pi], -1, 200, 5000, 5, analytic=exact_2, filename='central_time.pkl')
    # central_time.central_time()
    # central_time.calculate_analytic()
    # central_time.plot()
    # central_time.plot_analytic()

    # hopscotch = NSE([-np.pi, np.pi], -1, 200, 400, 5, analytic=exact_2, filename='hopscotch.pkl')
    # hopscotch.hopscotch()
    # hopscotch.calculate_analytic()
    # hopscotch.plot()
    # hopscotch.plot_analytic()

    cn_implicit_builtin_solver = NSE([-np.pi, np.pi], -1, 300, 600, 5, analytic=exact_2, filename='cn_implicit.pkl')
    cn_implicit_builtin_solver.cn_implicit_builtin_solver()
    cn_implicit_builtin_solver.calculate_analytic()
    cn_implicit_builtin_solver.plot()
    cn_implicit_builtin_solver.plot_analytic()
    cn_implicit_builtin_solver.animate_solution(True)

    # cn_linearized_1 = NSE([-np.pi, np.pi], -1, 200, 400, 5, analytic=exact_2, filename='cn_linearized.pkl')
    # cn_linearized_1.cn_liearized_1()
    # cn_linearized_1.calculate_analytic()
    # cn_linearized_1.plot()
    # cn_linearized_1.plot_analytic()
    # cn_linearized_1.animate_solution(True)

    # cn_linearized_2 = NSE([-np.pi, np.pi], -1, 200, 200, 5, analytic=exact_2, filename='cn_linearized.pkl')
    # cn_linearized_2.cn_liearized_2()
    # cn_linearized_2.calculate_analytic()
    # cn_linearized_2.plot()
    # cn_linearized_2.plot_analytic()
    # cn_linearized_2.animate_solution(True)

    # convergence_order_space(exact_2, -1)
    # convergence_order_time(exact_2, -1)

    # run_time(exact_2, -1)

    conservation_of_mass(exact_1, 2)
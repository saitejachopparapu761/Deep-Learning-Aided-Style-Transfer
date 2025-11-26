import numpy as np
import time

# Enhanced Equilibrium Optimizer (EEO)
def Proposed(Positions, fobj, lb, ub, max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    pop = np.random.rand(N, dim) * (ub - lb) + lb

    alpha = 0.1
    betamin = 0.2
    gamma = 1.0

    best_index = np.zeros((dim, 1))
    best_solution = float('inf')

    convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    while t < max_iter:

        for it in range(max_iter):
            fitness = np.array([fobj(ind) for ind in pop])
            best_index = np.argmin(fitness)
            best_solution = pop[best_index]
            # convergence_curve.append(fitness[best_index])

            for i in range(N):
                temp = pop[i].copy()

                for j in range(dim):
                    r1, r2, r3 = np.min(fitness) / (np.max(fitness) * np.mean(fitness))
                    A = 2 * r1 * alpha - alpha
                    C = 2 * r2 * gamma
                    l = (2 * r3) - 1

                    p = 0.5
                    if r3 < p:
                        p = 1.0

                    if np.abs(A) < 1:
                        D = np.abs(C * best_solution[j] - pop[i, j])
                        temp[j] = best_solution[j] - A * D

                    elif np.abs(A) >= 1:
                        rand_leader_index = np.random.randint(Positions)
                        X_rand = pop[rand_leader_index]
                        D = np.abs(C * X_rand[j] - pop[i, j])
                        temp[j] = X_rand[j] - A * D

                    if temp[j] < lb[i][j]:
                        temp[j] = (lb[i][j] + pop[i, j]) / 2

                    if temp[j] > ub[i][j]:
                        temp[j] = (ub[i][j] + pop[i, j]) / 2

                fit_temp = fobj(temp)
                if fit_temp < fitness[i]:
                    fitness[i] = fit_temp
                    pop[i] = temp

            # Update the global best solution
            if np.min(fitness) < fitness[best_index]:
                best_index = np.argmin(fitness)
                best_solution = pop[best_index]

        convergence_curve[t] = best_index
        t = t + 1

    best_index = convergence_curve[max_iter - 1][0]
    ct = time.time() - ct
    return best_index, convergence_curve, best_solution, ct

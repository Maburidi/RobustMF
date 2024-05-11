
import numpy as np


############################# SOLVER ###############################

def gnmf_objective(X, W, H, L, alpha):
    term1 = 0.5 * np.linalg.norm(X - np.dot(W, H.T), 'fro')**2
    term2 = alpha * np.trace(np.dot(np.dot(H.T, L), H))
    objective_function = term1 + term2
    return objective_function


def gnmf(A,S,num_features, alpha= 0.5, num_iterations=50):
    # Initialize W and H
    D = np.diag(np.sum(A, axis=1))
    L = D - S
    W = np.random.rand(A.shape[0], num_features)
    H = np.random.rand(A.shape[1], num_features)

    for _ in range(num_iterations):
        W *= ((A @ H) / (W @ H.T @ H))
        H *= ((A.T @ W + alpha * S @ H)/( H @ W.T @ W + alpha * D @ H ))
        print( f"Loss ={gnmf_objective(A, W, H, L, alpha)} ..... Iteration:{_}")

    return W, H

def symnmf_newton(A, k, params=None):
    """
    SYMNMF_NEWTON Newton-like algorithm for Symmetric NMF (SymNMF)
    Args:
        A: The symmetric matrix.
        k: The number of columns in the nonnegative matrix H.
        params (optional): Parameters for the algorithm.
    Returns:
        H: The low-rank n*k matrix used to approximate A.
        iter: Number of iterations before termination.
        obj: Objective value f(H) at the final solution H.
    """

    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a symmetric matrix!")

    if params is None:
        H = 2 * np.full((n, k), np.sqrt(np.mean(np.mean(A)) / k)) * np.random.rand(n, k)
        maxiter = 10000
        tol = 1e-4
        sigma = 0.1
        beta = 0.1
        computeobj = True
        debug = 0
    else:
        Hinit = params.get('Hinit', None)
        if Hinit is not None:
            n, kH = Hinit.shape
            if n != A.shape[0]:
                raise ValueError("A and params.Hinit must have same number of rows!")
            if kH != k:
                raise ValueError("params.Hinit must have k columns!")
            H = Hinit
        else:
            H = 2 * np.full((n, k), np.sqrt(np.mean(np.mean(A)) / k)) * np.random.rand(n, k)

        maxiter = params.get('maxiter', 10000)
        tol = params.get('tol', 1e-4)
        sigma = params.get('sigma', 0.1)
        beta = params.get('beta', 0.1)
        computeobj = params.get('computeobj', True)
        debug = params.get('debug', 0)

    projnorm_idx = np.zeros((n, k), dtype=bool)
    R = [None] * k
    p = np.zeros(k)
    left = H.T @ H
    obj = np.linalg.norm(A, 'fro')**2 - 2 * np.trace(H.T @ (A @ H)) + np.trace(left @ left)
    gradH = 4 * (H @ (H.T @ H) - A @ H)
    initgrad = np.linalg.norm(gradH, 'fro')
    if debug:
        print('init grad norm', initgrad)

    for iter in range(1, maxiter+1):
        gradH = 4 * (H @ (H.T @ H) - A @ H)
        projnorm_idx_prev = projnorm_idx.copy()
        projnorm_idx = np.logical_or(gradH <= np.finfo(float).eps, H > np.finfo(float).eps)
        projnorm = np.linalg.norm(gradH[projnorm_idx])
        if projnorm < tol * initgrad:
            if debug:
                print('final grad norm', projnorm)
            break
        else:
            if debug > 1:
                print('iter {}: grad norm {}'.format(iter, projnorm))

        if iter % 100 == 0:
            p = np.ones(k)

        step = np.zeros((n, k))
        hessian = [None] * k
        temp = H @ H.T - A

        for i in range(k):
            if np.any(projnorm_idx_prev[:, i] != projnorm_idx[:, i]):
                hessian[i] = hessian_blkdiag(temp, H, i, projnorm_idx)
                R[i], p[i] = np.linalg.cholesky(hessian[i], lower=True)
            if p[i] > 0:
                step[:, i] = gradH[:, i]
            else:
                step_temp = np.linalg.solve(R[i].T, np.linalg.solve(R[i], gradH[projnorm_idx[:, i], i]))
                step_part = np.zeros(n)
                step_part[projnorm_idx[:, i]] = step_temp
                step_part[np.logical_and(step_part > -np.finfo(float).eps, H[:, i] <= np.finfo(float).eps)] = 0
                if np.sum(gradH[:, i] * step_part) / np.linalg.norm(gradH[:, i]) / np.linalg.norm(step_part) <= np.finfo(float).eps:
                    p[i] = 1
                    step[:, i] = gradH[:, i]
                else:
                    step[:, i] = step_part

        alpha_newton = 1
        Hn = np.maximum(H - alpha_newton * step, 0)
        left = Hn.T @ Hn
        newobj = np.linalg.norm(A, 'fro')**2 - 2 * np.trace(Hn.T @ (A @ Hn)) + np.trace(left @ left)
        if newobj - obj > sigma * np.sum(np.sum(gradH * (Hn-H))):
            while True:
                alpha_newton *= beta
                Hn = np.maximum(H - alpha_newton * step, 0)
                left = Hn.T @ Hn
                newobj = np.linalg.norm(A, 'fro')**2 - 2 * np.trace(Hn.T @ (A @ Hn)) + np.trace(left @ left)
                if newobj - obj <= sigma * np.sum(np.sum(gradH * (Hn-H))):
                    H = Hn
                    obj = newobj
                    break
        else:
            H = Hn
            obj = newobj

    if not computeobj:
        obj = -1

    return H, iter, obj

def hessian_blkdiag(temp, H, i, projnorm_idx):
    """
    Function to compute the block diagonal elements of the Hessian matrix
    """
    return np.diag(np.sum((temp.T @ H[:, [j]])[projnorm_idx[:, j]] ** 2 for j in range(H.shape[1])))










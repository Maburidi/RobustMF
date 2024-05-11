
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










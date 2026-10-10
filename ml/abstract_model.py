from base import *
from abc import ABC, abstractmethod
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import norm


class AbstractLatentFactorModel(ABC):
    def __init__(self, tau, k, alpha, lambda_):
        self._tau = tau
        self._k = k
        self._alpha = alpha
        self._lambda = lambda_
        self._b = None
        self._W_theta = None
        self._training_history = None
        self.mse_test = None
        self.apk_test = None

    def _calculate_neighbor_weight_matrix(self, train_r: csc_matrix) -> np.ndarray:
        """Build W in float32, computing cosine similarities in row blocks."""
        train_r = train_r.astype(np.float32, copy=False)
        l2norm = np.asarray(norm(train_r, ord=2, axis=1), dtype=np.float32).reshape(-1)
        l2norm[l2norm == 0] = 1.0
        U = train_r.multiply((1.0 / l2norm).reshape(-1, 1)).tocsc()

        m = train_r.shape[0]
        W = np.empty((m, m), dtype=np.float32)
        block_rows = 256
        for start in tqdm(range(0, m, block_rows), desc="Building neighbor weights"):
            end = min(start + block_rows, m)
            similarities = U[start:end].dot(U.transpose()).toarray().astype(np.float32, copy=False)
            similarities = np.clip(similarities, -1.0, 1.0)
            weights = np.exp(self._tau * np.power(1.0 - similarities, self._k)).astype(np.float32, copy=False)
            W[start:end] = weights
            del similarities, weights

        np.fill_diagonal(W, 0.0)
        return W

    @abstractmethod
    def train(self, *args, **kwargs):
        raise Exception("Not implemented yet.")

    @abstractmethod
    def predict(self, arg):
        raise Exception("Not implemented yet.")

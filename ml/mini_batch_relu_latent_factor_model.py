from base import *
from scipy.sparse import csc_matrix
from ml.abstract_model import AbstractLatentFactorModel
import math


class MiniBatchReLuLatentFactorModel(AbstractLatentFactorModel):
    def __init__(self, tau=-2, k=2, alpha=0.01, lambda_=0.0001,
                 batch_size=128, max_epoch=1000):
        super(MiniBatchReLuLatentFactorModel, self).__init__(tau, k, alpha, lambda_)
        self._batch_size = int(batch_size)
        self._max_epoch = int(max_epoch)

    def train(self, train_r: csc_matrix):
        # Never densify the entire train_r matrix.
        train_r = train_r.astype(np.float32, copy=False)
        m, n = train_r.shape
        theta = np.zeros((m, m), dtype=np.float32)
        b = np.zeros((m, 1), dtype=np.float32)
        W = self._calculate_neighbor_weight_matrix(train_r)
        W_theta = np.multiply(W, theta, dtype=np.float32)

        def cal_loss():
            total_loss = 0.0
            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                B = train_r[:, start:end].toarray().astype(np.float32, copy=False)
                rhat = W_theta @ B
                rhat += b
                np.maximum(rhat, 0, out=rhat)
                diff = rhat - B
                total_loss += float(np.sum(diff * diff, dtype=np.float64))
                del B, rhat, diff
            return total_loss / (2.0 * n)

        self._training_history = [cal_loss()]
        print(f"Loss before training: {self._training_history[-1]:0.8E}")

        for epoch in range(self._max_epoch):
            pbar = tqdm(range(math.ceil(n / self._batch_size)))
            for batch_index in pbar:
                start = batch_index * self._batch_size
                end = min(start + self._batch_size, n)
                B = train_r[:, start:end].toarray().astype(np.float32, copy=False)
                bsize = B.shape[1]

                B_hat = W_theta @ B
                B_hat += b
                B_hat_B = B_hat - B
                B_hat_B[B_hat < 0] = 0

                grad_theta = B_hat_B @ B.T
                grad_theta *= (self._alpha / bsize)
                grad_theta += self._lambda * theta
                grad_b = np.sum(B_hat_B, axis=1, keepdims=True, dtype=np.float32)
                grad_b *= (self._alpha / bsize)

                theta -= grad_theta
                b -= grad_b
                np.multiply(W, theta, out=W_theta)

                delta_eps = float(
                    np.sum(grad_theta * grad_theta, dtype=np.float64)
                    + np.sum(grad_b * grad_b, dtype=np.float64)
                )
                pbar.set_description(
                    f"Epoch {epoch + 1} batch {batch_index + 1} delta={delta_eps:.8E}"
                )
                del B, B_hat, B_hat_B, grad_theta, grad_b

            pbar.close()
            loss = cal_loss()
            self._training_history.append(loss)
            print(f"Final Loss of epoch {epoch + 1}: {loss:0.8E}")

        self._W_theta = W_theta
        self._b = b

    def predict(self, r: np.ndarray):
        r = np.asarray(r, dtype=np.float32)
        res = self._W_theta @ r
        res += self._b
        np.maximum(res, 0, out=res)
        return res

from base import *
from scipy.sparse import csc_matrix
from ml.abstract_model import AbstractLatentFactorModel
import math
import os
import pickle
import time


class MiniBatchReLuLatentFactorModel(AbstractLatentFactorModel):
    """
    Same model and same gradient formula as the original implementation:

        r_hat = ReLU((W * theta) r + b)
        theta <- theta - ( alpha * W * (B_hat_B B^T) / bsize + lambda * theta )   (W elementwise)
        b     <- b     - alpha / bsize * sum(B_hat_B, axis=1)

    What changed is only HOW the products are computed: the document matrix B is sparse
    (a few MeSH terms per document), so B^T-times-dense and B-times-dense products cost
    nnz(B) * m instead of m * m * batch_size. All big matrices are stored TRANSPOSED
    (C-contiguous) so that scipy's sparse @ dense kernels never need a copy.
    Arrays are float32.
    """

    def __init__(self, tau=-2, k=2, alpha=0.01, lambda_=0.0001,
                 batch_size=10000, max_epoch=1000, loss_every=1, loss_batch_size=2048,
                 checkpoint_path=None, checkpoint_every=5):
        super(MiniBatchReLuLatentFactorModel, self).__init__(tau, k, alpha, lambda_)
        self._batch_size = int(batch_size)
        self._max_epoch = int(max_epoch)
        self._loss_every = int(loss_every)          # compute full loss every N epochs (last epoch always)
        self._loss_batch_size = int(loss_batch_size)
        self._checkpoint_path = checkpoint_path      # if set, save a usable model every N epochs
        self._checkpoint_every = int(checkpoint_every)

    def train(self, train_r: csc_matrix):
        R = train_r.tocsc().astype(np.float32)
        m, n = R.shape
        print(f"Training matrix {m} x {n}, nnz={R.nnz} ({R.nnz / n:.1f} per doc)", flush=True)

        # ---- neighbor weights (stored transposed, contiguous) ----
        t0 = time.time()
        W = self._calculate_neighbor_weight_matrix(R)
        print(f"W: min={W[W > 0].min():.6f} max={W.max():.6f} ({time.time() - t0:.0f}s)", flush=True)
        WT = np.ascontiguousarray(W.T)
        del W
        thetaT = np.zeros((m, m), dtype=np.float32)
        QT = np.zeros((m, m), dtype=np.float32)      # (W * theta)^T
        gradT = np.empty((m, m), dtype=np.float32)   # reusable buffer
        b = np.zeros((m, 1), dtype=np.float32)

        def forward_T(B_csc):
            """(ReLU-free) r_hat^T for a sparse batch B (m x bs): returns bs x m dense."""
            out = B_csc.T.dot(QT)          # csr (bs x m) @ dense (m x m) -> bs x m
            out += b.T
            return out

        def cal_loss():
            total = 0.0
            for s in range(0, n, self._loss_batch_size):
                B = R[:, s:s + self._loss_batch_size]
                rhat = forward_T(B)
                np.maximum(rhat, 0, out=rhat)
                rhat -= B.T.toarray()
                total += float(np.sum(rhat * rhat, dtype=np.float64))
            return total / (2.0 * n)

        # theta = 0, b = 0  =>  r_hat = 0, so the initial loss is exactly sum(R^2) / 2n
        self._training_history = [float(np.sum(R.data.astype(np.float64) ** 2)) / (2.0 * n)]
        print(f"Loss before training: {self._training_history[-1]:0.8E}", flush=True)

        num_batches = math.ceil(n / self._batch_size)
        train_start = time.time()
        for epoch in range(self._max_epoch):
            ep_start = time.time()
            pbar = tqdm(range(num_batches))
            for batch_index in pbar:
                s = batch_index * self._batch_size
                B = R[:, s:s + self._batch_size]       # sparse m x bs
                bsize = B.shape[1]

                B_hat_T = forward_T(B)                 # bs x m
                B_hat_B_T = B_hat_T - B.T.toarray()    # bs x m  (dense, bs*m floats)
                B_hat_B_T[B_hat_T < 0] = 0             # ReLU'(r_hat)

                # gradT = (B_hat_B @ B^T)^T = B @ B_hat_B^T
                np.copyto(gradT, B.dot(B_hat_B_T))
                gradT *= WT                            # elementwise W (alpha_W in the original)
                gradT *= (self._alpha / bsize)
                gradT += self._lambda * thetaT
                grad_b = np.sum(B_hat_B_T, axis=0).reshape(-1, 1)
                grad_b *= (self._alpha / bsize)

                thetaT -= gradT
                b -= grad_b
                np.multiply(WT, thetaT, out=QT)

                delta_eps = float(np.vdot(gradT.ravel(), gradT.ravel()) + np.vdot(grad_b, grad_b))
                pbar.set_description(f"Epoch {epoch + 1} batch {batch_index + 1} delta={delta_eps:.8E}")
                del B_hat_T, B_hat_B_T, grad_b
            pbar.close()

            msg = f"Epoch {epoch + 1}/{self._max_epoch} done in {time.time() - ep_start:.0f}s"
            if (epoch + 1) % self._loss_every == 0 or epoch + 1 == self._max_epoch:
                loss = cal_loss()
                self._training_history.append(loss)
                msg += f" | loss {loss:0.8E}"
            elapsed = time.time() - train_start
            msg += f" | ETA {elapsed / (epoch + 1) * (self._max_epoch - epoch - 1) / 60:.0f} min"
            print(msg, flush=True)

            if self._checkpoint_path and ((epoch + 1) % self._checkpoint_every == 0 or epoch + 1 == self._max_epoch):
                self._W_theta, self._b = QT.T, b
                tmp = self._checkpoint_path + ".tmp"
                with open(tmp, 'wb') as f:
                    pickle.dump(self, f, protocol=4)
                os.replace(tmp, self._checkpoint_path)
                print(f"Checkpoint saved after epoch {epoch + 1}: {self._checkpoint_path}", flush=True)

        # keep the original attribute meaning: W_theta (m x m) and b (m x 1); W_theta is a view of QT
        self._W_theta = QT.T
        self._b = b

    def predict(self, r: np.ndarray):
        r = np.asarray(r, dtype=np.float32)
        if r.ndim == 2 and r.shape[1] == 1:
            idx = np.flatnonzero(r[:, 0])
            if len(idx) < r.shape[0] // 8:          # sparse input: only touch the needed columns
                res = self._W_theta[:, idx] @ r[idx]
                res += self._b
                np.maximum(res, 0, out=res)
                return res
        res = self._W_theta @ r
        res += self._b
        np.maximum(res, 0, out=res)
        return res
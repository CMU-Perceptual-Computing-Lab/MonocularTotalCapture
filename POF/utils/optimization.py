import numpy as np


def transReProjectionLoss(t, X0, K, uv):
    assert t.shape == (3,)
    assert len(X0.shape) == 2 and X0.shape[1] == 3
    assert K.shape == (3, 3)
    assert len(uv.shape) == 2 and uv.shape[1] == 2

    X = X0 + t[np.newaxis, :]
    x = X.dot(K.T)
    x /= x[:, 2][:, np.newaxis]

    return np.sum(np.square(x[:, :2] - uv))

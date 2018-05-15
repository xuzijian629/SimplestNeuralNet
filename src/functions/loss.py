def mse(ys, ts):
    return np.sum((ys - ts) ** 2) / len(ts) / 2

def msedif(ys, ts):
    return (ys - ts) / len(ts)

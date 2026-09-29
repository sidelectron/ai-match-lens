import numpy as np


def homography(image, pitch):
    """Fit a planar mapping using normalized DLT; reject degenerate landmarks."""
    image, pitch = np.asarray(image, float), np.asarray(pitch, float)
    if image.shape != pitch.shape or image.ndim != 2 or image.shape[1] != 2 or len(image) < 4:
        raise ValueError('Supply at least four corresponding image/pitch points.')
    if not np.isfinite(image).all() or not np.isfinite(pitch).all():
        raise ValueError('Landmarks must be finite.')
    def normalize(p):
        center = p.mean(0)
        distance = np.linalg.norm(p - center, axis=1).mean()
        if distance < 1e-8:
            raise ValueError('Landmarks are degenerate.')
        s = np.sqrt(2) / distance
        t = np.array([[s, 0, -s*center[0]], [0, s, -s*center[1]], [0, 0, 1]])
        return project(t, p), t
    a, ta = normalize(image)
    b, tb = normalize(pitch)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                     [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
    matrix = np.asarray(rows)
    if np.linalg.matrix_rank(matrix) < 8:
        raise ValueError('Landmarks must span the pitch plane, not a single line.')
    _, _, vt = np.linalg.svd(matrix)
    h = np.linalg.inv(tb) @ vt[-1].reshape(3, 3) @ ta
    return h / np.linalg.norm(h)


def project(h, points):
    p = np.asarray(points, float).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(h).T
    out = np.full((len(p), 2), np.nan)
    valid = np.abs(q[:, 2]) > 1e-10
    out[valid] = q[valid, :2] / q[valid, 2:3]
    return out

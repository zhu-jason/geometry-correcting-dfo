import numpy as np


class InterpolationSet:
    """
    A thin container for an interpolation set:
    - `points`: array of points, shape (m, d)
    - `values`: array of function values, shape (m,)

    Provides small utilities for adding/removing points, shifting, and
    selecting the furthest point from a reference.
    """

    def __init__(self, points, values, p):
        self.points = np.asarray(points)
        m = self.points.shape[0]
        self.p = p
        if values is None:
            self.values = np.full(m, np.nan)
        else:
            self.values = np.asarray(values)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return (self.points[idx].copy(), self.values[idx].copy())

    def copy(self):
        return InterpolationSet(self.points.copy(), self.values.copy())

    def add_point(self, point, value=np.nan):
        point = np.asarray(point)
        point = point.reshape(1, -1)
        if self.points.size == 0:
            self.points = point
            self.values = np.array([value])
        else:
            self.points = np.vstack([self.points, point])
            self.values = np.append(self.values, value)

    def delete_point(self, index):
        self.points = np.delete(self.points, index, axis=0)
        self.values = np.delete(self.values, index)

    # RETURNS ZERO IF SET IS EMPTY
    def get_furthest(self, s=None):
        if self.points.size == 0:
            return None, None
        if s is None:
            s = np.zeros(self.points.shape[1])
        s = np.asarray(s)
        diffs = np.linalg.norm(self.points - s, axis=1)
        idx = int(np.argmax(diffs))
        return idx, self.points[idx]

    def append_origin(self, value=np.nan):
        self.add_point(np.zeros(self.p), value=value)

    def shift(self, s):
        """Shift all points by subtracting s (broadcast over rows)."""
        if self.points.size != 0:
            self.points = self.points - np.asarray(s)



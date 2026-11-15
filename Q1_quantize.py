"""
CS 351 - Assignment 03, Part B Q1
Starter skeleton: Colour quantization with K-Means and DBSCAN.

Complete the TODO sections according to the assignment instructions.
Only NumPy and Matplotlib should be used for the algorithms.
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_image(path):
    """Load an image as normalized RGB."""
    img = plt.imread(path)
    if img.dtype == np.uint8:
        img = img / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    return img[:, :, :3].astype(float)


def sq_dists(X, C):
    """Calculate pairwise squared distances."""
    d2 = (X**2).sum(1)[:, None] + (C**2).sum(1)[None, :] - 2 * X @ C.T
    return np.maximum(d2, 0)


def downsample(img, max_side=80):
    """Downsample an image."""
    h, w = img.shape[:2]
    rows = np.linspace(0, h - 1, min(max_side, h)).astype(int)
    cols = np.linspace(0, w - 1, min(max_side, w)).astype(int)
    return img[rows][:, cols]


def image_features(img, lam=None):
    """Extract RGB and optional spatial features."""
    # TODO
    raise NotImplementedError


def compression_ratio(k, n_pixels):
    """Calculate compression ratio."""
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# K-Means
# ----------------------------------------------------------------------

class KMeans:
    def __init__(self, k, init="random", max_iter=50, tol=0.001,
                 seed=0, snapshot_iters=()):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.snapshot_iters = set(snapshot_iters)

    def _init_random(self, X):
        # TODO
        raise NotImplementedError

    def _init_kmeanspp(self, X):
        # TODO
        raise NotImplementedError

    def fit(self, X):
        # TODO
        raise NotImplementedError

    def predict(self, X):
        # TODO
        raise NotImplementedError

    def quantize(self, img):
        # TODO
        raise NotImplementedError


# ----------------------------------------------------------------------
# DBSCAN
# ----------------------------------------------------------------------

CORE, BORDER, NOISE = 0, 1, 2

MIN_PTS = 5#first i tried with 3, then gradually increased to 5

#based on my plots i initially set them to {1:9,2:4,3:6,4:3,5:6} but then i adjusted them manually by rechecking how they were showing in dbscan_results.png
#these are on the 0-255 colour scale, the images here are 0-1 so they get divided by 255 when used
EPS_FOR = {"sample1": 5, "sample2": 5, "sample3": 6, "sample4": 3, "sample5": 6}


class DBSCAN:
    def __init__(self, eps, min_pts, chunk=1024):
        self.eps = eps
        self.min_pts = min_pts
        self.chunk = chunk

    def _neighbours(self, X):
        nbrs = []
        for i in range(0, len(X), self.chunk):
            d2 = sq_dists(X[i:i + self.chunk], X)
            for row in d2 <= self.eps**2 + 1e-9:
                nbrs.append(np.flatnonzero(row))
        return nbrs

    def fit(self, X):
        n = len(X)
        nbrs = self._neighbours(X)
        is_core = np.array([len(a) >= self.min_pts for a in nbrs])
        labels = np.full(n, -1)
        c = 0
        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue
            labels[i] = c
            stack = [i]
            while stack:
                p = stack.pop()
                if not is_core[p]:
                    continue
                for q in nbrs[p]:
                    if labels[q] == -1:
                        labels[q] = c
                        stack.append(q)
            c += 1
        #core, border or noise
        types = np.where(is_core, CORE, np.where(labels != -1, BORDER, NOISE))
        self.labels_ = labels
        self.types_ = types
        return self


def k_distance(X, k, chunk=1024):
    """Calculate sorted k-nearest-neighbour distances."""
    out = []
    for i in range(0, len(X), chunk):
        d2 = sq_dists(X[i:i + chunk], X)
        out.append(np.partition(d2, k - 1, axis=1)[:, k - 1])
    return np.sort(np.sqrt(np.concatenate(out)))


def elbow_index(y):
    """Find the elbow point."""
    #the point that sits farthest below the straight line joining the two ends
    x = np.arange(len(y)) / (len(y) - 1)
    yn = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    return int(np.argmax(x - yn))


def choose_eps(X, min_pts, ax=None, title=""):
    kd = k_distance(X, min_pts)
    eps = kd[elbow_index(kd)]
    if ax is not None:
        ax.plot(kd)
        ax.axhline(eps, color="gray", ls=":")
        ax.set_title(title)
        ax.set_xlabel("points sorted")
        ax.set_ylabel(f"distance to {min_pts}th neighbour")
    return eps


def dbscan_image(img_small, labels):
    """Generate the clustered image."""
    h, w = img_small.shape[:2]
    flat = img_small.reshape(-1, 3)
    out = np.zeros((h * w, 3))
    for c in range(labels.max() + 1):
        out[labels == c] = flat[labels == c].mean(0)
    #noise stays black
    return out.reshape(h, w, 3)


# ----------------------------------------------------------------------
# Toy data
# ----------------------------------------------------------------------

def make_rings(n=800, seed=0):
    """Generate concentric rings."""
    # TODO
    raise NotImplementedError


def make_blobs_outliers(n=600, seed=0):
    """Generate Gaussian blobs with outliers."""
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

def save(fig, out, name):
    """Save a figure."""
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)
    plt.close(fig)


def part_b_init(img, out, k=8, runs=5):
    # TODO
    raise NotImplementedError


def part_c_convergence(img, out, name, k=8):
    # TODO
    raise NotImplementedError


def part_d_choose_k(img, out, name, ks=(2, 4, 8, 16, 32)):
    # TODO
    raise NotImplementedError


def part_e_dbscan(img, out, name, min_pts=6):
    small = downsample(img)
    X = small.reshape(-1, 3)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    elbow_eps = choose_eps(X, min_pts, ax[1], f"{name} k-distance")
    eps = EPS_FOR.get(os.path.splitext(name)[0], elbow_eps * 255) / 255
    ax[1].axhline(eps, color="r", ls="--")

    db = DBSCAN(eps, min_pts).fit(X)
    labels = db.labels_
    print(name, "clusters", labels.max() + 1, "noise %", round((labels == -1).mean() * 100, 1))

    ax[0].imshow(small)
    ax[0].set_title(f"{name} (downsampled)")
    ax[2].imshow(dbscan_image(small, labels))
    ax[2].set_title(f"DBSCAN eps={eps * 255:.1f}, {labels.max() + 1} clusters")
    ax[0].axis("off")
    ax[2].axis("off")
    save(fig, out, f"{name}_dbscan.png")
    return db


def part_f_shapes(out):
    # TODO
    raise NotImplementedError


def part_f_images(img, out, name, k, lam=0.5):
    # TODO
    raise NotImplementedError


def part_f_sensitivity(img, name, k, eps, min_pts=6):
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images")
    parser.add_argument("--out", default="out/q1")
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
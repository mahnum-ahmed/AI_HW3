import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

MIN_PTS = 5#first i tried with 3, then gradually increased to 5

def downsample(img, size=80):
    h, w = img.shape[:2]
    rows = np.linspace(0, h - 1, min(size, h)).astype(int)
    cols = np.linspace(0, w - 1, min(size, w)).astype(int)
    return img[rows][:, cols]

def chunk_d2(X, start, chunk):
    part = X[start:start + chunk]
    d2 = (part**2).sum(1)[:, None] + (X**2).sum(1)[None, :] - 2 * part @ X.T
    return np.maximum(d2, 0)

def k_distance(X, k, chunk=500):
    out = []
    for i in range(0, len(X), chunk):
        d2 = chunk_d2(X, i, chunk)
        out.append(np.partition(d2, k - 1, axis=1)[:, k - 1])
    return np.sort(np.sqrt(np.concatenate(out)))

def dbscan(X, eps, min_pts, chunk=500):
    n = len(X)
    nbrs = []
    for i in range(0, n, chunk):
        d2 = chunk_d2(X, i, chunk)
        for row in d2 <= eps**2 + 1e-9:
            nbrs.append(np.flatnonzero(row))
    is_core = np.array([len(a) >= min_pts for a in nbrs])
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
    #2 core, 1 border, 0 noise
    types = np.where(is_core, 2, np.where(labels != -1, 1, 0))
    return labels, types

def segmented(small, labels):
    h, w = small.shape[:2]
    flat = small.reshape(-1, 3)
    out = np.zeros((h * w, 3))
    for c in range(labels.max() + 1):
        out[labels == c] = flat[labels == c].mean(0)
    return out.reshape(h, w, 3).astype(np.uint8)

#based on my plots i initially set them to {1:9,2:4,3:6,4:3,5:6} but then i adjusted them manually by rechecking how they were showing in dbscan_results.png
eps_for = {1: 5, 2: 5, 3: 6, 4: 3, 5: 6}

fig1, ax1 = plt.subplots(1, 5, figsize=(20, 3.5))
fig2, ax2 = plt.subplots(2, 5, figsize=(20, 7))

for i in range(1, 6):
    img = np.array(Image.open(f'images/sample{i}.jpg').convert('RGB'), float)
    small = downsample(img)
    X = small.reshape(-1, 3)

    kd = k_distance(X, MIN_PTS)
    ax1[i - 1].plot(kd)
    ax1[i - 1].axhline(eps_for[i], color='r', ls='--')
    ax1[i - 1].set_title(f'sample{i}')

    labels, types = dbscan(X, eps_for[i], MIN_PTS)
    print(i, 'clusters', labels.max() + 1, 'noise %', round((labels == -1).mean() * 100, 1))
    ax2[0, i - 1].imshow(small.astype(np.uint8))
    ax2[1, i - 1].imshow(segmented(small, labels))
    ax2[0, i - 1].axis('off')
    ax2[1, i - 1].axis('off')

fig1.savefig('kdist.png')
fig2.savefig('dbscan_results.png')
plt.show()
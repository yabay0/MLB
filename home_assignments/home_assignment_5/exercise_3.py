import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# ----------------------------------------------------------------------
# Question 3.3
# ----------------------------------------------------------------------
def compute_circle(points, signs):
    pts = np.asarray(points)
    signs = np.array(signs)
    pos = pts[signs == 1]
    n_pos = len(pos)

    if n_pos == 0:
        return np.array([1.5, 1.5]), 0.2
    if n_pos == 1:
        return pos[0], 0.3
    if n_pos == 2:
        mid = pos.mean(axis=0)
        if tuple(signs) == (-1, 1, 1):
            mid += np.array([0.2, 0.2])
        r = np.linalg.norm(pos[0] - pos[1]) / 2 + 0.15
        return mid, r
    if n_pos == 3:
        A, B, C = pos
        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        Ux = ((A**2).sum() * (B[1] - C[1]) + (B**2).sum() * (C[1] - A[1]) + (C**2).sum() * (A[1] - B[1])) / D
        Uy = ((A**2).sum() * (C[0] - B[0]) + (B**2).sum() * (A[0] - C[0]) + (C**2).sum() * (B[0] - A[0])) / D
        center = np.array([Ux, Uy])
        r = np.linalg.norm(center - A) + 0.15
        return center, r

    return None, None

points = np.array([[0, 0], [1, 0], [0, 1]])
labels = ['P0', 'P1', 'P2']
labelings = list(product([-1, 1], repeat=len(points)))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, labeling in zip(axes.flatten(), labelings):
    ax.set(xlim=(-1, 2), ylim=(-1, 2), aspect='equal')
    ax.set_title(f'Labels: {labeling}')

    # Plot points and labels
    for (x, y), lab, sign in zip(points, labels, labeling):
        color = 'blue' if sign == 1 else 'red'
        ax.plot(x, y, 'o', color=color)
        ax.text(x + 0.05, y + 0.05, lab, fontsize=12)

    center, radius = compute_circle(points, labeling)
    if center is not None:
        circle = plt.Circle(center, radius, fill=False, color='black')
        ax.add_patch(circle)

plt.tight_layout()
plt.savefig("exercise_3_3.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# Question 3.4
# ----------------------------------------------------------------------
def plot_circle(ax, center, radius, color='black', **kwargs):
    circle = plt.Circle(center, radius, fill=False, color=color, **kwargs)
    ax.add_patch(circle)

def compute_circles(points, labeling):
    points = np.asarray(points)
    signs = np.array(labeling)
    pos = points[signs == 1]
    neg = points[signs == -1]
    circles = []

    if len(pos) == 0:
        circles.append({'center': np.array([2.5, 2.5]), 'radius': 0.4, 'color': 'black'})
        return circles
    if len(neg) == 0:
        circles.append({'center': np.array([2.5, 2.5]), 'radius': 0.4, 'color': 'red'})
        return circles

    if len(pos) == 3 and len(neg) == 1:
        for p in neg:
            circles.append({'center': p, 'radius': 0.3, 'color': 'red'})
        return circles

    special_triplet = {
        (1, -1, -1, 1),
        (1, -1, 1, -1),
        (-1, -1, 1, 1)
    }
    if tuple(labeling) in special_triplet:
        circle_points = neg
        circle_color = 'red'
    else:
        if len(pos) <= 3:
            circle_points = pos
            circle_color = 'black'
        elif len(neg) <= 3:
            circle_points = neg
            circle_color = 'red'
        else:
            return circles

    n = len(circle_points)
    if n == 1:
        circles.append({'center': circle_points[0], 'radius': 0.3, 'color': circle_color})
    elif n == 2:
        center = circle_points.mean(axis=0)
        radius = np.linalg.norm(circle_points[0] - circle_points[1]) / 2 + 0.15
        circles.append({'center': center, 'radius': radius, 'color': circle_color})
    elif n == 3:
        A, B, C = circle_points
        D = 2 * (A[0]*(B[1] - C[1]) + B[0]*(C[1] - A[1]) + C[0]*(A[1] - B[1]))
        if D == 0:
            return circles
        Ux = ((A**2).sum() * (B[1] - C[1]) + (B**2).sum() * (C[1] - A[1]) + (C**2).sum() * (A[1] - B[1])) / D
        Uy = ((A**2).sum() * (C[0] - B[0]) + (B**2).sum() * (A[0] - C[0]) + (C**2).sum() * (B[0] - A[0])) / D
        center = np.array([Ux, Uy])
        radius = np.linalg.norm(center - A) + 0.15
        circles.append({'center': center, 'radius': radius, 'color': circle_color})

    return circles

points = np.array([[0, 0], [0, 1], [-1, 2], [1, 2]])
labels = ['P0', 'P1', 'P2', 'P3']
labelings = list(product([-1, 1], repeat=len(points)))
custom_order = [0, 1, 2, 4, 8, 5, 6, 12, 9, 10, 11, 13, 14, 3, 7, 15]
ordered_labelings = [labelings[i] for i in custom_order]

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

for ax, labeling in zip(axes, ordered_labelings):
    ax.set(xlim=(-2, 3), ylim=(-1, 3), aspect='equal')
    ax.set_title(f'Labels: {labeling}')

    for (x, y), lab, sign in zip(points, labels, labeling):
        color = 'blue' if sign == 1 else 'red'
        ax.plot(x, y, 'o', color=color)
        ax.text(x + 0.05, y + 0.05, lab, fontsize=12)

    for circ in compute_circles(points, labeling):
        plot_circle(ax, circ['center'], circ['radius'], color=circ['color'], linestyle='-')

fig.subplots_adjust(hspace=-0.3)
plt.savefig("exercise_3_4.png", dpi=300, bbox_inches="tight")
plt.show()
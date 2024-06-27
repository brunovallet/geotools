import math, random
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import copy

make_figures = True # True or False: most of the time is spent making figures

# optimizer parameters
min_dist_improve = 1  # minimum distance improvement to continue optimization
max_iter = 30  # max number of iterations, for security in case optimization goes wrong
damping = 1 # more damping makes convergence slower but more stable

# Circle generator parameters
d_var = 1000  # variance of the distance from a circle center to (0,0)
n_circle = 20  # number of circles
r_av = 300  # average circle radius
r_var = 200  # variance on circle radius


def det(v1: np.array, v2: np.array):
    return np.linalg.det([v1, v2])


class Bbox:
    def __init__(self, p1: np.array = np.zeros(2), p2: np.array = np.zeros(2)):
        self.min = np.min([p1, p2], axis=0)
        self.max = np.max([p1, p2], axis=0)

    def add(self, bbx):
        self.min = np.min([self.min, bbx.min], axis=0)
        self.max = np.max([self.max, bbx.max], axis=0)


class Circle:
    def __init__(self, center=np.zeros(2), r=0):
        self.center = center
        self.r = r

    def bbox(self):
        diag = self.r * np.ones(2)
        return Bbox(self.center - diag, self.center + diag)

    def inside(self, p: np.array):
        return np.linalg.norm(self.center - p) < self.r

    # intersection between this circle (self) and [ab]
    # if there are 2, return any, if there are none return None
    def intersection(self, a: np.array, b: np.array):
        ab = b - a
        ac = self.center - a
        nab = np.linalg.norm(ab)
        nac = np.linalg.norm(ac)
        if nab == 0.:
            # a = b
            if nac == self.r:
                return a
            return None
        u = ab / nab  # unit
        x = np.dot(ac, u)
        y = abs(det(ac, u))
        if y > self.r:
            return None  # circle does not intersect (ab)
        delta = math.sqrt(self.r * self.r - y * y)
        for t in [x - delta, x + delta]:  # two roots for two intersections with (ab)
            if 0 <= t <= nab:
                return a + t * u
        return None

    def __str__(self):
        return "Circle(%s, %s)" % (self.center, self.r)


# length of a path stored in a 2D array (dim 0: (x,y) coords, dim 1: list of points)
def length(path: np.array):
    vects = path[:, 1:] - path[:, :-1]
    return np.sum(np.linalg.norm(vects, axis=0))


if __name__ == "__main__":
    # sample random circles
    circles = [Circle(np.random.normal(0, d_var, 2), max(1, np.random.normal(r_av, r_var))) for i in range(n_circle)]

    # get circles bbox for display and initialize path to the one linking the circle centers
    bbx = circles[0].bbox()
    paths = np.zeros((2, n_circle, max_iter))
    lengths = np.zeros(max_iter)
    for i in range(n_circle):
        circle = circles[i]
        if make_figures:
            bbx.add(circle.bbox())
        paths[:, i, 0] = circle.center
    lengths[0] = length(paths[:, :, 0])
    print('length of path connecting circle centers: %s' % lengths[0])

    # optimization iterations
    for j in range(max_iter - 1):
        if make_figures:
            fig, ax = plt.subplots()  # just for display
            # adjust range to bbox of the circles
            ax.set_xlim((bbx.min[0], bbx.max[0]))
            ax.set_ylim((bbx.min[1], bbx.max[1]))
            ax.set_aspect('equal', adjustable='box')
        for i in range(n_circle):
            circle = circles[i]
            I = None
            vi = np.zeros((2)) # direction from circle center to contact point
            if 0 < i < n_circle - 1:  # not endpoint, look if circle is already intersected
                I = circle.intersection(paths[:, i - 1, j], paths[:, i + 1, j])
            if I is not None:
                vi = I-circle.center
            else:
                if i > 0:
                    seg_i = paths[:, i - 1, j] - paths[:, i, j]
                    vi += seg_i / np.linalg.norm(seg_i)
                if i < n_circle - 1:
                    seg_i = paths[:, i + 1, j] - paths[:, i, j]
                    vi += seg_i / np.linalg.norm(seg_i)
                # average between previous ray and actual for damping
            ray = (circle.r / np.linalg.norm(vi)) * vi + damping*(paths[:, i, j] - circle.center)
            paths[:, i, j + 1] = circle.center + (circle.r / np.linalg.norm(ray)) * ray
            if make_figures:
                ax.add_patch(plt.Circle((circle.center[0], circle.center[1]), circle.r, color='blue', fill=False))
                plt.plot([circle.center[0], paths[0, i, j]], [circle.center[1], paths[1, i, j]], color='green')
        lengths[j+1] = length(paths[:, :, j + 1])
        print('iter %d, length: %s' % (j, lengths[j+1]))
        if make_figures:
            plt.plot(paths[0, :, j], paths[1, :, j], color='red')
            fig.savefig('min_path%02d.png' % j)
            plt.close()
        if abs(lengths[j]-lengths[j+1]) < min_dist_improve: # stop criterion
            break
    print("Optimal path:")
    for i in range(n_circle):
        print(paths[:, i, j+1])

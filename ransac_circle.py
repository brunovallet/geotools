# detection de thermiques par RANSAC à partir de traces GPS
# l'idée est que les paras forment des cercles approchés dans les thermiques
#
# import math, random
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import copy

# paramètres de RANSAC
max_dist = 5  # distance min pour être considéré inlier
max_angle = 0.3 # angle max pour être considéré inlier
n_min = 5  # nb min de points pour détecter un cercle
n_iter = 50000  # nombre de tirages pour chaque RANSAC

# paramètres du générateur de paras
d_max = 50 # distance max d'un para/thermique au point (0,0)
n_thrm = 3 # nombre de thermiques
n_para = 9  # nombre de paras par thermique
n_outliers = 12 # nombre de paras pas dans un thermique
pos_noise = 2 # bruit sur la position d'un para
dir_noise = 0.1 # bruit sur la direction d'un para
r_av_thrm = 20 # rayon moyen d'un thermique
r_var_thrm = 3 # variance sur le rayon d'un thermique


class Para:
    def __init__(self, pos=np.zeros(2), dir=0):
        self.pos = pos
        self.dir = dir

    def __str__(self):
        return "Pr(%s, %s°)" % (self.pos, 180 * self.dir / math.pi)


class Thermique:
    def __init__(self, pos=np.zeros(2), r=0):
        self.pos = pos
        self.r = r
        self.inliers = []
        self.outliers = []

    def __str__(self):
        return "Thrm(%s, %s)" % (self.pos, self.r)


def det(v1: np.array, v2: np.array):
    return np.linalg.det([v1, v2])


def unit(dir):
    return np.array([math.sin(dir), math.cos(dir)])


def perp(dir):
    return np.array([-math.cos(dir), math.sin(dir)])


def intersect(para1: Para, para2: Para):
    ray1 = perp(para1.dir)
    ray2 = perp(para2.dir)
    v12 = para2.pos - para1.pos
    d12 = det(ray1, ray2)
    if d12 == 0:
        # print("Droites //, pas d'intersection")
        return None
    return para1.pos + (det(v12, ray2) / d12) * ray1


# Detection de cercles par RANSAC avec nombre fixé d'itérations n_iter, seuil de distance d'inlier et seuil angulaire (en radians)
# renvoie la distance des points à la meilleure droite (permet de selectionner facilement in/outliers
def ransac(paras, n_iter: int, max_dist: float, max_angle: float):
    sin_max_angle = math.sin(max_angle)
    best_thermique = Thermique()
    for i in range(n_iter):
        # tirage aléatoire de deux points de coords
        select = np.random.choice(np.arange(len(paras)), size=2, replace=False)
        para1 = paras[select[0]]
        para2 = paras[select[1]]
        center = intersect(para1, para2)
        if center is None: # rayons //
            continue
        r1 = np.linalg.norm(para1.pos - center)
        r2 = np.linalg.norm(para2.pos - center)
        if abs(r1 - r2) > max_dist:
            continue  # les rayons sont trop différents pour être sur le même cercle
        thermique = Thermique(center, 0.5 * (r1 + r2))
        inliers = []
        outliers = []
        for para in paras:
            v = para.pos - thermique.pos
            dist = np.linalg.norm(v)
            # print(para, thermique, v, dist)
            # la distance au centre et la direction du para sont compatibles avec ce thermique
            if abs(dist - thermique.r) < max_dist and abs(np.dot(v, unit(para.dir))) < dist * sin_max_angle:
                inliers.append(para)
            else:
                outliers.append(para)

        if len(inliers) > len(best_thermique.inliers):
            print("Trouvé un meilleur thermique %s avec %s paras" % (thermique, len(inliers)))
            best_thermique = copy.copy(thermique)
            best_thermique.inliers = copy.copy(inliers)
            best_thermique.outliers = copy.copy(outliers)

    return best_thermique


# Detection de droites multiples en itérant RANSAC
# Renvoie l'ensemble des droites trouvées, définies par un ensemble de segments (chacune)
def multi_ransac(paras: np.array, n_iter: int, max_dist: float, max_angle: float, n_min: int):
    ret = []
    n_inliers = n_min + 1
    while n_inliers > n_min and len(paras) > n_min:
        print('RANSAC itération %s, reste %s points non attribués' % (len(ret) + 1, len(paras)))
        thermique = ransac(paras, n_iter, max_dist, max_angle)
        n_inliers = len(thermique.inliers)
        if n_inliers > n_min:
            ret.append(thermique)
            paras = thermique.outliers
    return ret


if __name__ == "__main__":
    paras = []
    d_max = 100  # distance max d'un para/thermique au point (0,0)
    n_thrm = 3  # nombre de thermiques
    n_para = 9  # nombre de paras par thermique
    n_outliers = 24  # nombre de paras pas dans un thermique
    pos_noise = 2  # bruit sur la position d'un para
    dir_noise = 0.1  # bruit sur la direction d'un para
    r_av_thrm = 15  # rayon moyen d'un thermique
    r_var_thrm = 5  # variance sur le rayon d'un thermique

    thermiques = [Thermique(np.random.normal(0, d_max, 2), np.random.normal(r_av_thrm, r_var_thrm)) for i in range(n_thrm)]
    for thermique in thermiques:
        for i in range(n_para):
            dir = 2*math.pi*random.random()
            perturb = nr.normal(0, pos_noise, 2)
            paras.append(Para(thermique.pos + thermique.r * unit(dir-math.pi/2) + perturb,
                              dir + nr.normal(0, dir_noise)))
    for i in range(n_outliers):
        paras.append(Para(nr.normal(0, d_max, 2), 2*math.pi*random.random()))

    thermiques_estimes = multi_ransac(paras, n_iter, max_dist, max_angle, n_min)
    # dessiner les thermiques
    print('Vérité:')
    for thermique in thermiques:
        print(thermique)
    print('Estimé:')
    for thermique in thermiques_estimes:
        print(thermique)
        for para in thermique.inliers:
            plt.plot([para.pos[0], thermique.pos[0]], [para.pos[1], thermique.pos[1]], color='red')
    # dessiner les paras
    for para in paras:
        radial = unit(para.dir-math.pi/2)
        axial = unit(para.dir)
        tri = [para.pos+4*radial, para.pos-4*radial, para.pos+2*axial, para.pos+4*radial]
        plt.plot([p[0] for p in tri], [p[1] for p in tri], color='blue')
    plt.show()

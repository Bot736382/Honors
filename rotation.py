import numpy as np
import matplotlib.pyplot as plt

# number of points
n = 50

# parameter t
t = np.linspace(0, 2*np.pi, n, endpoint=False)

# generate an arbitrary closed curve
x = 1.2*np.cos(t) + 0.3*np.sin(3*t) + 0.15*np.cos(5*t + 1.0) + 10
y = 0.8*np.sin(t) + 0.25*np.cos(4*t) + 0.1*np.sin(6*t + 0.7) + 5

pts = np.vstack([x, y]).T
mean_pts = np.mean(pts, axis=0)

# correct rotation matrix
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
R = np.array([[c, -s],
              [s,  c]])

# rotate around origin (this is what your pts2 intended)
pts2 = (R @ pts.T).T
mean_pts2 = np.mean(pts2, axis=0)

# rotate around centroid (this is pts3)
pts3 = pts - mean_pts
pts3 = (R @ pts3.T).T
pts3 += mean_pts
mean_pts3 = np.mean(pts3, axis=0)

# PLOT
plt.figure(figsize=(6,6))
plt.plot(pts[:,0],  pts[:,1],  '-o', label='Original')
plt.plot(pts2[:,0], pts2[:,1], '-o', label='Rotated about origin')
plt.plot(pts3[:,0], pts3[:,1], '-o', label='Rotated about centroid')

# draw centroid vectors
plt.plot([0, mean_pts[0]],  [0, mean_pts[1]],  'r--')
plt.plot([0, mean_pts2[0]], [0, mean_pts2[1]], 'g--')
plt.plot([0, mean_pts3[0]], [0, mean_pts3[1]], 'b--')

plt.legend()
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Arbitrary Closed Curve With Rotations")
plt.show()

# core/projection.py
import numpy as np

def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy) / 2)
    mat = np.zeros((4,4), dtype=np.float32)

    mat[0,0] = f / aspect
    mat[1,1] = f
    mat[2,2] = (far + near) / (near - far)
    mat[2,3] = (2 * far * near) / (near - far)
    mat[3,2] = -1.0
    return mat

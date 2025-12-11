# core/model_cube.py
from pyglet.gl import *
import numpy as np
from ctypes import c_void_p

def build_cube():
    # Position (x,y,z), Normal (nx,ny,nz), UV (u,v)
    vertices = np.array([
        # --- Front face ---
        -0.5, 0.0,  0.5,   0,0,1,   0,0,
         0.5, 0.0,  0.5,   0,0,1,   1,0,
         0.5, 1.0,  0.5,   0,0,1,   1,1,

        -0.5, 0.0,  0.5,   0,0,1,   0,0,
         0.5, 1.0,  0.5,   0,0,1,   1,1,
        -0.5, 1.0,  0.5,   0,0,1,   0,1,

        # --- Back face ---
        -0.5, 0.0, -0.5,   0,0,-1,   1,0,
         0.5, 0.0, -0.5,   0,0,-1,   0,0,
         0.5, 1.0, -0.5,   0,0,-1,   0,1,

        -0.5, 0.0, -0.5,   0,0,-1,   1,0,
         0.5, 1.0, -0.5,   0,0,-1,   0,1,
        -0.5, 1.0, -0.5,   0,0,-1,   1,1,

        # --- Left face ---
        -0.5, 0.0, -0.5,  -1,0,0,   0,0,
        -0.5, 0.0,  0.5,  -1,0,0,   1,0,
        -0.5, 1.0,  0.5,  -1,0,0,   1,1,

        -0.5, 0.0, -0.5,  -1,0,0,   0,0,
        -0.5, 1.0,  0.5,  -1,0,0,   1,1,
        -0.5, 1.0, -0.5,  -1,0,0,   0,1,

        # --- Right face ---
         0.5, 0.0, -0.5,   1,0,0,   1,0,
         0.5, 0.0,  0.5,   1,0,0,   0,0,
         0.5, 1.0,  0.5,   1,0,0,   0,1,

         0.5, 0.0, -0.5,   1,0,0,   1,0,
         0.5, 1.0,  0.5,   1,0,0,   0,1,
         0.5, 1.0, -0.5,   1,0,0,   1,1,

        # --- Top face ---
        -0.5, 1.0, -0.5,   0,1,0,   0,1,
         0.5, 1.0, -0.5,   0,1,0,   1,1,
         0.5, 1.0,  0.5,   0,1,0,   1,0,

        -0.5, 1.0, -0.5,   0,1,0,   0,1,
         0.5, 1.0,  0.5,   0,1,0,   1,0,
        -0.5, 1.0,  0.5,   0,1,0,   0,0,

        # --- Bottom face ---
        -0.5, 0.0, -0.5,   0,-1,0,   0,0,
         0.5, 0.0, -0.5,   0,-1,0,   1,0,
         0.5, 0.0,  0.5,   0,-1,0,   1,1,

        -0.5, 0.0, -0.5,   0,-1,0,   0,0,
         0.5, 0.0,  0.5,   0,-1,0,   1,1,
        -0.5, 0.0,  0.5,   0,-1,0,   0,1,

    ], dtype=np.float32)

    vao = GLuint()
    glGenVertexArrays(1, vao)
    glBindVertexArray(vao)

    vbo = GLuint()
    glGenBuffers(1, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    stride = 8 * 4  # 8 floats per vertex

    # Position
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, c_void_p(0))

    # Normal
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, c_void_p(12))

    # UV
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, c_void_p(24))

    vertex_count = len(vertices) // 8
    return vao, vertex_count

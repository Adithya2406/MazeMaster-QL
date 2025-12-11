# renderer_3d.py

import math
import numpy as np
import pyglet
from pyglet.gl import *
from noise import pnoise2


# =====================================================
# Procedural Textures
# =====================================================
def generate_stone_texture(size=128):
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            n = pnoise2(i * 0.1, j * 0.1, octaves=3)
            v = int((n + 1.0) * 127.5)
            tex[i, j] = (v, v, v)
    return tex


def generate_wood_texture(size=128):
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            grain = int((math.sin(i * 0.25) + 1.0) * 80.0)  # 0–160
            r = min(139 + grain, 255)
            g = min(69 + grain // 2, 255)
            b = 19
            tex[i, j] = (r, g, b)
    return tex


def create_gl_texture(image):
    h, w = image.shape[:2]
    data = (GLubyte * (h * w * 3))(*image.flatten())

    tex = GLuint()
    glGenTextures(1, tex)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        w,
        h,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        data,
    )
    return tex


# =====================================================
# 3D Maze Renderer
# =====================================================
class MazeRenderer3D:
    def __init__(self, maze, width=1280, height=720, window_visible=False):
        self.maze = np.array(maze, copy=True)
        self.H, self.W = self.maze.shape
        self.win_w = width
        self.win_h = height

        config = pyglet.gl.Config(
            major_version=2,
            minor_version=1,
            depth_size=24,
            double_buffer=True,
        )

        self.window = pyglet.window.Window(
            width,
            height,
            caption="Maze 3D",
            resizable=False,
            visible=window_visible,
            config=config,
        )
        self.window.switch_to()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)

        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(
            GL_LIGHT0,
            GL_POSITION,
            (GLfloat * 4)(2.5, 6.0, 2.5, 1.0),
        )
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(1, 1, 1, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.25, 0.25, 0.25, 1))

        # Textures
        self.tex_wall = create_gl_texture(generate_stone_texture())
        self.tex_floor = create_gl_texture(generate_wood_texture())

    # -------------------------------------------------
    def draw_textured_cube(self, tex, x, z, h=1.0, color=(1, 1, 1)):
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor3f(*color)

        glBegin(GL_QUADS)

        # South
        glTexCoord2f(0, 0); glVertex3f(x, 0, z + 1)
        glTexCoord2f(1, 0); glVertex3f(x + 1, 0, z + 1)
        glTexCoord2f(1, 1); glVertex3f(x + 1, h, z + 1)
        glTexCoord2f(0, 1); glVertex3f(x, h, z + 1)

        # North
        glTexCoord2f(0, 0); glVertex3f(x + 1, 0, z)
        glTexCoord2f(1, 0); glVertex3f(x, 0, z)
        glTexCoord2f(1, 1); glVertex3f(x, h, z)
        glTexCoord2f(0, 1); glVertex3f(x + 1, h, z)

        # East
        glTexCoord2f(0, 0); glVertex3f(x + 1, 0, z + 1)
        glTexCoord2f(1, 0); glVertex3f(x + 1, 0, z)
        glTexCoord2f(1, 1); glVertex3f(x + 1, h, z)
        glTexCoord2f(0, 1); glVertex3f(x + 1, h, z + 1)

        # West
        glTexCoord2f(0, 0); glVertex3f(x, 0, z)
        glTexCoord2f(1, 0); glVertex3f(x, 0, z + 1)
        glTexCoord2f(1, 1); glVertex3f(x, h, z + 1)
        glTexCoord2f(0, 1); glVertex3f(x, h, z)

        # Top
        glTexCoord2f(0, 0); glVertex3f(x, h, z)
        glTexCoord2f(1, 0); glVertex3f(x + 1, h, z)
        glTexCoord2f(1, 1); glVertex3f(x + 1, h, z + 1)
        glTexCoord2f(0, 1); glVertex3f(x, h, z + 1)

        glEnd()

    # -------------------------------------------------
    def draw_world(self, visited=None):
        # Floor tiles
        for r in range(self.H):
            for c in range(self.W):
                if self.maze[r, c] == 0:
                    glBindTexture(GL_TEXTURE_2D, self.tex_floor)
                    if visited is not None and visited[r, c]:
                        glColor3f(0.5, 1.0, 0.5)  # explored
                    else:
                        glColor3f(1.0, 1.0, 1.0)

                    glBegin(GL_QUADS)
                    glTexCoord2f(0, 0); glVertex3f(c, 0, r)
                    glTexCoord2f(1, 0); glVertex3f(c + 1, 0, r)
                    glTexCoord2f(1, 1); glVertex3f(c + 1, 0, r + 1)
                    glTexCoord2f(0, 1); glVertex3f(c, 0, r + 1)
                    glEnd()

        # Walls
        glColor3f(1.0, 1.0, 1.0)
        for r in range(self.H):
            for c in range(self.W):
                if self.maze[r, c] == 1:
                    self.draw_textured_cube(self.tex_wall, c, r, h=1.0)

    # -------------------------------------------------
    def set_view_topdown(self):
        gluLookAt(
            self.W / 2.0,
            35.0,
            self.H / 2.0,
            self.W / 2.0,
            0.0,
            self.H / 2.0,
            0.0,
            0.0,
            -1.0,
        )

    def set_view_firstperson(self, cam_x, cam_z, cam_rot_deg):
        lx = math.sin(math.radians(cam_rot_deg))
        lz = math.cos(math.radians(cam_rot_deg))

        gluLookAt(
            cam_x,
            0.3,
            cam_z,
            cam_x + lx,
            0.3,
            cam_z + lz,
            0.0,
            1.0,
            0.0,
        )

    # -------------------------------------------------
    def render_frame(self, mode, visited=None, agent_pos=None, cam_rot=0.0):
        """
        mode: "topdown" or "firstperson"
        visited: HxW bool array
        agent_pos: (x, z) in grid coordinates (center of cell)
        cam_rot: yaw angle in degrees
        """
        self.window.switch_to()
        self.window.dispatch_events()

        glViewport(0, 0, self.win_w, self.win_h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, self.win_w / float(self.win_h), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if mode == "topdown":
            self.set_view_topdown()
        elif mode == "firstperson":
            if agent_pos is None:
                cam_x, cam_z = (self.W / 2.0, self.H / 2.0)
            else:
                cam_x, cam_z = agent_pos
            self.set_view_firstperson(cam_x, cam_z, cam_rot)

        self.draw_world(visited=visited)

        # Read pixels back
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        img = buffer.get_image_data()
        arr = np.frombuffer(
            img.get_data("RGB", self.win_w * 3), dtype=np.uint8
        )
        frame = arr.reshape(self.win_h, self.win_w, 3)
        return np.flipud(frame)

    # -------------------------------------------------
    def close(self):
        try:
            self.window.close()
        except Exception:
            pass

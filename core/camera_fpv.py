# core/camera_fpv.py
import numpy as np
import pyglet

class CameraFPV:
    def __init__(self, window, sensitivity=0.002, speed=0.1):
        self.window = window
        self.yaw = 0.0
        self.pitch = 0.0
        self.speed = speed
        self.sensitivity = sensitivity

        self.pos = np.array([1.0, 0.5, 1.0], dtype=np.float32)

        window.push_handlers(self)

        self.keys = pyglet.window.key.KeyStateHandler()
        window.push_handlers(self.keys)

        self.center = window.width//2, window.height//2
        window.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = np.clip(self.pitch, -1.2, 1.2)

    def update(self):
        fwd = np.array([
            np.cos(self.pitch)*np.sin(self.yaw),
            0,
            np.cos(self.pitch)*np.cos(self.yaw)
        ])

        right = np.array([
            np.cos(self.yaw), 0, -np.sin(self.yaw)
        ])

        if self.keys[pyglet.window.key.W]: self.pos += fwd * self.speed
        if self.keys[pyglet.window.key.S]: self.pos -= fwd * self.speed
        if self.keys[pyglet.window.key.A]: self.pos -= right * self.speed
        if self.keys[pyglet.window.key.D]: self.pos += right * self.speed

    def get_view_matrix(self):
        fwd = np.array([
            np.cos(self.pitch)*np.sin(self.yaw),
            np.sin(self.pitch),
            np.cos(self.pitch)*np.cos(self.yaw)
        ])

        target = self.pos + fwd
        up = np.array([0, 1, 0])

        return lookAt(self.pos, target, up)


def lookAt(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)

    r = np.cross(f, up)
    r /= np.linalg.norm(r)

    u = np.cross(r, f)

    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = r
    mat[1, :3] = u
    mat[2, :3] = -f

    mat[:3, 3] = -eye @ mat[:3, :3]

    return mat

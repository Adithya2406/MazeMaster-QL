import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

class FPSCamera:
    def __init__(self, pos=np.array([1.5,1.0,1.5],dtype=float)):
        self.position = pos
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 0.07

    def view_matrix(self):
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ])
        front = normalize(front)
        right = normalize(np.cross(front, np.array([0,1,0])))
        up    = normalize(np.cross(right, front))

        eye = self.position
        target = self.position + front

        return look_at(eye, target, up)

class ThirdPersonCamera:
    def __init__(self, target=np.array([1.5,0.5,1.5]), dist=6):
        self.target = target
        self.dist = dist
        self.angle = 45.0

    def update_target(self, tgt):
        self.target = np.array([tgt[0], 0.5, tgt[1]])

    def view_matrix(self):
        camx = self.target[0] + self.dist*np.cos(np.radians(self.angle))
        camz = self.target[2] + self.dist*np.sin(np.radians(self.angle))
        eye = np.array([camx, 6.0, camz])
        up = np.array([0,1,0])
        return look_at(eye, self.target, up)

# ---------------- matrix helper --------------------

def look_at(eye, target, up):
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4)
    m[0,:3] = s
    m[1,:3] = u
    m[2,:3] = -f
    m[0,3] = -np.dot(s, eye)
    m[1,3] = -np.dot(u, eye)
    m[2,3] =  np.dot(f, eye)
    return m.astype(np.float32)

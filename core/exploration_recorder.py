# exploration_recorder.py
import imageio
import numpy as np
from core.projection import perspective
from core.camera_fpv import CameraFPV

class ExplorationRecorder:
    def __init__(self, renderer, fps=30):
        self.renderer = renderer
        self.fps = fps

    # -------------------------------------------------------------

    def record_topdown(self, env, visited, outfile="exploration3d_topdown.gif"):
        frames = []

        # top-down camera: orthographic above maze
        H, W = env.H, env.W

        eye = np.array([W/2, max(W,H)*1.2, H/2], dtype=np.float32)
        target = np.array([W/2, 0, H/2], dtype=np.float32)
        up = np.array([1,0,0], dtype=np.float32)

        def lookAt(eye, target, up):
            f = target - eye
            f = f / np.linalg.norm(f)
            r = np.cross(f, up)
            r = r / np.linalg.norm(r)
            u = np.cross(r, f)

            mat = np.eye(4, dtype=np.float32)
            mat[0,:3] = r
            mat[1,:3] = u
            mat[2,:3] = -f
            mat[:3,3] = -eye @ mat[:3,:3]
            return mat

        view = lookAt(eye, target, up)
        proj = perspective(60, 1280/720, 0.1, 200.0)

        # Generate frames by replaying visited path
        for step, pos in enumerate(visited):
            ax, az = pos if len(pos)==2 else pos[::-1]
            gx, gz = env.goal

            img = self.renderer.render_to_image(
                view, proj,
                agent_pos=(ax, az),
                goal_pos=(gx, gz)
            )
            frames.append(img)

        imageio.mimsave(outfile, frames, fps=self.fps)
        print("Saved:", outfile)

    # -------------------------------------------------------------

    def record_firstperson(self, env, visited, outfile="exploration3d_fpv.gif"):
        frames = []
        cam = CameraFPV(self.renderer.window)
        proj = perspective(75, 1280/720, 0.05, 200.0)

        # Disable real-time input
        self.renderer.window.set_exclusive_mouse(False)

        # Replay visited cells as agent movement
        for pos in visited:
            ax, az = pos if len(pos)==2 else pos[::-1]

            # Simulate camera following agent
            cam.pos = np.array([ax + 0.5, 0.4, az + 0.5], dtype=np.float32)

            # Look slightly forward
            cam.yaw = 0.0
            cam.pitch = 0.0

            view = cam.get_view_matrix()
            gx, gz = env.goal

            img = self.renderer.render_to_image(
                view, proj,
                agent_pos=(ax, az),
                goal_pos=(gx, gz)
            )
            frames.append(img)

        imageio.mimsave(outfile, frames, fps=self.fps)
        print("Saved:", outfile)

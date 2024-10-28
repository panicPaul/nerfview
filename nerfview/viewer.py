import dataclasses
import time
from threading import Lock
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import viser
import viser.transforms as vt
from jaxtyping import Float32, UInt8

from ._renderer import Renderer, RenderTask


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]

    def get_K(self, img_wh: Tuple[int, int]) -> Float32[np.ndarray, "3 3"]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array([
            [focal_length, 0.0, W / 2.0],
            [0.0, focal_length, H / 2.0],
            [0.0, 0.0, 1.0],
        ])
        return K


@dataclasses.dataclass
class ViewerState(object):
    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0
    status: Literal["rendering", "preparing", "training", "paused", "completed"] = ("training")
    render_mode: Literal["color", "depth"] = "color"
    depth_bounds: tuple[float, float] = (0.0, 1.0)


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:

    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer(object):
    """This is the main class for working with nerfview viewer.

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution and the current frame as input and returns an image as a
            uint8 numpy array. Optionally, it can return a tuple of two images, where
            the second image is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int], int],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        mode: Literal["rendering", "training"] = "rendering",
        num_frames: int = 1,
    ):
        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = ViewerState()
        if self.mode == "rendering":
            self.state.status = "rendering"
        self.current_frame = 0
        self.num_frames = num_frames

        # Private states.
        self._renderers: dict[int, Renderer] = {}
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self._define_guis()

    def _define_guis(self):
        # Stats
        with self.server.gui.add_folder(
                "Stats", visible=self.mode == "training") as self._stats_folder:
            self._stats_text_fn = (lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step}
                </sub>""")
            self._stats_text = self.server.gui.add_markdown(self._stats_text_fn())

        # Training
        with self.server.gui.add_folder(
                "Training", visible=self.mode == "training") as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9)
            self._train_util_slider.on_update(self.rerender)

        # Rendering
        with self.server.gui.add_folder("Rendering") as self._rendering_folder:
            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048)
            self._max_img_res_slider.on_update(self.rerender)

        # Playback control
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Timestep",
                min=0,
                max=self.num_frames - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            self.gui_timestep.on_update(self.rerender)
            self.gui_next_frame = self.server.gui.add_button("Next Frame", disabled=True)
            self.gui_prev_frame = self.server.gui.add_button("Prev Frame", disabled=True)
            self.gui_playing = self.server.gui.add_checkbox("Playing", True)
            self.gui_playing.on_update(self.rerender)
            self.gui_framerate = self.server.gui.add_slider(
                "FPS", min=1, max=24, step=0.1, initial_value=24)
            self.gui_framerate.on_update(self.rerender)
            self.gui_framerate_options = self.server.gui.add_button_group(
                "Playback Speed", ("0.25x", "0.5x", "1x"))
            self.gui_framerate_options.on_click(self.rerender)

        # depth button and depth bounds
        with self.server.gui.add_folder("Depth"):
            self.render_depth_button = self.server.gui.add_button("Render Depth")
            self.render_depth_button.on_click(self._toggle_render_depth_button)
            self.render_depth_button.on_click(self._toggle_render_depth_state)
            self.render_depth_button.on_click(self.rerender)
            self.render_color_button = self.server.gui.add_button("Render Color")
            self.render_color_button.visible = False
            self.render_color_button.on_click(self._toggle_render_depth_button)
            self.render_color_button.on_click(self._toggle_render_depth_state)
            self.render_color_button.on_click(self.rerender)
            self.depth_bounds = self.server.gui.add_vector2(
                "Depth Bounds", hint="min, max", initial_value=(0.0, 1.0))

        @self.depth_bounds.on_update
        def _(_) -> None:
            self.state.depth_bounds = self.depth_bounds.value
            self.rerender()

        @self.gui_next_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames

        @self.gui_prev_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % self.num_frames

        # Disable frame controls when we're playing.
        @self.gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = self.gui_playing.value
            self.gui_next_frame.disabled = self.gui_playing.value
            self.gui_prev_frame.disabled = self.gui_playing.value

        # Set the framerate when we click one of the options.
        @self.gui_framerate_options.on_click
        def _(_) -> None:
            self.gui_framerate.value = int(float(self.gui_framerate_options.value[:-1]) * 24)

    def _toggle_train_buttons(self, _):
        self._pause_train_button.visible = not self._pause_train_button.visible
        self._resume_train_button.visible = not self._resume_train_button.visible

    def _toggle_train_s(self, _):
        if self.state.status == "completed":
            return
        self.state.status = "paused" if self.state.status == "training" else "training"

    def _toggle_render_depth_button(self, _):
        self.render_depth_button.visible = not self.render_depth_button.visible
        self.render_color_button.visible = not self.render_color_button.visible

    def _toggle_render_depth_state(self, _):
        self.state.render_mode = "depth" if self.state.render_mode == "color" else "color"

    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(viewer=self, client=client, lock=self.lock)
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate([vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update(self, step: int, num_train_rays_per_step: int):
        if self.mode == "rendering":
            raise ValueError("`update` method is only available in training mode.")
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self._renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if self.state.status == "training" and self._train_util_slider.value != 1:
            assert (
                self.state.num_train_rays_per_sec
                is not None), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.state.num_train_rays_per_sec
            view_s = self.state.num_view_rays_per_sec
            train_util = self._train_util_slider.value
            view_n = self._max_img_res_slider.value**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = (train_util * view_time / (train_time - train_util*train_time))
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.server.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    assert camera_state is not None
                    self._renderers[client_id].submit(RenderTask("update", camera_state))
                with self.server.atomic(), self._stats_folder:
                    self._stats_text.content = self._stats_text_fn()

    def complete(self):
        self.state.status = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        self._train_util_slider.disabled = True
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""

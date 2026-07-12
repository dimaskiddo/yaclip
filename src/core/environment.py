from __future__ import annotations

import atexit
import ctypes
import importlib.util
import os
import shutil
import sys
from pathlib import Path

from loguru import logger

from src.core.constants import MEDIAPIPE_GL_LIBS
from src.core.exceptions import DetectionError
from src.core.workspace import BIN_DIR, MODELS_DIR, TMP_DIR


def _has_usable_gpu() -> bool:
    """Return True only when a GPU is actually accessible at runtime.

    Deliberately avoids importing torch (heavy, lazy-load per AGENTS §11.3).
    WSL2 exposes /dev/dxg for the DirectX paravirt GPU; bare Linux exposes
    /dev/nvidia0.  nvidia-smi is the final fallback.
    """
    if Path("/dev/nvidia0").exists():
        return True
    if Path("/dev/dxg").exists():
        # /dev/dxg exists on WSL2 even without a CUDA-capable GPU, so do a
        # quick readability check: if we can stat it we likely have D3D access,
        # but for our purposes (blocking the triton crash) we treat WSL as
        # CPU-only unless the user explicitly opts in via YACLIP_FORCE_TRITON.
        # This is the conservative choice — a user with CUDA on WSL sets the
        # opt-in env var (see guard_triton_segfault below).
        return False
    return shutil.which("nvidia-smi") is not None


def guard_triton_segfault() -> None:
    """Mask the `triton` package when it cannot be safely imported.

    Background
    ----------
    The CUDA build of torch ships `triton`. On CPU-only / WSL environments,
    importing triton *after* MediaPipe has run its native FaceDetector
    inference (which loads LLVM/abseil/XNNPACK native libs) causes a SIGSEGV
    inside triton/knobs.py — a hard native crash with no Python traceback.
    torch's own `has_triton_package()` guard catches only `ImportError`, not
    SIGSEGV, so it cannot protect itself.

    This function runs at startup, before any vision or torch work, and
    installs a null sentinel (`sys.modules["triton"] = None`) so that any
    subsequent `import triton` raises `ImportError` instead of segfaulting.
    torch's guard then correctly returns False and skips triton-only paths.

    Opt-out
    -------
    Users with a real GPU who deliberately install a CUDA torch build can set
    the environment variable ``YACLIP_FORCE_TRITON=1`` to disable this guard.

    No-op paths
    -----------
    - triton is not installed (CPU torch build, the recommended default) → return.
    - GPU is detected and YACLIP_FORCE_TRITON is not set → return (GPU is real,
      triton is safe to use). This branch is conservative: /dev/dxg on WSL is
      *not* treated as a real CUDA GPU (see _has_usable_gpu).
    """
    if importlib.util.find_spec("triton") is None:
        # CPU torch build: triton not installed, nothing to do.
        return

    if os.environ.get("YACLIP_FORCE_TRITON", "0") == "1":
        # User explicitly opted in (e.g. real CUDA GPU on bare Linux).
        return

    if _has_usable_gpu():
        # A real GPU device node is present: triton is safe to load.
        return

    # CUDA torch installed but no GPU accessible (common on CPU-only WSL).
    # Mask triton so torch's has_triton_package() returns False via ImportError
    # instead of crashing with SIGSEGV.
    sys.modules["triton"] = None  # type: ignore[assignment]
    logger.info(
        "GPU acceleration (triton) disabled — no compatible GPU detected. Running in CPU-only mode. "
        "Set YACLIP_FORCE_TRITON=1 if you have a real CUDA-capable GPU. "
        "To remove this message, install the CPU-only version of PyTorch (see README for instructions)."
    )


def setup_environment() -> None:
    """Inject required environment variables for paths, caches, and log suppression."""

    # Nuke any leftover Gradio file-copy directory from a prior session so we
    # start clean before Gradio even bootstraps (the env var below then
    # redirects new copies to our workspace path).
    _cleanup_gradio_cache()

    # Redirect Gradio's file-copy cache from /tmp/gradio to workspace/tmp/gradio
    # so that rendered video proxies don't bloat the system temp directory.
    os.environ["GRADIO_TEMP_DIR"] = str(TMP_DIR / "gradio")

    # AI Model Cache Path
    hf_workspace_dir = MODELS_DIR.resolve()
    os.environ["HF_HOME"] = str(hf_workspace_dir)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"

    # Binaries Path (prioritize our local FFmpeg cache)
    bin_dir = BIN_DIR.resolve()
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    # Suppress verbose C++ stderr logging from MediaPipe and TensorFlow.
    # GLOG_minloglevel=3 silences everything below ERROR: the absl::InitializeLog warning,
    # face_landmarker_graph.cc info lines, inference_feedback_manager warnings, and the
    # "Created TensorFlow Lite XNNPACK delegate" info.  Without this, every FaceLandmarker
    # load dumps ~10 lines of C++ noise to stderr.
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["GLOG_alsologtostderr"] = "0"
    os.environ["GLOG_logtostderr"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Silence libav H.264 decoder stderr ("mmco: unref short failure") emitted when OpenCV
    # random-seeks into a non-keyframe during frame sampling. Benign — the decoder recovers —
    # but 1080p DASH-merged videos have longer GOPs so seeks trip it often. -8 = AV_LOG_QUIET.
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"

    # Guard against the MediaPipe × triton SIGSEGV (see guard_triton_segfault docstring).
    guard_triton_segfault()


def _cleanup_gradio_cache() -> None:
    """Remove Gradio's file-copy directory on process exit.

    Registered as an atexit hook so that ``workspace/tmp/gradio/`` — where
    Gradio stashes copies of rendered clips for video serving — is cleaned up
    even when the user simply closes the WebUI without rendering new clips.
    """
    cache_dir = TMP_DIR / "gradio"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


atexit.register(_cleanup_gradio_cache)


def _gl_install_hint() -> str:
    """Return the distro-specific command to install MediaPipe's GL/EGL system libraries."""
    os_release = Path("/etc/os-release")
    ids = ""
    if os_release.exists():
        for line in os_release.read_text(encoding="utf-8").splitlines():
            if line.startswith(("ID=", "ID_LIKE=")):
                ids += " " + line.split("=", 1)[1].strip().strip('"').lower()

    if any(name in ids for name in ("debian", "ubuntu")):
        return "sudo apt-get install -y libegl1 libgles2 libgl1"
    if any(name in ids for name in ("fedora", "rhel", "centos")):
        return "sudo dnf install -y mesa-libEGL mesa-libGLES mesa-libGL"
    if "arch" in ids:
        return "sudo pacman -S --noconfirm libglvnd mesa"
    if "alpine" in ids:
        return "sudo apk add mesa-egl mesa-gles"
    if any(name in ids for name in ("opensuse", "suse")):
        return "sudo zypper install -y Mesa-libEGL1 Mesa-libGLESv2-2"
    return f"install your distro's packages providing {' and '.join(MEDIAPIPE_GL_LIBS)}"


def ensure_vision_runtime() -> None:
    """Verify MediaPipe's GL/EGL system libraries are loadable before the vision pipeline runs.

    Raises:
        DetectionError: if a required library is missing, with a distro-specific fix command.
    """
    # Only Linux MediaPipe wheels link these libraries; Windows/macOS wheels do not.
    if not sys.platform.startswith("linux"):
        return

    missing = []
    for lib in MEDIAPIPE_GL_LIBS:
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append(lib)

    if not missing:
        return

    raise DetectionError(
        "MediaPipe requires system OpenGL ES libraries that are missing: "
        f"{', '.join(missing)}.\n"
        f"Install them with:\n    {_gl_install_hint()}\n"
        "See the README 'System Dependencies' section for other distros, "
        "or run YaClip via the provided Dockerfile to avoid host setup entirely."
    )

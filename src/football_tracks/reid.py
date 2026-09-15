"""Stage 2c - what each detected player looks like, so two men in one tackle can be told apart.

A track breaks where players touch, and the box the detector draws there holds both of them,
so its position cannot say which fragment afterwards is whose (D94). The shirt cannot either
when they are team-mates. What can is how the player looks on the clean frames either side of
the contact -- build, hair, skin, boots -- which a person re-identification network is trained
to embed so that one person lands close to himself and far from everybody else (D95).

OSNet-AIN x1.0, trained on MSMT17. AIN is the variant built to generalise to footage it was not
trained on, which is what a broadcast is. The architecture is vendored (`osnet`, MIT); the
weights are not committed. They come from the author's own repository, are checked against a
pinned hash BEFORE torch reads them, and are loaded with `weights_only`, because a checkpoint
is a pickle and a pickle can run code.

torch is imported inside the functions that need it, as in `detect`. Embeddings are cached to
work/<clip>/appearance.npz, one row per detection, keyed by the detection they came from.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .detect import Detection, device, trust_certifi

WEIGHTS_FILE = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr"
    "_b64_fb10_softmax_labsmth_flip_jitter.pth"
)
WEIGHTS_URL = f"https://huggingface.co/kaiyangzhou/osnet/resolve/main/{WEIGHTS_FILE}"
# The SHA-256 HuggingFace publishes for that file. Checked before every load, not only after
# a download, so a checkpoint swapped on disk is refused the same way.
WEIGHTS_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"

# What the network was trained on: 256 x 128 crops under ImageNet normalisation.
CROP_W, CROP_H = 128, 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FEATURES = 512
BATCH = 256

# Clean crops averaged into one look at each end of a fragment. Eight is a third of a second at
# 25 fps: enough to average out a turned head, near enough the break to be the same moment.
GALLERY = 8

Vec = npt.NDArray[np.float32]
Key = tuple[int, float, float, float, float]


def key(d: Detection) -> Key:
    return (d.f, d.x1, d.y1, d.x2, d.y2)


def crop(bgr: Any, d: Detection) -> Vec | None:
    """A detection as the network's input: RGB, 256 x 128, normalised, channels first."""
    h, w = bgr.shape[:2]
    x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
    x2, y2 = min(w, int(d.x2)), min(h, int(d.y2))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    rgb = cv2.cvtColor(cv2.resize(bgr[y1:y2, x1:x2], (CROP_W, CROP_H)), cv2.COLOR_BGR2RGB)
    out: Vec = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights(directory: Path) -> Path:
    """The checkpoint, downloaded on first use and verified before every load."""
    path = directory / WEIGHTS_FILE
    if not path.exists():
        trust_certifi()
        directory.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(WEIGHTS_URL, partial)
        if _sha256(partial) != WEIGHTS_SHA256:
            partial.unlink()
            raise RuntimeError(f"{WEIGHTS_URL} did not match its pinned sha256; nothing kept")
        partial.rename(path)
    digest = _sha256(path)
    if digest != WEIGHTS_SHA256:
        raise RuntimeError(f"{path} has sha256 {digest}, expected {WEIGHTS_SHA256}; not loading it")
    return path


def load_model(directory: Path, dev: str | None = None) -> tuple[Any, str]:
    """The network with its weights, in eval mode, and the device it is on."""
    import torch

    from .osnet import OSNet

    dev = dev or device()
    checkpoint = torch.load(weights(directory), map_location="cpu", weights_only=True)
    state = {
        k.removeprefix("module."): v for k, v in checkpoint.get("state_dict", checkpoint).items()
    }
    model: Any = OSNet()
    # Strict: the classifier head is the only thing upstream has that this does not.
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("classifier.")})
    return model.eval().to(dev), dev


def run(
    frames_dir: Path,
    detections: list[Detection],
    directory: Path,
    progress: Callable[[int], None] | None = None,
) -> tuple[Vec, npt.NDArray[np.bool_]]:
    """One unit embedding per detection, in their order, and which rows hold one."""
    import torch

    model, dev = load_model(directory)
    features = np.zeros((len(detections), FEATURES), dtype=np.float32)
    valid = np.zeros(len(detections), dtype=bool)
    by_frame: dict[int, list[int]] = {}
    for i, d in enumerate(detections):
        by_frame.setdefault(d.f, []).append(i)
    pending: list[tuple[int, Vec]] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.from_numpy(np.stack([c for _, c in pending])).to(dev)
        with torch.no_grad():
            out = torch.nn.functional.normalize(model(batch), dim=1).cpu().numpy()
        for (i, _), v in zip(pending, out, strict=True):
            features[i], valid[i] = v, True
        pending.clear()

    for f in sorted(by_frame):
        bgr = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        if bgr is not None:
            for i in by_frame[f]:
                c = crop(bgr, detections[i])
                if c is not None:
                    pending.append((i, c))
                if len(pending) == BATCH:
                    flush()
        if progress:
            progress(f)
    flush()
    return features, valid


def write(
    path: Path, detections: list[Detection], features: Vec, valid: npt.NDArray[np.bool_]
) -> Path:
    keys = np.array([key(d) for d in detections], dtype=np.float64).reshape(-1, 5)
    with path.open("wb") as fh:
        np.savez_compressed(fh, keys=keys, features=features, valid=valid)
    return path


def read(path: Path, detections: list[Detection]) -> dict[Key, Vec] | None:
    """Embeddings by detection, or None where there are none or they are of other detections.

    The same trap as every cache keyed by frame (see `ft frames`): embeddings of a previous
    detection pass would attach one player's look to another's box.
    """
    if not path.exists():
        return None
    with np.load(path) as data:
        keys, features, valid = data["keys"], data["features"], data["valid"]
    ours = np.array([key(d) for d in detections], dtype=np.float64).reshape(-1, 5)
    if keys.shape != ours.shape or not np.allclose(keys, ours):
        return None
    return {key(d): features[i] for i, d in enumerate(detections) if valid[i]}


def look(features: list[Vec]) -> Vec | None:
    """One player's appearance over several crops: the mean of their unit embeddings, as a unit."""
    if not features:
        return None
    mean = np.mean([v / np.linalg.norm(v) for v in features], axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return None
    out: Vec = (mean / norm).astype(np.float32)
    return out


def distance(a: Vec, b: Vec) -> float:
    """Cosine distance between two looks: 0 is the same look, 1 is unrelated."""
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

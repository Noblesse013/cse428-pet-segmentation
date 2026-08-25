"""
Dataset acquisition for cloud notebooks.

The Oxford-IIIT Pet data is gitignored (it is ~800 MB), so a fresh Colab or
Kaggle session has the code but not the images.  This module fetches and
extracts the two official tarballs, then hands back the resolved paths.

Typical use inside a notebook:

    from src.data_setup import ensure_dataset
    image_dir, annotation_dir = ensure_dataset()

On Kaggle the usual route is to attach an existing Oxford-IIIT Pet dataset via
"+ Add Input" instead of downloading; `ensure_dataset` detects that case first
and skips the download entirely.
"""

import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

# Official mirrors, in the order they are tried.  The Oxford host is the
# canonical source; the second is a widely used HTTPS mirror that stays up
# when the Oxford server is slow.
ARCHIVES = {
    "images": [
        "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    ],
    "annotations": [
        "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
    ],
}


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}GB"


def _download(urls, dest: Path, retries: int = 2) -> Path:
    """Download the first URL that works, showing coarse progress."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [cached] {dest.name} ({_human(dest.stat().st_size)})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None

    for url in urls:
        for attempt in range(retries):
            try:
                print(f"  downloading {dest.name} from {url}")
                tmp = dest.with_suffix(dest.suffix + ".part")
                start = time.time()

                def _hook(block_num, block_size, total_size, _start=start):
                    if total_size <= 0:
                        return
                    done = min(block_num * block_size, total_size)
                    pct = 100.0 * done / total_size
                    if block_num % 200 == 0 or done == total_size:
                        print(
                            f"    {pct:5.1f}%  {_human(done)} / {_human(total_size)}"
                            f"  ({time.time() - _start:.0f}s)",
                            end="\r",
                        )

                urllib.request.urlretrieve(url, tmp, reporthook=_hook)
                print()
                tmp.replace(dest)
                return dest
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                last_error = exc
                print(f"    failed ({exc}); retrying" if attempt + 1 < retries
                      else f"    failed ({exc})")

    raise RuntimeError(
        f"Could not download {dest.name} from any mirror. Last error: {last_error}\n"
        "If your notebook has no outbound internet, enable it "
        "(Kaggle: Session options -> Internet ON) or attach the dataset "
        "manually with '+ Add Input'."
    )


def _extract(archive: Path, dest_root: Path) -> None:
    """Extract *archive* into *dest_root*, skipping work already done."""
    print(f"  extracting {archive.name} -> {dest_root}")
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # `filter="data"` is the safe extraction mode (Python 3.12+); older
        # versions extract the same members without the sanity checks.
        try:
            tar.extractall(dest_root, filter="data")
        except TypeError:
            tar.extractall(dest_root)


def download_oxford_pet(dest_root: Path, keep_archives: bool = False) -> Path:
    """Download + extract both tarballs under *dest_root*.

    Produces  dest_root/images/  and  dest_root/annotations/.
    """
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    images_dir = dest_root / "images"
    annotations_dir = dest_root / "annotations"

    if not (images_dir.is_dir() and any(images_dir.glob("*.jpg"))):
        archive = _download(ARCHIVES["images"], dest_root / "images.tar.gz")
        _extract(archive, dest_root)
        if not keep_archives:
            archive.unlink(missing_ok=True)
    else:
        print("  [skip] images already extracted")

    if not (annotations_dir / "trimaps").is_dir():
        archive = _download(ARCHIVES["annotations"], dest_root / "annotations.tar.gz")
        _extract(archive, dest_root)
        if not keep_archives:
            archive.unlink(missing_ok=True)
    else:
        print("  [skip] annotations already extracted")

    return dest_root


def default_data_root() -> Path:
    """Where to put a downloaded copy, per environment."""
    import config

    if config.IN_KAGGLE:
        # /kaggle/input is read-only; /kaggle/working is the writable scratch.
        return Path("/kaggle/working/data")
    if config.IN_COLAB:
        return Path("/content/data")
    return config.PROJECT_ROOT / "data"


def ensure_dataset(data_root: Optional[Path] = None,
                   force_download: bool = False) -> Tuple[Path, Path]:
    """Guarantee the dataset is present and return (image_dir, annotation_dir).

    Order of preference:
      1. an already-discovered copy (repo checkout, or a Kaggle attached input)
      2. download into *data_root* (defaults per environment)

    Also refreshes `config`'s module-level paths so code imported earlier in the
    notebook picks up the new location.
    """
    import config

    if not force_download:
        image_dir, annotation_dir = config.discover_dataset()
        if image_dir is not None:
            print(f"Dataset already available:\n  images      : {image_dir}\n"
                  f"  annotations : {annotation_dir}")
            config.refresh_dataset_paths()
            return config.IMAGE_DIR, config.ANNOTATION_DIR

    root = Path(data_root) if data_root is not None else default_data_root()
    print(f"Fetching Oxford-IIIT Pet into {root} (~800 MB, a few minutes)...")
    download_oxford_pet(root)

    if not config.refresh_dataset_paths():
        raise RuntimeError(
            f"Downloaded into {root} but could not locate images/ and "
            "annotations/trimaps/ afterwards. Inspect that directory."
        )

    print(f"Dataset ready:\n  images      : {config.IMAGE_DIR}\n"
          f"  annotations : {config.ANNOTATION_DIR}")
    return config.IMAGE_DIR, config.ANNOTATION_DIR


def dataset_summary() -> str:
    """Counts of images and trimaps, for a quick sanity check after setup."""
    import config

    n_images = len(list(config.IMAGE_DIR.glob("*.jpg"))) if config.IMAGE_DIR.is_dir() else 0
    n_trimaps = (
        len([p for p in config.TRIMAP_DIR.glob("*.png") if not p.name.startswith("._")])
        if config.TRIMAP_DIR.is_dir() else 0
    )
    return (f"images : {n_images} .jpg files\n"
            f"trimaps: {n_trimaps} .png files\n"
            f"(the official dataset has 7390 images and 7390 trimaps)")


def free_disk_space(path: Optional[Path] = None) -> str:
    """Report free space; Kaggle sessions cap out around 20 GB."""
    import config

    target = Path(path) if path is not None else config.OUTPUT_ROOT
    usage = shutil.disk_usage(target if target.exists() else target.parent)
    return f"{_human(usage.free)} free of {_human(usage.total)} at {target}"

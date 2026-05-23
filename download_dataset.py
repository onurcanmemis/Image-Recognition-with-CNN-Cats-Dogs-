"""Download the Oxford-IIIT Pet dataset and organize it into class folders.

The dataset is downloaded once to ./dataset_oxford_pets/ and laid out as:

    dataset_oxford_pets/
    ├── trainval/   # ~3,680 images — train_compare.py carves 20 % out for val
    │   ├── Abyssinian/
    │   ├── american_bulldog/
    │   └── ... (37 breed folders)
    └── test/       # ~3,669 images — never touched during training
        ├── Abyssinian/
        └── ...

The official trainval.txt / test.txt splits from the annotations tarball are
honoured. Breed name = the part of the filename before the trailing "_N.jpg",
so case is preserved (cat breeds Capitalised, dog breeds lowercase) and Keras's
alphabetical class assignment is determined by those folder names.

Idempotent: re-running this script after a successful download is a no-op.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_ROOT = Path(__file__).parent / "dataset_oxford_pets"
RAW_DIR = DATA_ROOT / "_raw"
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  ✓ already present: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))


def _extract(archive: Path, into: Path) -> None:
    print(f"  ⊞ extracting {archive.name} → {into.name}/")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(into)


def _breed_from_filename(name: str) -> str:
    """`Abyssinian_100.jpg` → `Abyssinian`; `american_bulldog_42.jpg` → `american_bulldog`."""
    stem = Path(name).stem
    breed, _, _ = stem.rpartition("_")
    return breed


def _read_split(split_file: Path) -> list[str]:
    """Each line: `<image_name> <class_id> <species> <breed_id>`. We only need image_name."""
    entries = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line.split()[0] + ".jpg")
    return entries


def _organize(split_name: str, image_names: list[str], images_dir: Path) -> None:
    out_root = DATA_ROOT / split_name
    if out_root.exists() and any(out_root.iterdir()):
        print(f"  ✓ {split_name}/ already populated")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    placed = 0
    missing = 0
    for img in tqdm(image_names, desc=f"  → {split_name}"):
        src = images_dir / img
        if not src.exists():
            missing += 1
            continue
        breed_dir = out_root / _breed_from_filename(img)
        breed_dir.mkdir(exist_ok=True)
        shutil.copy2(src, breed_dir / img)
        placed += 1
    print(f"    placed {placed}, missing {missing}")


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    images_tar = RAW_DIR / "images.tar.gz"
    annotations_tar = RAW_DIR / "annotations.tar.gz"

    print("1/4 download")
    _download(IMAGES_URL, images_tar)
    _download(ANNOTATIONS_URL, annotations_tar)

    print("2/4 extract")
    images_dir = RAW_DIR / "images"
    annotations_dir = RAW_DIR / "annotations"
    if not images_dir.exists():
        _extract(images_tar, RAW_DIR)
    if not annotations_dir.exists():
        _extract(annotations_tar, RAW_DIR)

    trainval_file = annotations_dir / "trainval.txt"
    test_file = annotations_dir / "test.txt"
    if not trainval_file.exists() or not test_file.exists():
        print(f"  ✗ split files missing under {annotations_dir}")
        return 1

    print("3/4 read splits")
    trainval_names = _read_split(trainval_file)
    test_names = _read_split(test_file)
    print(f"    trainval: {len(trainval_names)} images")
    print(f"    test:     {len(test_names)} images")

    print("4/4 organise into class folders")
    _organize("trainval", trainval_names, images_dir)
    _organize("test", test_names, images_dir)

    breeds = sorted({_breed_from_filename(n) for n in trainval_names})
    print(f"\n✓ ready — {len(breeds)} breed classes at {DATA_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

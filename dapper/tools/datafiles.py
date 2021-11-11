"""Manage the data files created by DAPPER."""

import os
import shutil
from datetime import datetime
from pathlib import Path

import dill
from tqdm.auto import tqdm

import dapper.tools.remote.uplink as uplink
from dapper.dpr_config import rc

XP_TIMESTAMP_TEMPLATE = "run_%Y-%m-%d__%H-%M-%S"


def create_run_dir(save_as, mp):
    """Validate `save_as` and create dir `rc.dirs.data / save_as` and sub-dirs.

    The data gets saved here unless `save_as` is `False`/`None`.

    Note: multiprocessing (locally or in the cloud) requires saving/loading data.
    """
    if save_as in [None, False]:
        assert not mp, "Multiprocessing requires saving data."
        # Parallelization w/o storing is possible, especially w/ threads.
        # But it involves more complicated communication set-up.
        def xpi_dir(*args): return None

    else:
        save_as = rc.dirs.data / Path(save_as).stem
        save_as /= datetime.now().strftime(XP_TIMESTAMP_TEMPLATE)
        os.makedirs(save_as)
        print(f"Experiment stored at {save_as}")

        def xpi_dir(i):
            path = save_as / str(i)
            os.mkdir(path)
            return path

    return save_as, xpi_dir


def find_latest_run(root: Path):
    """Find the latest experiment (dir containing many)"""
    def parse(d):
        try:
            return datetime.strptime(d.name, XP_TIMESTAMP_TEMPLATE)
        except ValueError:
            return None
    dd = [e for e in (parse(d) for d in root.iterdir()) if e is not None]
    d = max(dd)
    d = datetime.strftime(d, XP_TIMESTAMP_TEMPLATE)
    return d


def load_HMM(save_as):
    """Load HMM from `xp.com` from given dir."""
    save_as = Path(save_as).expanduser()
    HMM = dill.load(open(save_as/"xp.com", "rb"))["HMM"]
    return HMM


def load_xps(save_as):
    """Load `xps` (as a `list`) from given dir."""
    save_as = Path(save_as).expanduser()
    files = [d/"xp" for d in uplink.list_job_dirs(save_as)]
    if not files:
        raise FileNotFoundError(f"No results found at {save_as}.")

    def load_any(filepath):
        """Load any/all `xp's` from `filepath`."""
        with open(filepath, "rb") as F:
            # If experiment crashed, then xp will be empty
            try:
                data = dill.load(F)
            except EOFError:
                return []
            # Always return list
            try:
                return data["xps"]
            except KeyError:
                return [data["xp"]]

    print("Loading %d files from %s" % (len(files), save_as))
    xps = []  # NB: progbar wont clean up properly w/ list compr.
    for f in tqdm(files, desc="Loading"):
        xps.extend(load_any(f))

    if len(xps) == 0:
        raise RuntimeError("No completed experiments found.")
    elif len(xps) < len(files):
        print(len(files)-len(xps), "files could not be loaded,",
              "presumably because their respective jobs crashed.")

    return xps


def save_xps(xps, save_as, nDir=100):
    """Save `xps` (list of `xp`s) in `nDir` subfolders: `save_as/i`.

    Example
    -------
    Rename attr. `n_iter` to `nIter` in some saved data:

    ```py
    proj_name = "Stein"
    dd = rc.dirs.data / proj_name
    save_as = dd / "run_2020-09-22__19:36:13"

    for save_as in dd.iterdir():
        save_as = dd / save_as

        xps = load_xps(save_as)
        HMM = load_HMM(save_as)

        for xp in xps:
            if hasattr(xp,"n_iter"):
                xp.nIter = xp.n_iter
                del xp.n_iter

        overwrite_xps(xps, save_as)
    ```
    """
    save_as = Path(save_as).expanduser()
    save_as.mkdir(parents=False, exist_ok=False)

    n = int(len(xps) // nDir) + 1
    splitting = [xps[i:i + n] for i in range(0, len(xps), n)]
    for i, sub_xps in enumerate(tqdm(splitting, desc="Saving")):
        if len(sub_xps):
            iDir = save_as / str(i)
            os.mkdir(iDir)
            with open(iDir/"xp", "wb") as F:
                dill.dump({'xps': sub_xps}, F)


def overwrite_xps(xps, save_as, nDir=100):
    """Save xps in save_as, but safely (by first saving to tmp)."""
    save_xps(xps, save_as/"tmp", nDir)

    # Delete
    for d in tqdm(uplink.list_job_dirs(save_as),
                  desc="Deleting old"):
        shutil.rmtree(d)

    # Mv up from tmp/ -- goes quick, coz there are not many.
    for d in os.listdir(save_as/"tmp"):
        shutil.move(save_as/"tmp"/d, save_as/d)

    shutil.rmtree(save_as/"tmp")


def reduce_inodes(save_as, nDir=100):
    """Reduce the number of `xp` dirs.

    Done by packing multiple `xp`s into lists (`xps`).
    This reduces the **number** of files (inodes) on the system, which is limited.

    It also deletes files "xp.var" and "out",
    whose main content tends to be the printed progbar.
    This probably leads to some reduced loading time.

    FAQ: Why isn't the default for `nDir` simply 1?
    So that we can get a progressbar when loading.
    """
    overwrite_xps(load_xps(save_as), save_as, nDir)

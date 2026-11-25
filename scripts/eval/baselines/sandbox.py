"""Import an unmodified upstream repository next to CSG-Fusion.

The baselines ship top-level packages named ``src`` (and Splatt3R also
``main``, ``utils``, ``data``, ``workspace``) that clash with ours. A ``ModuleSandbox`` keeps two private
copies of ``sys.modules`` / ``sys.path`` / cwd: CSG-Fusion's and the upstream repo's. Code that touches the
baseline (building it, running its forward pass) runs inside ``with sandbox:``; everything else sees ours.
"""
import os
import sys
from pathlib import Path

# Top-level names that the baselines and CSG-Fusion (or MASt3R/DUSt3R/CroCo inside them) may both define.
SHARED_TOP_LEVEL = {'src', 'main', 'utils', 'data', 'workspace', 'config', 'dust3r', 'mast3r', 'croco', 'models',
                    'curope', 'datasets', 'demo', 'train', 'visloc'}


def _top_level_names(root: Path):
    names = set()
    for p in root.iterdir():
        if p.suffix == '.py':
            names.add(p.stem)
        elif p.is_dir() and not p.name.startswith('.'):
            names.add(p.name)
    return names


def _inside(path: Path, parent: Path):
    return path == parent or parent in path.parents


class ModuleSandbox:

    def __init__(self, root, hide=()):
        """root: the upstream repo. hide: directories (our repo) whose sys.path entries the baseline must not
        see, so that e.g. its ``import mast3r`` cannot resolve to our copy of MASt3R."""
        self.root = Path(root).resolve()
        if not self.root.is_dir() or not any(self.root.iterdir()):
            raise FileNotFoundError(f'{self.root} is empty: run `git submodule update --init --recursive`')
        self.names = (SHARED_TOP_LEVEL | _top_level_names(self.root)) - {'scripts'}   # never hide this package
        self._modules = {}              # the baseline's modules while we are outside the sandbox
        hide = [Path(h).resolve() for h in hide]
        visible = lambda p: p not in ('', '.') and not any(_inside(Path(p).resolve(), h) for h in hide)
        self._path = [str(self.root)] + [p for p in sys.path if visible(p)]
        self._outer = []

    def _owned(self, name):
        return name.split('.')[0] in self.names

    def __enter__(self):
        outer_modules = {k: sys.modules.pop(k) for k in list(sys.modules) if self._owned(k)}
        self._outer.append((outer_modules, list(sys.path), os.getcwd()))
        sys.modules.update(self._modules)
        sys.path[:] = self._path
        os.chdir(self.root)
        return self

    def __exit__(self, *exc):
        outer_modules, outer_path, outer_cwd = self._outer.pop()
        self._modules = {k: sys.modules.pop(k) for k in list(sys.modules) if self._owned(k)}
        self._path = list(sys.path)     # keep sys.path entries the baseline appended (e.g. Splatt3R's src/mast3r_src)
        sys.modules.update(outer_modules)
        sys.path[:] = outer_path
        os.chdir(outer_cwd)
        return False

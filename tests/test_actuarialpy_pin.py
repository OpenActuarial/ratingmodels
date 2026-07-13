"""Guard the declared actuarialpy range, mirroring projectionmodels' guard.

Derived from the pin, not hardcoded, so bumping the range can't strand this
guard on a stale expectation.
"""

import importlib.metadata
import importlib.util
import re
from pathlib import Path

import actuarialpy as ap


def test_tests_use_the_installed_actuarialpy_package():
    requirement = next(
        r for r in importlib.metadata.requires("ratingmodels")
        if r.startswith("actuarialpy")
    )
    floor = re.search(r">=\s*([0-9][0-9.]*)", requirement).group(1)
    installed = tuple(int(p) for p in ap.__version__.split(".")[:3])
    minimum = tuple(int(p) for p in floor.split(".")[:3])
    assert installed >= minimum, (
        f"installed actuarialpy {ap.__version__} is below the declared floor {floor}"
    )
    # Pre-1.0 minors may break APIs, so the declared range must also carry an
    # upper bound at the next minor -- and the installed version must sit
    # inside it.
    cap_match = re.search(r"<\s*([0-9][0-9.]*)", requirement)
    assert cap_match is not None, (
        f"pin {requirement!r} has no upper bound; pre-1.0 requires a cap"
    )
    cap = tuple(int(p) for p in cap_match.group(1).split("."))
    assert cap == (minimum[0], minimum[1] + 1), (
        f"cap {cap_match.group(1)} is not the next minor above the floor {floor}"
    )
    assert installed < cap + (0,) * (3 - len(cap)), (
        f"installed actuarialpy {ap.__version__} exceeds the declared cap"
    )
    # The imported module must be the one the import system resolves (an
    # injected fake in sys.modules would fail this) and must not resolve from
    # THIS repo's own source tree -- i.e. a vendored or stray copy under src/.
    # A venv created inside the checkout dir is fine (the min-deps CI job does
    # exactly that); an installed package lives in site-packages, not src/.
    spec = importlib.util.find_spec("actuarialpy")
    assert spec is not None and spec.origin is not None
    ap_path = Path(ap.__file__).resolve()
    assert ap_path == Path(spec.origin).resolve()
    repo_src = (Path(__file__).resolve().parent.parent / "src").resolve()
    assert not ap_path.is_relative_to(repo_src), (
        f"actuarialpy resolved from this repository's source tree: {ap_path}"
    )

"""A collection of heavier tests to run locally.

Useful for debugging or testing new features.

There are three degrees of testing:
- lightweight: BALK
- medium: BENLDE
- heavy: euro34

!!!!!!!!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!
- You normally want to run these tests one case at a time. E.g.:

    `pytest tests/local_test.py::test_name[BALK]`

- Do not run this on Github's CI!
"""

import subprocess
from pathlib import Path

import pytest

TEST_PNG = ["pipelines", "salt_cavern_h2_potential"]


def build_request_all(shape: str):
    """Construct a request for the given categories."""
    return " ".join([f"results/{shape}/{file}.png" for file in TEST_PNG])


@pytest.mark.parametrize("shape", ["BALK", "BENLDE", "euro34"])
def test_full_run(
    user_path: Path, shape: str
):
    """Test a full request of module outputs (using images as proxy)."""
    request = build_request_all(shape)

    assert subprocess.run(
        f"snakemake --use-conda --cores 4 --forceall {request}",
        shell=True,
        check=True,
        cwd=user_path.parent.parent,
    )
    assert subprocess.run(
        f"snakemake --use-conda --cores 4 {request} --report results/{shape}/report.html",
        shell=True,
        check=True,
        cwd=user_path.parent.parent,
    )
    assert subprocess.run(
        f"snakemake --use-conda --cores 4 {request} --rulegraph | dot -Tpng > results/{shape}/rulegraph.png",
        shell=True,
        check=True,
        cwd=user_path.parent.parent,
    )

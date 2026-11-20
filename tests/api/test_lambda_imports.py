import subprocess
import sys

LAMBDA_MODULES = ["detector.api", "detector.finalize", "detector.unzip", "detector.render"]


def test_lambda_modules_do_not_import_torch():
    """The Lambda image has no torch: these modules (and what they import) must not need it."""
    code = (
        "import sys, importlib\n"
        f"for m in {LAMBDA_MODULES!r}: importlib.import_module(m)\n"
        "heavy = sorted(m for m in ('torch', 'torchvision', 'ultralytics', 'onnxruntime')"
        " if m in sys.modules)\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={"BUCKET": "b", "TABLE": "t", "PATH": ""},
    )  # fmt: skip
    assert out.stdout.strip() == ""

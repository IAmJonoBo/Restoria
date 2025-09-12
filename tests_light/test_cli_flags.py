import subprocess
import sys


def run_ok(args):
    subprocess.run([sys.executable, "inference_gfpgan.py", "--dry-run", *args], check=True)


def test_dry_run_device_no_download():
    run_ok(["--device", "cpu", "--no-download", "-v", "1.4"])


def test_cli_bg_opts_and_verbose():
    # Accept new background options and verbose flag
    run_ok(["--device", "cpu", "--no-download", "-v", "1.3", "--bg_upsampler", "none", "--verbose"])
    run_ok(["--device", "cpu", "--no-download", "-v", "1.3", "--bg_precision", "fp32"])


def test_cli_detector_and_no_parse_and_manifest(tmp_path):
    out_manifest = tmp_path / "man.json"
    run_ok(
        [
            "--device",
            "cpu",
            "--no-download",
            "-v",
            "1.4",
            "--detector",
            "retinaface_resnet50",
            "--no-parse",
            "--manifest",
            str(out_manifest),
        ]
    )


def test_cli_sweep_skip_max_and_printenv(tmp_path):
    out_manifest = tmp_path / "man.json"
    run_ok(
        [
            "--device",
            "cpu",
            "--no-download",
            "-v",
            "1.4",
            "--sweep-weight",
            "0.3,0.5",
            "--skip-existing",
            "--max-images",
            "1",
            "--print-env",
            "--manifest",
            str(out_manifest),
        ]
    )


def test_cli_workers_dry_run():
    # Ensure the parser accepts workers flag; no execution on --dry-run
    run_ok(["--device", "cpu", "--no-download", "-v", "1.4", "--workers", "2"])

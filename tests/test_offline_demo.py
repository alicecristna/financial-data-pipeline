"""The demo must run offline on bundled CSVs."""
import socket
import subprocess
import sys
from pathlib import Path

import demo

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_offline_and_succeeds():
    result = subprocess.run(
        [sys.executable, "demo.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "DEMO OK" in result.stdout
    assert "no network calls" in result.stdout


def test_demo_makes_no_socket_calls(monkeypatch, capsys):
    """Guard against any accidental network call inside demo.main()."""

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    assert demo.main() == 0
    capsys.readouterr()

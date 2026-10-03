#!/usr/bin/env python3
"""
create_venv.py — AegisRAG virtual environment setup script.

Usage:
    python create_venv.py            # creates .venv/ and installs all deps
    python create_venv.py --check    # validates the active venv has all packages

Run from the project root (same directory as requirements.txt).
"""

import subprocess
import sys
import os
import argparse
import venv

VENV_DIR = ".venv"
REQUIREMENTS = "requirements.txt"


def run(cmd: list, **kw):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def get_python(venv_dir: str) -> str:
    """Returns the Python executable path inside the venv."""
    if sys.platform.startswith("win"):
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def create_venv(venv_dir: str) -> None:
    print(f"\n[1/4] Creating virtual environment at '{venv_dir}/'...")
    builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=True)
    builder.create(venv_dir)
    print("      Done.")


def upgrade_pip(python: str) -> None:
    print("\n[2/4] Upgrading pip, setuptools, wheel...")
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"])
    print("      Done.")


def install_deps(python: str, requirements: str) -> None:
    print(f"\n[3/4] Installing dependencies from '{requirements}'...")
    run([python, "-m", "pip", "install", "-r", requirements])
    print("      Done.")


def validate(python: str) -> None:
    print("\n[4/4] Validating key imports...")
    checks = [
        ("fastapi", "FastAPI"),
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("langchain_groq", "langchain_groq"),
        ("qdrant_client", "qdrant_client"),
        ("logfire", "logfire"),
        ("ragas", "ragas 0.2.x"),
        ("docx", "python-docx"),
        ("pptx", "python-pptx"),
        ("flashrank", "flashrank"),
        ("sentence_transformers", "sentence-transformers"),
    ]
    errors = []
    for mod, label in checks:
        result = subprocess.run(
            [python, "-c", f"import {mod}; print('OK: {label}')"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"      {result.stdout.strip()}")
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"
            print(f"      FAIL: {label} — {err}")
            errors.append(label)

    if errors:
        print(f"\n  WARN: {len(errors)} package(s) failed validation: {errors}")
        print("  Run: python create_venv.py  (to reinstall)")
    else:
        print("\n  All imports OK!")


def main():
    parser = argparse.ArgumentParser(description="AegisRAG venv setup")
    parser.add_argument("--check", action="store_true", help="Only validate, do not create")
    parser.add_argument("--venv-dir", default=VENV_DIR, help=f"Venv directory (default: {VENV_DIR})")
    args = parser.parse_args()

    venv_dir = args.venv_dir
    python = get_python(venv_dir)

    if args.check:
        if not os.path.isfile(python):
            print(f"ERROR: No venv found at '{venv_dir}/'. Run: python create_venv.py")
            sys.exit(1)
        validate(python)
        return

    create_venv(venv_dir)
    upgrade_pip(python)
    install_deps(python, REQUIREMENTS)
    validate(python)

    print(f"""
========================================================
  AegisRAG venv ready!

  Activate with:
    Windows:  {venv_dir}\\Scripts\\activate
    Linux/Mac: source {venv_dir}/bin/activate

  Then run the API:
    uvicorn app.main:app --reload

  Or run the eval suite:
    python -m evals.cli --mode all
========================================================
""")


if __name__ == "__main__":
    main()

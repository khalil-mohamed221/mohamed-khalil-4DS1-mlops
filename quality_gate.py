import subprocess
import sys
import re


def run_command(cmd: str, description: str) -> None:
    """
    Run a shell command and FAIL immediately if it returns non-zero.

    Used for HARD quality checks (formatting, security, style).
    """
    print(f"\n🔎 Checking: {description}")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"❌ FAILED: {description}")
        sys.exit(1)

    print(f"✔ PASSED: {description}")


def check_pylint(min_score: float) -> None:
    """
    Run pylint and enforce a MINIMUM SCORE instead of zero warnings.

    This is the correct approach for ML projects where:
    - Many variables
    - Long functions
    - Scientific naming (X_train, y_test, etc.)
    """

    print("\n🔎 Checking: Code Quality (Pylint – score based)")

    result = subprocess.run(
        "pylint *.py --score=y",
        shell=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    # Extract pylint score from output
    match = re.search(r"rated at ([0-9.]+)/10", result.stdout)

    if not match:
        print("❌ FAILED: Could not extract Pylint score")
        sys.exit(1)

    score = float(match.group(1))

    if score < min_score:
        print(f"❌ FAILED: Pylint score {score} < {min_score}")
        sys.exit(1)

    print(f"✔ PASSED: Pylint score {score} ≥ {min_score}")


def main():
    """
    QUALITY GATE LEVELS
    ===================

    You can change MIN_PYLINT_SCORE depending on context:

    - Local development:      7.0 – 8.0
    - CI (GitHub Actions):    8.5   ✅ (recommended)
    - Production / Library:   9.0 – 9.5
    """

    MIN_PYLINT_SCORE = 8.5  # ⬅️ CHANGE THIS TO TEST DIFFERENT STRICTNESS LEVELS

    # -------------------------------
    # HARD FAIL CHECKS (no tolerance)
    # -------------------------------
    run_command("black *.py --check", "Code Format (Black)")
    run_command("flake8 *.py", "Style Check (Flake8)")
    run_command("bandit -r .", "Security Scan (Bandit)")

    # -------------------------------
    # SOFT FAIL CHECK (score-based)
    # -------------------------------
    c

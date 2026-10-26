"""Starts the results app in the browser.

From the repo folder, with the conda environment master active:

    python src/app/run.py
    python -m app.run

Extra arguments go to Streamlit, for example another port:

    python -m app.run --server.port 8502

Stop the app with Ctrl+C in the terminal. See docs/app.md.
"""
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "build" / "app.py"


def main(argv=None):
    """Runs `streamlit run app.py` with the Python that runs this file.

    Returns its exit code.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP), *args])
    except KeyboardInterrupt:                    # Ctrl+C also stops Streamlit
        return 0


if __name__ == "__main__":
    sys.exit(main())

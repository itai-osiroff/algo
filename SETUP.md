# Setup

- Python 3.10 on Windows is what the camera code was developed against.
- `pip install -r requirements.txt` installs the Python dependencies.
- `pyspin` / `simple_pyspin` need the Spinnaker SDK and its matching PySpin wheel.
  These are proprietary and are NOT included in this repository. Download them
  from Teledyne FLIR (Spinnaker SDK, Python bindings for your Python version).
- Camera acquisition scripts (`test_acquire_images_PG.py`, `PG_class.py`) require
  the physical camera. The image-processing code in `moment_images/` runs without it.
- Some scripts contain placeholder paths (for example in `PG_class.py`); set them to
  your local files before running.

"""Repair a *copy* of the old MSS WAMIT tanker MAT file's drift heading mirror.

The checked-out Oct 2026 MSS wamit2vessel.m mistakenly copies drift data
from headings 10:10:170 into 190:10:350 without reversing the heading order.
For a port-starboard symmetric hull, 190 must come from 170, not 10.

Original tanker.mat is deliberately never modified by this script.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'data_examples' / 'tanker.mat'
CORRECTED = ROOT / 'data_examples' / 'tanker_drift_corrected.mat'


def main():
    digest = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    mat = loadmat(ORIGINAL, simplify_cells=True)
    vessel = mat['vessel']
    headings = np.rad2deg(np.asarray(vessel['headings']).ravel())
    np.testing.assert_allclose(headings, np.arange(0, 360, 10), atol=1e-8)
    drift_cells = vessel['driftfrc']['amp']
    for k, parity in enumerate((1, -1, -1)):  # surge, sway, yaw
        values = drift_cells[k]
        if values.shape[1] != 36:
            raise ValueError('Expected 36 heading columns')
        # Column 20:36 (MATLAB 1-based) mirrors columns 18:2 reversed.
        values[:, 19:36, 0] = parity * values[:, 17:0:-1, 0]
        # The original `.mat` has nine padded zero-speed cases; leave those alone.
        np.testing.assert_allclose(values[:, 19:36, 0],
                                   parity * values[:, 17:0:-1, 0], rtol=0, atol=0)
    if CORRECTED.exists():
        raise FileExistsError(f'Refusing to overwrite {CORRECTED}')
    savemat(CORRECTED, {'vessel': vessel}, format='5',
            long_field_names=True, do_compression=True)
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Source file unexpectedly changed')
    print('Original preserved:', ORIGINAL)
    print('Drift-mirrored copy:', CORRECTED)
    print('Source sha256:', digest)


if __name__ == '__main__':
    main()

# Gunnerus WAMIT teacher-requested hydrodynamic plots

This folder now contains **exactly five PDFs**, one for each requested
category:

1. 01_Motion_RAOs.pdf - all six motion RAOs, amplitude and phase, 19 headings.
2. 02_Added_Mass.pdf - all 36 added mass A coefficients, detailed 6x6 plots.
3. 03_Potential_Damping.pdf - all 36 potential damping B coefficients.
4. 04_First_Order_Wave_Load_Coefficients.pdf - all six first-order exciting loads.
5. 05_Second_Order_Wave_Load_Coefficients.pdf - mean wave-drift loads only.

All horizontal axes show **wave period T in seconds**. The results use the
100 calculated frequencies in data_examples/GunnerusL35_20201112.mat, and all original
19 headings 0, 10, ..., 180 degrees. No interpolation has been performed
in these plots; synthetic matrix-only zero/infinite limiting entries are not
plotted as original WAMIT calculations.

To regenerate the five PDFs from the root of the repository:

    python3 scripts/plot_wamit_teacher_five.py

The CSV and figures from the previous 50-vs-100-vs-150 interpolation
experiment were removed from this folder because they are not requested
in this teacher package. The separate interpolation plotting code is
still in scripts/plot_wamit_interpolation.py if needed in future.

The JSON file records exact plot source and data checks. PDFs are ignored
by Git so the repo stays lightweight.

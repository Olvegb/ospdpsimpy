"""Compare 50/150 interpolated MSS/WAMIT values to original 100 points.

From the ospdpsimpy repository root:
    python3 scripts/plot_wamit_interpolation.py

Uses production hull.helpfunctions.Wamit.load_mat_vessel; does not run any FMUs.
License: MIT (see the repository root LICENSE).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ospfmusim.hull.helpfunctions.Wamit import load_mat_vessel
from ospfmusim.hull.helpfunctions.WaveLoads import selected_drift_method

# -------------- Visual-test settings --------------
SOURCE_MAT = ROOT / "data_examples" / "gunnerus.mat"
COUNTS = (50, 150)
HEADINGS = tuple(range(0, 181, 10))
OUTPUT = ROOT / "src" / "ospfmusim" / "hull" / "plots" / "wamit_interpolation"
# --------------------------------------------------
DOFS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
LOADS = ("Surge", "Sway", "Yaw")
COLORS = {"original": "#242424", 50: "#1479a8", 150: "#dd731f"}


def load(count):
    return load_mat_vessel(
        SOURCE_MAT, frequency_count=count, heading_step_deg=10,
        frequency_spacing="log", use_mat_viscous_damping=True,
        drift_method=selected_drift_method(), require_simulation_ready=True
    )


def sorted_period(data):
    t = 2 * np.pi / np.asarray(data["frequency_grid"])
    idx = np.argsort(t)
    return t[idx], idx


def make_axes(ax, x_label=False):
    ax.set_xlim(2, 30)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    if x_label:
        ax.set_xlabel("Wave period T [s]")
    ax.grid(alpha=.2)
    ax.tick_params(labelsize=8)


def values_at(data, table, j, beta):
    h = np.flatnonzero(np.isclose(data["headings_deg"], beta, atol=1e-6))
    if len(h) != 1:
        raise ValueError(f"Heading {beta}° not in {table}")
    t, order = sorted_period(data)
    z = np.asarray(data[table].values[j, :, h[0]])[order]
    return t, z


def amp_and_phase(a, p, datasets, table, j, beta, scale, unit):
    for count, data in datasets.items():
        t, z = values_at(data, table, j, beta)
        c = COLORS[count]
        is_source = count == "original"
        kw = dict(color=c, alpha=.8)
        if is_source:
            a.plot(t, np.abs(z)*scale, linestyle="None", marker="o",
                   markersize=2.3, label="Original 100 WAMIT", **kw)
        else:
            a.plot(t, np.abs(z)*scale, lw=1.5, label=f"Interpolated {count}", **kw)
        if np.max(np.abs(z)) > 1e-14:
            # Wamit.py encodes positive phase lag as exp(-i phase).
            phase = -np.angle(z)
            phase[np.abs(z) < 1e-7*np.max(np.abs(z))] = np.nan
            if is_source:
                p.plot(t, phase, linestyle="None", marker="o",
                       markersize=2.1, **kw)
            else:
                p.plot(t, phase, lw=1.3, **kw)
    a.set_ylabel(unit)
    p.set_ylabel("Phase lag [rad]")
    make_axes(a)
    make_axes(p)


def motion_page(datasets, beta):
    fig, ax = plt.subplots(6, 2, figsize=(16.5, 12))
    fig.suptitle(f"Motion RAOs | {beta:03d}° | 100 original vs 50/150 interpolated",
                 fontsize=16, weight="bold", y=.99)
    for j, name in enumerate(DOFS):
        scale = 1 if j < 3 else 180/np.pi
        unit = "m/m" if j < 3 else "deg/m"
        amp_and_phase(ax[j,0], ax[j,1], datasets, "motion_table", j, beta,
                      scale, unit)
        ax[j,0].set_title(f"{j+1}. {name}: amplitude", fontsize=10)
        ax[j,1].set_title(f"{j+1}. {name}: phase", fontsize=10)
        if j == 5:
            make_axes(ax[j,0], True)
            make_axes(ax[j,1], True)
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(.5,.003), fontsize=9)
    fig.subplots_adjust(top=.94, bottom=.08, left=.07, right=.975,
                        hspace=.75, wspace=.2)
    return fig


def load_page(datasets, beta):
    fig = plt.figure(figsize=(16.5, 12))
    gs = fig.add_gridspec(6, 2, top=.94, bottom=.075,
                          left=.07, right=.975, hspace=.7, wspace=.19)
    fig.suptitle(f"Wave loads | {beta:03d}° | 100 original vs 50/150 interpolated",
                 fontsize=16, weight="bold")
    first = None
    for j, name in enumerate(LOADS):
        a, p = fig.add_subplot(gs[j, 0]), fig.add_subplot(gs[j, 1])
        amp_and_phase(a,p,datasets,"force_table",j,beta,1/1000,
                      "kN/m" if j<2 else "kNm/m")
        a.set_title(f"First order: {name} amplitude", fontsize=10)
        p.set_title(f"First order: {name} phase", fontsize=10)
        if first is None:
            first = a
    for j, name in enumerate(LOADS):
        ax = fig.add_subplot(gs[j+3, :])
        for count, data in datasets.items():
            t,z = values_at(data,"drift_table",j,beta)
            if count == "original":
                ax.plot(t,z,linestyle="None",marker="o",markersize=2.3,
                        color=COLORS[count],alpha=.8)
            else:
                ax.plot(t,z,lw=1.4,color=COLORS[count],alpha=.85)
        ax.set_ylabel("N/m²" if j<2 else "Nm/m²")
        ax.set_title(f"Second order: mean {name.lower()} drift (not full QTF)",
                     fontsize=10)
        make_axes(ax, x_label=(j==2))
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(.5,.003), fontsize=9)
    return fig


def matrix_figure(datasets, field, title):
    fig, axes = plt.subplots(3,2,figsize=(12,10))
    fig.suptitle(f"Gunnerus {title} — six diagonal coefficients\n"
                 "100 calculated WAMIT points vs 50/150 resampled",
                 fontsize=14,weight="bold")
    for j,ax in enumerate(axes.flat):
        for count,data in datasets.items():
            t,ordr=sorted_period(data)
            vals=data[field][j,j,ordr]
            if count == "original":
                ax.plot(t,vals,linestyle="None",marker="o",markersize=2.5,
                        color=COLORS[count],alpha=.75,label="Original 100")
            else:
                ax.plot(t,vals,lw=1.4,color=COLORS[count],
                        label=f"Interpolated {count}")
        ax.set_title(f"{DOFS[j]} [{j+1},{j+1}]",fontsize=10)
        ax.set_ylabel("Coefficient [SI]")
        make_axes(ax,True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,
               bbox_to_anchor=(.5,.005),fontsize=9)
    fig.tight_layout(rect=[0,.045,1,.935])
    return fig


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source=load(None)
    original_n=len(source["frequency_grid"])
    if original_n != 100:
        raise ValueError(f"Expected 100 calculated points; found {original_n}")
    if "no frequency resampling" not in source["frequency_info"]["grid_mode"]:
        raise AssertionError("Original source grid was not preserved")
    datasets={"original":source, **{count:load(count) for count in COUNTS}}
    summary={"source_file":str(SOURCE_MAT),"calculated_source_count":original_n,
             "grids":{},"headings_deg":list(HEADINGS),
             "warning":"150 interpolated points contain no additional WAMIT calculations."}
    with (OUTPUT/"frequency_grids.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["count","point_index","omega_rad_s","period_seconds","mode"])
        for key,data in datasets.items():
            w=np.asarray(data["frequency_grid"])
            info=data["frequency_info"]
            assert len(w)==(original_n if key=="original" else key)
            label=str(original_n) if key=="original" else str(key)
            summary["grids"][label]={
                "count":len(w),
                "min_omega_rad_s":float(w[0]),
                "max_omega_rad_s":float(w[-1]),
                "min_period_s":float(2*np.pi/w[-1]),
                "max_period_s":float(2*np.pi/w[0]),
                "mode":info["grid_mode"]}
            print(f"{label:>3} frequencies: {info['grid_mode']}, "
                  f"omega {w[0]:.9f}–{w[-1]:.9f} rad/s, "
                  f"T {2*np.pi/w[-1]:.2f}–{2*np.pi/w[0]:.2f} s",flush=True)
            for i,omega in enumerate(w):
                writer.writerow([label,i,f"{omega:.15g}",
                                 f"{2*np.pi/omega:.15g}",info["grid_mode"]])
    base="100_original_vs_50_and_150"
    with PdfPages(OUTPUT/f"00_Motion_RAOs_{base}.pdf") as fm, \
         PdfPages(OUTPUT/f"01_Wave_loads_{base}.pdf") as fl:
        for beta in HEADINGS:
            fig=motion_page(datasets,beta)
            fm.savefig(fig)
            if beta in (0,90,180):
                fig.savefig(OUTPUT/f"Preview_motion_{beta:03d}deg.png",dpi=150)
            plt.close(fig)
            fig=load_page(datasets,beta)
            fl.savefig(fig)
            plt.close(fig)
            print(f"Completed heading {beta:03d}°",flush=True)
    for name,field,title in [
        ("02_Added_mass_A","added_mass_by_frequency","added mass A"),
        ("03_Radiation_damping_B","radiation_damping_by_frequency",
         "potential radiation damping B")]:
        fig=matrix_figure(datasets,field,title)
        fig.savefig(OUTPUT/f"{name}_{base}.png",dpi=155)
        plt.close(fig)
    (OUTPUT/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"Complete: {OUTPUT}")
    print("No FMU, LF/WF simulation, or Git push performed.")


if __name__=="__main__":
    main()

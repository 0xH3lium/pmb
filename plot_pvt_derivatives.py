"""Standalone script to plot PVT properties and their derivatives."""

import numpy as np
import pytensor.tensor as pt
from pytensor import function
import matplotlib.pyplot as plt

from src.mcmc_pmb.pvt import MaterialBalancePVT


def main():
    pvt_params = MaterialBalancePVT(bubble_point=3200.0)
    engine = pvt_params.build_engine()

    pressure = np.linspace(1000.0, 4000.0, 500)

    bo_np, dbo_np = engine.oil_fvf_numpy(pressure)
    bg_np, dbg_np = engine.gas_fvf_numpy(pressure)
    rs_np, drs_np = engine.solution_gor_numpy(pressure)

    p_tensor = pt.vector("p")
    bo_pt, dbo_pt, bg_pt, dbg_pt, rs_pt, drs_pt = engine.evaluate_all(p_tensor)
    eval_fn = function([p_tensor], [bo_pt, dbo_pt, bg_pt, dbg_pt, rs_pt, drs_pt])
    bo_eval, dbo_eval, bg_eval, dbg_eval, rs_eval, drs_eval = eval_fn(pressure)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)
    fig.suptitle(
        "PVT Properties and Derivatives from Differentiable Splines", fontsize=14
    )

    axes[0, 0].plot(pressure, bo_np, "b-", linewidth=1.5)
    axes[0, 0].axvline(x=3200.0, color="k", linestyle="--", linewidth=1.0)
    axes[0, 0].set_xlabel("Pressure (psi)")
    axes[0, 0].set_ylabel(r"$B_o$ (rb/stb)")
    axes[0, 0].set_title("Oil Formation Volume Factor")

    axes[0, 1].plot(pressure, bg_np, "g-", linewidth=1.5)
    axes[0, 1].axvline(x=3200.0, color="k", linestyle="--", linewidth=1.0)
    axes[0, 1].set_xlabel("Pressure (psi)")
    axes[0, 1].set_ylabel(r"$B_g$ (rb/scf)")
    axes[0, 1].set_title("Gas Formation Volume Factor")

    axes[0, 2].plot(pressure, rs_np, "r-", linewidth=1.5)
    axes[0, 2].axvline(x=3200.0, color="k", linestyle="--", linewidth=1.0)
    axes[0, 2].set_xlabel("Pressure (psi)")
    axes[0, 2].set_ylabel(r"$R_s$ (scf/stb)")
    axes[0, 2].set_title("Solution Gas-Oil Ratio")

    axes[1, 0].plot(pressure, dbo_eval, "b-", linewidth=1.5)
    axes[1, 0].axvline(x=3200.0, color="k", linestyle="--", linewidth=1.0)
    axes[1, 0].set_xlabel("Pressure (psi)")
    axes[1, 0].set_ylabel(r"$dB_o/dp$ (rb/stb/psi)")
    axes[1, 0].set_title("Derivative of $B_o$")

    axes[1, 1].plot(pressure, dbg_eval, "g-", linewidth=1.5)
    axes[1, 1].axvline(x=3200.0, color="k", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xlabel("Pressure (psi)")
    axes[1, 1].set_ylabel(r"$dB_g/dp$ (rb/scf/psi)")
    axes[1, 1].set_title("Derivative of $B_g$")

    axes[1, 2].plot(pressure, drs_eval, "r-", linewidth=1.5)
    axes[1, 2].axvline(x=3200.0, color="k", linestyle="--", linewidth=1.0)
    axes[1, 2].set_xlabel("Pressure (psi)")
    axes[1, 2].set_ylabel(r"$dR_s/dp$ (scf/stb/psi)")
    axes[1, 2].set_title("Derivative of $R_s$")

    for ax in axes.flat:
        ax.annotate(
            "C1 Continuity Preserved for HMC/NUTS",
            xy=(0.5, 0.02),
            xycoords="axes fraction",
            ha="center",
            fontsize=8,
            style="italic",
            alpha=0.7,
        )

    plt.tight_layout()
    plt.savefig("pvt_derivatives.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

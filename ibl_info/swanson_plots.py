# unabashedly copied from bwm
from iblatlas.plots import plot_swanson_vector, plot_scalar_on_slice
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions

ba = AllenAtlas()
br = BrainRegions()


def add_cbar(cmap, vmin, vmax, ax, label, cbar_kwargs=dict(), associated=True):
    """
    Add a colorbar to an axis
    :return:
    """
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if associated:  # whether the axis passed in is stand alone or associated to another plot
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            **cbar_kwargs,
        )
    else:
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")  # type: ignore
    ticks = np.round(np.linspace(vmin, vmax, num=3), 2)
    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_tick_params(pad=5, labelsize=8)
    cbar.outline.set_visible(False)  # type: ignore
    cbar.set_label(label, fontsize=8, ha="center", va="top")

    return cbar


def plot_vertical_swanson(
    acronyms,
    scores,
    mask=None,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    fontsize=7,
    annotate_kwargs=dict(),
    cbar=True,
    cbar_label=None,
    cbar_shrink=0.8,
    legend=True,
    mask_label=None,
):
    """
    Plot a vertical swanson figure with optional colorbar and legend
    :return:
    """

    ax = plot_swanson_vector(
        acronyms,
        scores,
        hemisphere=None,
        orientation="portrait",
        cmap=cmap,
        br=br,
        ax=ax,
        empty_color="silver",
        linewidth=0.1,
        mask=mask,
        mask_color="silver",
        vmin=vmin,
        vmax=vmax,
        fontsize=fontsize,
        **annotate_kwargs,
    )

    ax.set_axis_off()
    ax.axes.invert_xaxis()

    if cbar:
        cbar_kwargs = {"shrink": cbar_shrink, "aspect": 12, "pad": 0.025}
        cax = add_cbar(cmap, vmin, vmax, ax, cbar_label, cbar_kwargs=cbar_kwargs)

    if legend:
        leg = add_sig_legends(ax, mask_label)  # type: ignore
        leg.set_bbox_to_anchor((0.65, 0.11))

    return ax, cax if cbar else None


def plot_horizontal_swanson(
    acronyms,
    scores,
    mask=None,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    fontsize=7,
    annotate_kwargs=dict(),
    cbar=True,
    cbar_label=None,
    cbar_shrink=0.45,
    legend=True,
    mask_label=None,
):
    """
    Plot a horzontal swanson figure with optional colorbar and legend
    :return:
    """

    ax = plot_swanson_vector(
        acronyms,
        scores,
        hemisphere=None,
        orientation="landscape",
        cmap=cmap,
        br=br,
        ax=ax,
        empty_color="silver",
        linewidth=0.1,
        mask=mask,
        mask_color="silver",
        vmin=vmin,
        vmax=vmax,
        fontsize=fontsize,
        **annotate_kwargs,
    )

    ax.set_axis_off()

    if cbar:
        cbar_kwargs = {"shrink": cbar_shrink, "aspect": 12, "pad": 0.025}
        cax = add_cbar(cmap, vmin, vmax, ax, cbar_label, cbar_kwargs=cbar_kwargs)

    if legend:
        if cbar:
            bbox = cax.ax.get_position()
            cax.ax.set_position([bbox.x0 - 0.15, bbox.y0, bbox.width, bbox.height])  # type: ignore
        leg = add_sig_legends(ax, mask_label)  # type: ignore
        leg.set_bbox_to_anchor((1, -0.05))

    return ax, cax if cbar else None


def add_sig_legends(ax, mask_label="Not significant"):
    """
    Add legend elements to an axis to indicate regions in mask (silver) and regions not analyzed (white)
    :return:
    """

    legend_elements = [
        Rectangle(
            (0, 0), 5, 5, facecolor="silver", edgecolor="black", linewidth=0.5, label=mask_label
        ),
        Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=0.5, label="Not analyzed"
        ),
    ]

    legend = ax.legend(
        handles=legend_elements,
        frameon=False,
        fontsize=8,
        handlelength=0.7,
        handletextpad=0.5,
    )
    return legend

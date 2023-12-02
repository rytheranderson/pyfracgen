from pathlib import Path
from typing import Sequence

import matplotlib.animation as animation
import numpy as np
from matplotlib import colormaps, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, figaspect

from pyfracgen.common import Result

ANIMATION_DEFAULT_SAVE = Path("ani.gif")
DEFAULT_COLORMAP = colormaps["hot"]


def get_stacked_cmap(cmap: colors.Colormap, nstacks: int) -> colors.Colormap:

    cline = np.vstack([np.array(cmap(np.linspace(0, 1, 200))) for _ in range(nstacks)])
    mymap = colors.LinearSegmentedColormap.from_list("stacked", cline)
    return mymap


def image(
    res: Result,
    cmap: colors.Colormap = DEFAULT_COLORMAP,
    ticks: bool = False,
    gamma: float = 0.3,
    vert_exag: float = 0.0,
    ls: tuple[int, int] = (315, 10),
) -> tuple[Figure, Axes]:

    fig, ax0 = plt.subplots(figsize=figaspect(res.image_array), dpi=res.dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    norm = colors.PowerNorm(gamma)
    if vert_exag > 0.0:
        light = colors.LightSource(azdeg=ls[0], altdeg=ls[1])
        shade = light.shade(
            res.image_array, cmap=cmap, norm=norm, vert_exag=vert_exag, blend_mode="hsv"
        )
        ax0.imshow(shade, origin="lower")
    else:
        ax0.imshow(res.image_array, origin="lower", cmap=cmap, norm=norm)
    if not ticks:
        plt.axis("off")
    fs = plt.gcf()
    fs.set_size_inches(res.width_inches, res.height_inches)
    return fig, ax0


def nebula_image(
    results: tuple[Result, Result, Result],
    ticks: bool = False,
    gamma: float = 1.0,
) -> tuple[Figure, Axes]:

    blue, green, red = results
    fig, ax0 = plt.subplots(figsize=figaspect(blue.image_array), dpi=blue.dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    arrays = [red.image_array, green.image_array, blue.image_array]
    final = np.dstack([arr / np.amax(arr) for arr in arrays])
    ax0.imshow(final**gamma, origin="lower")
    if not ticks:
        plt.axis("off")
    fs = plt.gcf()
    fs.set_size_inches(blue.width_inches, blue.height_inches)
    return fig, ax0


def markus_lyapunov_image(
    res: Result,
    cmap_neg: colors.Colormap,
    cmap_pos: colors.Colormap,
    gammas: tuple[float, float] = (1.0, 1.0),
    ticks: bool = False,
) -> tuple[Figure, Axes]:

    fig, ax0 = plt.subplots(figsize=figaspect(res.image_array), dpi=res.dpi)
    ax0.imshow(
        np.ma.masked_where(  # type: ignore[no-untyped-call]
            res.image_array > 0.0, res.image_array
        ),
        cmap=cmap_neg,
        origin="lower",
        norm=colors.PowerNorm(gammas[0]),
    )
    ax0.imshow(
        np.ma.masked_where(  # type: ignore[no-untyped-call]
            res.image_array < 0.0, res.image_array
        ),
        cmap=cmap_pos,
        origin="lower",
        norm=colors.PowerNorm(gammas[1]),
    )
    fig.subplots_adjust(0, 0, 1, 1)
    if not ticks:
        plt.axis(ticks)
    fs = plt.gcf()
    fs.set_size_inches(res.width_inches, res.height_inches)
    return fig, ax0


def randomwalk_image(
    res: Result,
    cmap: colors.Colormap = DEFAULT_COLORMAP,
    ticks: bool = False,
    gamma: float = 0.3,
    alpha_scale: float = 1.0,
) -> tuple[Figure, Axes]:

    fig, ax0 = plt.subplots(figsize=figaspect(res.image_array[:, :, 0]), dpi=res.dpi)
    max_ind = float(res.image_array.shape[-1] + 1)
    for i in range(res.image_array.shape[-1]):
        im = res.image_array[..., i]
        im = np.ma.masked_where(im == 0, im)  # type: ignore[no-untyped-call]
        alpha = alpha_scale * (1 - (i + 1) / max_ind)
        norm = colors.PowerNorm(gamma)
        ax0.imshow(
            im, origin="lower", alpha=alpha, cmap=cmap, norm=norm, interpolation=None
        )
    fig.subplots_adjust(0, 0, 1, 1)
    if not ticks:
        plt.axis(ticks)
    fs = plt.gcf()
    fs.set_size_inches(res.width_inches, res.height_inches)
    return fig, ax0


def save_animation(
    series: Sequence[Result],
    cmap: colors.Colormap = DEFAULT_COLORMAP,
    fps: int = 15,
    bitrate: int = 1800,
    file: Path = ANIMATION_DEFAULT_SAVE,
    ticks: bool = True,
    gamma: float = 0.3,
    vert_exag: float = 0,
    ls: tuple[int, int] = (315, 10),
) -> None:

    fig = plt.figure()
    fig.subplots_adjust(0, 0, 1, 1)
    if not ticks:
        plt.axis(ticks)
    fs = plt.gcf()
    norm = colors.PowerNorm(gamma)
    light = colors.LightSource(azdeg=ls[0], altdeg=ls[1])
    first, *_ = series
    fs.set_size_inches(first.width_inches, first.height_inches)
    writer = animation.PillowWriter(
        fps=fps, metadata=dict(artist="Me"), bitrate=bitrate
    )

    ims = []
    for res in series:
        shade = light.shade(
            res.image_array, cmap=cmap, norm=norm, vert_exag=vert_exag, blend_mode="hsv"
        )
        im = plt.imshow(shade, origin="lower", norm=norm)
        ims.append([im])

    if file.suffix != ".gif":
        file = file.with_suffix(".gif")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(file, dpi=first.dpi, writer=writer)

from __future__ import annotations

import matplotlib.animation as animation
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from pyfracgen.result import Result


def stack_cmaps(cmap: colors.Colormap, n_stacks: int) -> colors.LinearSegmentedColormap:

    colorline = np.array(cmap(np.linspace(0, 1, 200)))
    for _ in range(n_stacks - 1):
        colorline = np.vstack((colorline, cmap(np.linspace(0, 1, 200))))
    mymap = colors.LinearSegmentedColormap.from_list("my_colormap", colorline)
    return mymap


def image(
    res: Result,
    cmap: colors.Colormap = plt.cm.hot,
    ticks: str = "off",
    gamma: float = 0.3,
    vert_exag: float = 0,
    ls: tuple[int, int] = (315, 10),
) -> tuple[Figure, plt.Axes]:

    arr = res.image_array
    width = res.width_inches
    height = res.height_inches
    dpi = res.dpi
    w, h = plt.figaspect(arr)
    fig, ax0 = plt.subplots(figsize=(w, h), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    plt.axis(ticks)
    norm = colors.PowerNorm(gamma)
    light = colors.LightSource(azdeg=ls[0], altdeg=ls[1])

    if vert_exag != 0.0:
        ls = light.shade(
            arr, cmap=cmap, norm=norm, vert_exag=vert_exag, blend_mode="hsv"
        )
        ax0.imshow(ls, origin="lower")
    else:
        ax0.imshow(arr, origin="lower", cmap=cmap, norm=norm)

    fs = plt.gcf()
    fs.set_size_inches(width, height)
    return fig, ax0


def nebula_image(
    res_blue: Result,
    res_green: Result,
    res_red: Result,
    ticks: str = "off",
    gamma: float = 1.0,
) -> tuple[Figure, plt.Axes]:

    arr_blue = res_blue.image_array
    width = res_blue.width_inches
    height = res_blue.height_inches
    dpi = res_blue.dpi
    arr_green = res_green.image_array
    arr_red = res_red.image_array
    final = np.dstack(
        (
            arr_red / np.amax(arr_red),
            arr_green / np.amax(arr_green),
            arr_blue / np.amax(arr_blue),
        )
    )
    w, h = plt.figaspect(arr_blue)
    fig, ax0 = plt.subplots(figsize=(w, h), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    plt.axis(ticks)
    fs = plt.gcf()
    fs.set_size_inches(width, height)
    ax0.imshow(final**gamma, origin="lower")
    return fig, ax0


def markus_lyapunov_image(
    res: Result,
    cmap_negative: colors.Colormap,
    cmap_positive: colors.Colormap,
    gammas: tuple[float, float] = (1.0, 1.0),
    ticks: str = "off",
) -> tuple[Figure, plt.Axes]:

    arr = res.image_array
    width = res.width_inches
    height = res.height_inches
    dpi = res.dpi
    fig, ax0 = plt.subplots(figsize=(width, height), dpi=dpi)
    ax0.imshow(
        np.ma.masked_where(arr > 0.0, arr),  # type: ignore[no-untyped-call]
        cmap=cmap_negative,
        origin="lower",
        norm=colors.PowerNorm(gammas[0]),
    )
    ax0.imshow(
        np.ma.masked_where(arr < 0.0, arr),  # type: ignore[no-untyped-call]
        cmap=cmap_positive,
        origin="lower",
        norm=colors.PowerNorm(gammas[1]),
    )
    fig.subplots_adjust(0, 0, 1, 1)
    plt.axis(ticks)
    fs = plt.gcf()
    fs.set_size_inches(width, height)
    return fig, ax0


def randomwalk_image(
    res: Result,
    cmap: colors.Colormap = plt.cm.hot,
    ticks: str = "off",
    gamma: float = 0.3,
    alpha_scale: float = 1.0,
) -> tuple[Figure, plt.Axes]:

    arr = res.image_array
    width = res.width_inches
    height = res.height_inches
    dpi = res.dpi
    w, h = plt.figaspect(arr[:, :, 0])
    fig, ax0 = plt.subplots(figsize=(w, h), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    plt.axis(ticks)
    max_ind = float(arr.shape[-1] + 1)

    for i in range(arr.shape[-1]):
        im = arr[..., i]
        im = np.ma.masked_where(im == 0, im)  # type: ignore[no-untyped-call]
        alpha = 1 - (i + 1) / max_ind
        alpha *= alpha_scale
        norm = colors.PowerNorm(gamma)
        ax0.imshow(
            im, origin="lower", alpha=alpha, cmap=cmap, norm=norm, interpolation=None
        )

    fs = plt.gcf()
    fs.set_size_inches(width, height)
    return fig, ax0


def save_animation(
    series: list[Result],
    fps: int = 15,
    bitrate: int = 1800,
    cmap: int = plt.cm.hot,
    filename: str = "ani",
    ticks: str = "off",
    gamma: float = 0.3,
    vert_exag: float = 0,
    ls: tuple[int, int] = (315, 10),
) -> None:

    width = series[0].width_inches
    height = series[0].height_inches
    dpi = series[0].dpi
    fig = plt.figure()
    fig.subplots_adjust(0, 0, 1, 1)
    fs = plt.gcf()
    fs.set_size_inches(width, height)
    plt.axis(ticks)

    writer = animation.PillowWriter(
        fps=fps, metadata=dict(artist="Me"), bitrate=bitrate
    )
    norm = colors.PowerNorm(gamma)
    light = colors.LightSource(azdeg=ls[0], altdeg=ls[1])

    ims = []
    for s in series:
        arr = s.image_array
        ls = light.shade(
            arr, cmap=cmap, norm=norm, vert_exag=vert_exag, blend_mode="hsv"
        )
        im = plt.imshow(ls, origin="lower", norm=norm)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(f"{filename}.gif", dpi=dpi, writer=writer)

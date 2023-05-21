from __future__ import annotations

import argparse
import itertools as itt
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import pyfracgen as pf


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-M",
        "--mandelbrot",
        action="store_true",
        help="If present, run the Mandelbrot example.",
    )
    parser.add_argument(
        "-J",
        "--julia",
        action="store_true",
        help="If present, run the Julia animation example.",
    )
    parser.add_argument(
        "-L",
        "--lyapunov",
        action="store_true",
        help="If present, run the Lyapunov example.",
    )
    parser.add_argument(
        "-R",
        "--randomwalk",
        action="store_true",
        help="If present, run the random walk example.",
    )
    parser.add_argument(
        "-B",
        "--buddhabrot",
        action="store_true",
        help="If present, run the Buddhabrot example.",
    )
    return parser


def mandelbrot_example() -> None:

    start_time = time.time()
    xbound = (
        0.3602404434376143632361252444495 - 0.00000000000003,
        0.3602404434376143632361252444495 + 0.00000000000025,
    )
    ybound = (
        -0.6413130610648031748603750151793 - 0.00000000000006,
        -0.6413130610648031748603750151793 + 0.00000000000013,
    )
    res = pf.mandelbrot(
        xbound, ybound, pf.funcs.power, width=4, height=3, dpi=300, maxiter=5000
    )
    stacked = pf.images.get_stacked_cmap(plt.cm.gist_gray, 50)
    pf.images.image(res, cmap=stacked, gamma=0.8)
    plt.savefig("example_images/mandelbrot_ex.png")
    print(f"calculation took {round((time.time() - start_time), 2)} seconds")


def julia_animation_example() -> None:

    start_time = time.time()
    reals = itt.chain(np.linspace(-1, 2, 60)[0:-1], np.linspace(2, 3, 40))
    series = pf.julia(
        (complex(real, 0.75) for real in reals),
        xbound=(-1, 1),
        ybound=(-0.75, 1.25),
        update_func=pf.funcs.magnetic_2,
        maxiter=300,
        width=5,
        height=4,
        dpi=200,
    )
    pf.images.save_animation(
        list(series),
        cmap=plt.cm.ocean,
        gamma=0.6,
        file=Path("example_images/julia_animation_ex"),
    )
    print(f"calculation took {round((time.time() - start_time), 2)} seconds")


def lyapunov_example() -> None:

    start_time = time.time()
    string = "AAAAAABBBBBB"
    xbound = (2.5, 3.4)
    ybound = (3.4, 4.0)
    res = pf.lyapunov(
        string, xbound, ybound, width=4, height=3, dpi=300, ninit=2000, niter=2000
    )
    pf.images.markus_lyapunov_image(res, plt.cm.bone, plt.cm.bone_r, gammas=(8, 1))
    plt.savefig("example_images/lyapunov_ex.png")
    print(f"calculation took {round((time.time() - start_time), 2)} seconds")


def randomwalk_example() -> None:

    start_time = time.time()
    basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    moves = pf.construct_moves(basis)
    res = pf.randomwalk(moves, niter=5000000, width=4, height=3, depth=10, dpi=300)
    pf.images.randomwalk_image(res, cmap=plt.cm.gist_yarg, gamma=1.0)
    plt.savefig("example_images/randomwalk_ex.png")
    print(f"calculation took {round((time.time() - start_time), 2)} seconds")


def buddhabrot_example() -> None:  # this will take awhile

    start_time = time.time()
    xbound = (-1.75, 0.85)
    ybound = (-1.10, 1.10)
    res = pf.buddhabrot(
        xbound,
        ybound,
        ncvals=10000000,
        update_func=pf.funcs.power,
        horizon=1.0e6,
        maxiters=(100, 1000, 10000),
        width=4,
        height=3,
        dpi=300,
    )
    pf.images.nebula_image(tuple(res), gamma=0.6)  # type: ignore[arg-type]
    plt.savefig("example_images/buddhabrot_ex.png")
    print(f"calculation took {round((time.time() - start_time), 2)} seconds")


def main() -> None:
    args = parse_args().parse_args()
    if args.mandelbrot:
        print("Running Mandelbrot example...")
        mandelbrot_example()
    if args.julia:
        print("Running Julia animation example...")
        julia_animation_example()
    if args.lyapunov:
        print("Running Lyapunov example...")
        lyapunov_example()
    if args.randomwalk:
        print("Running random walk example...")
        randomwalk_example()
    if args.buddhabrot:
        print("Running Buddhabrot example...")
        buddhabrot_example()


if __name__ == "__main__":
    main()

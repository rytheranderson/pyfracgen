## Authors

- Ryther Anderson

## Description
Python Fractal Generation is a package for making interesting/aesthetic fractal
images quickly (for Python) and (hopefully) easily. Many unique fractal images
can be generated using only a few functions.
## Installation

`pyfracgen` can currently be installed from the following sources (if you want
to install from GH, probably do so using `poetry`).

### Git
```bash
git clone https://github.com/rytheranderson/pyfracgen.git
poetry install
```

### PyPI
```bash
pip install pyfracgen
```

## Example Images

All the package functions can be accessed from a single import:
```
import pyfracgen as pf
from matplotlib import pyplot as plt
```

### Mandelbrot Set
![](https://github.com/rytheranderson/pyfracgen/raw/main/example_images/mandelbrot_ex.png?raw=true)

Image produced with this code:

```python
# x and y bounds, x is the real part and y is the imaginary part
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
```

### Julia Set Animation
![](https://github.com/rytheranderson/pyfracgen/raw/main/example_images/julia_animation_ex.gif?raw=true)

Animation produced with this code:

```python
import itertools as itt

reals = itt.chain(np.linspace(-1, 2, 60)[0:-1],  np.linspace(2, 3, 40))
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
```

### Markus-Lyapunov Fractal
![](https://github.com/rytheranderson/pyfracgen/raw/main/example_images/lyapunov_ex.png?raw=true)

Image produced with this code:

```python
string = "AAAAAABBBBBB"
xbound = (2.5, 3.4)
ybound = (3.4, 4.0)
res = pf.lyapunov(
    string, xbound, ybound, width=4, height=3, dpi=300, ninit=2000, niter=2000
)
pf.images.markus_lyapunov_image(res, plt.cm.bone, plt.cm.bone_r, gammas=(8, 1))
plt.savefig("example_images/lyapunov_ex.png")
```

### Random Walk
![](https://github.com/rytheranderson/pyfracgen/raw/main/example_images/randomwalk_ex.png?raw=true)

Image produced with this code:

```python
basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
moves = pf.construct_moves(basis)
res = pf.randomwalk(moves, niter=5000000, width=4, height=3, depth=10, dpi=300)
pf.images.randomwalk_image(res, cmap=plt.cm.gist_yarg, gamma=1.0)
plt.savefig("example_images/randomwalk_ex.png")
```

### Buddhabrot with Nebula Coloring
![](https://github.com/rytheranderson/pyfracgen/raw/main/example_images/buddhabrot_ex.png?raw=true)

Image produced with this code:

```python
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
pf.images.nebula_image(tuple(res), gamma=0.4)  # type: ignore[arg-type]
plt.savefig("example_images/buddhabrot_ex.png")
```
## Fractal "Types" Supported
* Mandelbrot
* Julia
* Buddhabrot
* Markus-Lyapunov
* 3D random walks

## Image Creation
* Function `image` wrapping `matplotlib.pyplot.imshow`
* Function `nebula_image` for Buddhabrot ["nebula"](https://en.wikipedia.org/wiki/Buddhabrot#Nuances) coloration
* Function `markus_lyapunov_image` for [Markus-Lyapunov](https://doi.org/10.1016/0097-8493(89)90019-8) coloration
* Function `randomwalk_image` for coloring 3D random walks with depth
* Function `save_animation` for animating a sequence of results

## More than Quadratic Polynomials
Mandelbrot, Julia, and Buddhabrot fractal images are almost always created by
iterating the function $f_c(z) = z^2 + c$. Makes sense, since this function is
part of the definition of the Mandelbrot set. However, you can iterate lots of
other functions to produce similarly striking images: see the `updaters` module
of `pyfracgen` for a few examples.

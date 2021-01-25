import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.linalg import svd
import starry
import os


class Star(object):
    def __init__(
        self, nlon=300, ydeg=15, linear=True, eps=1e-12, smoothing=0.1
    ):
        # Generate a uniform intensity grid
        self.nlon = nlon
        self.nlat = nlon // 2
        self.lon = np.linspace(-180, 180, self.nlon)
        self.lat = np.linspace(-90, 90, self.nlat)
        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        self.intensity = np.zeros_like(self.lat)
        self.linear = linear

        # Instantiate a starry map
        self.map = starry.Map(ydeg, lazy=False)

        # cos(lat)-weighted SHT
        w = np.cos(self.lat.flatten() * np.pi / 180)
        P = self.map.intensity_design_matrix(
            lat=self.lat.flatten(), lon=self.lon.flatten()
        )
        PTSinv = P.T * (w ** 2)[None, :]
        self.Q = np.linalg.solve(PTSinv @ P + eps * np.eye(P.shape[1]), PTSinv)
        if smoothing > 0:
            l = np.concatenate(
                [np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)]
            )
            s = np.exp(-0.5 * l * (l + 1) * smoothing ** 2)
            self.Q *= s[:, None]

    def _angular_distance(self, lam1, lam2, phi1, phi2):
        # https://en.wikipedia.org/wiki/Great-circle_distance
        return (
            np.arccos(
                np.sin(phi1 * np.pi / 180) * np.sin(phi2 * np.pi / 180)
                + np.cos(phi1 * np.pi / 180)
                * np.cos(phi2 * np.pi / 180)
                * np.cos((lam2 - lam1) * np.pi / 180)
            )
            * 180
            / np.pi
        )

    def reset(self):
        self.intensity = np.zeros_like(self.lat)

    def add_spot(self, lon, lat, radius, contrast):
        idx = self._angular_distance(lon, self.lon, lat, self.lat) <= radius
        if self.linear:
            self.intensity[idx] -= contrast
        else:
            self.intensity[idx] = -contrast

    def get_y(self):
        return self.Q @ self.intensity.flatten()


# Settings
norm = [
    Normalize(vmin=0.9, vmax=1.025),
    Normalize(vmin=0.89, vmax=1.035),
    Normalize(vmin=0.87, vmax=1.055),
]
np.random.seed(3)

# Instantiate maps at low and high degree
star15 = Star(ydeg=15, smoothing=0.1)
star35 = Star(ydeg=35, smoothing=0.03)
stars = [star15, star15, star15, star35]

# The Ylm coeffs for each star
y = [None, None, None, None]

# Star #1: One spot
stars[0].reset()
stars[0].add_spot(30, 30, 25, 0.2)
y[0] = stars[0].get_y()

# Star #2: Ten spots at +/- 30 latitude
stars[1].reset()
sign = 1
for n in range(10):
    longitude = np.random.uniform(-180, 180)
    latitude = sign * 30 + 5 * np.random.randn()
    stars[1].add_spot(longitude, latitude, 10 + np.random.randn(), 0.1)
    sign *= -1
y[1] = stars[1].get_y()

# Star #3: 20 spots distributed isotropically
stars[2].reset()
for n in range(20):
    longitude = np.random.uniform(-180, 180)
    latitude = (np.arccos(np.random.uniform(-1, 1)) - np.pi / 2) * 180 / np.pi
    stars[2].add_spot(longitude, latitude, 10 + np.random.randn(), 0.2)
y[2] = stars[2].get_y()

# Star #4: An alien message
image = plt.imread(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "greetings.png")
)
image = np.flipud(np.mean(image[:, :, :3], axis=-1))
image *= 0.15
image -= 0.15
stars[3].intensity = image
y[3] = stars[3].get_y()

# Random inclinations for the ensemble plots
incs = [60, 85, np.arccos(np.random.random(50)) * 180 / np.pi]

# Loop over each star
for k in range(len(stars)):

    # Loop over each inclination
    for j in range(len(incs)):

        # Set up the figure
        fig, ax = plt.subplots(
            2, 3, figsize=(12, 3), gridspec_kw={"height_ratios": [1, 0.5]}
        )

        # Compute the null space operators
        t = np.linspace(0, 1, 1000, endpoint=False)
        theta = 360 * (t - 0.5)
        map = stars[k].map

        if not hasattr(incs[j], "__len__"):

            # Compute the operators at a single inclination
            map.inc = incs[j]
            A = map.design_matrix(theta=theta)
            rank = np.linalg.matrix_rank(A)
            _, _, VT = svd(A)
            N = VT[rank:].T @ VT[rank:]  # null space operator
            R = VT[:rank].T @ VT[:rank]  # row space operator

        else:

            # Compute the operators for ensemble analysis at
            # all inclinations. In this case, the null space
            # consists of only the odd modes with l > 3
            r = np.zeros(map.Ny)
            l = np.concatenate(
                [
                    l * np.ones(2 * l + 1, dtype=int)
                    for l in range(map.ydeg + 1)
                ]
            )
            r[l == 1] = 1  # dipole
            r[l % 2 == 0] = 1  # all even modes
            R = np.diag(r)
            N = np.eye(map.Ny) - R
            N[0] = 1  # hack so we can visualize it on the same colorscale

        # True map
        map[:, :] = y[k]
        image = map.render(projection="moll")
        image += 1 - np.nanmedian(image)
        map.show(ax=ax[0, 0], image=image, projection="moll", norm=norm[0])
        ax[0, 0].annotate(
            "true",
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 10),
            textcoords="offset points",
            va="bottom",
            ha="center",
            fontsize=16,
        )
        if not hasattr(incs[j], "__len__"):
            flux = map.flux(theta=theta)
            flux -= np.mean(flux)
            flux *= 1e3
            ax[1, 0].plot(t, flux)
            ymax = 1.2 * np.max(np.abs(flux))
        else:
            ymax = 0.0
            for inc in incs[j]:
                map.inc = inc
                flux = map.flux(theta=theta)
                flux -= np.mean(flux)
                flux *= 1e3
                ax[1, 0].plot(t, flux, lw=0.75, alpha=0.25, color="C0")
                ymax = max(ymax, 1.2 * np.max(np.abs(flux)))
        ax[1, 0].set_ylim(-ymax, ymax)

        # row space only
        map[:, :] = R @ y[k]
        image = map.render(projection="moll")
        image += 1 - np.nanmedian(image)
        map.show(ax=ax[0, 1], image=image, projection="moll", norm=norm[1])
        ax[0, 1].annotate(
            "preimage",
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 10),
            textcoords="offset points",
            va="bottom",
            ha="center",
            fontsize=16,
        )
        if not hasattr(incs[j], "__len__"):
            flux = map.flux(theta=theta)
            flux -= np.mean(flux)
            flux *= 1e3
            ax[1, 1].plot(t, flux)
        else:
            for inc in incs[j]:
                map.inc = inc
                flux = map.flux(theta=theta)
                flux -= np.mean(flux)
                flux *= 1e3
                ax[1, 1].plot(t, flux, lw=0.75, alpha=0.25, color="C0")
        ax[1, 1].set_ylim(*ax[1, 0].get_ylim())

        # Null space only
        map[:, :] = N @ y[k]
        image = map.render(projection="moll")
        image += 1 - np.nanmedian(image)
        map.show(ax=ax[0, 2], image=image, projection="moll", norm=norm[2])
        ax[0, 2].annotate(
            "null space",
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 10),
            textcoords="offset points",
            va="bottom",
            ha="center",
            fontsize=16,
        )
        if not hasattr(incs[j], "__len__"):
            flux = map.flux(theta=theta)
            flux -= np.mean(flux)
            flux *= 1e3
            ax[1, 2].plot(t, flux)
        else:
            for inc in incs[j]:
                map.inc = inc
                flux = map.flux(theta=theta)
                flux -= np.mean(flux)
                flux *= 1e3
                ax[1, 2].plot(t, flux, lw=0.75, alpha=0.25, color="C0")
        ax[1, 2].set_ylim(*ax[1, 0].get_ylim())

        # Appearance
        for n in range(3):
            ax[1, n].spines["top"].set_visible(False)
            ax[1, n].spines["right"].set_visible(False)
            ax[1, n].set_xlabel("rotation phase", fontsize=8)
            ax[1, n].set_ylabel("flux [ppt]", fontsize=8)
            ax[1, n].set_xticks([0, 0.25, 0.5, 0.75, 1])
            for tick in (
                ax[1, n].xaxis.get_major_ticks()
                + ax[1, n].yaxis.get_major_ticks()
            ):
                tick.label.set_fontsize(6)
            ax[1, n].tick_params(direction="in")
            ax[0, n].set_rasterization_zorder(2)

        # We're done
        fig.savefig(
            os.path.abspath(__file__).replace(
                ".py",
                "_{}{}.pdf".format(
                    incs[j] if not hasattr(incs[j], "__len__") else "",
                    chr(97 + k),
                ),
            ),
            bbox_inches="tight",
        )

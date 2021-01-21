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
ydeg = 15
norm = Normalize(vmin=0.9, vmax=1.025)
star = Star(ydeg=ydeg)
nspots = [1, 5, 20]
radius = [25, 20, 10]
contrast = [0.2, 0.1, 0.2]
seeds = [3, 3, 3]
nplots = len(nspots)

#
for k in range(nplots):

    # Seed the randomizer
    np.random.seed(seeds[k])

    # Get the ylm expansion
    star.reset()
    for n in range(nspots[k]):
        longitude = np.random.uniform(-180, 180)
        latitude = (
            (np.arccos(np.random.uniform(-1, 1)) - np.pi / 2) * 180 / np.pi
        )
        star.add_spot(longitude, latitude, radius[k], contrast[k])
    y = star.get_y()

    # Set up the figure
    fig, ax = plt.subplots(
        2, 3, figsize=(12, 3), gridspec_kw={"height_ratios": [1, 0.5]}
    )

    # Compute the null space operators at I = 60
    t = np.linspace(0, 1, 1000, endpoint=False)
    theta = 360 * (t - 0.5)
    map = starry.Map(ydeg, lazy=False)
    map.inc = 60
    A = map.design_matrix(theta=theta)
    rank = np.linalg.matrix_rank(A)
    _, _, VT = svd(A)
    N = VT[rank:].T @ VT[rank:]  # null space operator
    R = VT[:rank].T @ VT[:rank]  # row space operator

    # True map
    map[:, :] = y
    image = map.render(projection="moll")
    image += 1
    map.show(ax=ax[0, 0], image=image, projection="moll", norm=norm)
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
    flux = map.flux(theta=theta)
    flux -= np.mean(flux)
    flux *= 1e3
    ax[1, 0].plot(t, flux)
    ymax = 1.2 * np.max(np.abs(flux))
    ax[1, 0].set_ylim(-ymax, ymax)

    # row space only
    map[:, :] = R @ y
    image = map.render(projection="moll")
    image += 1
    map.show(ax=ax[0, 1], image=image, projection="moll", norm=norm)
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
    flux = map.flux(theta=theta)
    flux -= np.mean(flux)
    flux *= 1e3
    ax[1, 1].plot(t, flux)
    ax[1, 1].set_ylim(*ax[1, 0].get_ylim())

    # Null space only
    map[:, :] = N @ y
    image = map.render(projection="moll")
    image += 1
    map.show(ax=ax[0, 2], image=image, projection="moll", norm=norm)
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
    flux = map.flux(theta=theta)
    flux -= np.mean(flux)
    flux *= 1e3
    ax[1, 2].plot(t, flux)
    ax[1, 2].set_ylim(*ax[1, 0].get_ylim())

    # Appearance
    for n in range(3):
        ax[1, n].spines["top"].set_visible(False)
        ax[1, n].spines["right"].set_visible(False)
        ax[1, n].set_xlabel("rotation phase", fontsize=8)
        ax[1, n].set_ylabel("flux [ppt]", fontsize=8)
        ax[1, n].set_xticks([0, 0.25, 0.5, 0.75, 1])
        for tick in (
            ax[1, n].xaxis.get_major_ticks() + ax[1, n].yaxis.get_major_ticks()
        ):
            tick.label.set_fontsize(6)
        ax[1, n].tick_params(direction="in")

    # We're done
    fig.savefig(
        os.path.abspath(__file__).replace(".py", "_{}.pdf".format(k)),
        bbox_inches="tight",
    )

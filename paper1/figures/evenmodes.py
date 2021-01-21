import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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

    def get_y(self, smoothing=0.1):
        return self.Q @ self.intensity.flatten()


# Settings
np.random.seed(2)
ydeg = 15
nspots = 10
radius = lambda: 15
longitude = lambda: np.random.uniform(-180, 180)
latitude = lambda: (
    (1 if np.random.random() < 0.5 else -1)
    * min(90, max(0, 45 + 5 * np.random.randn()))
)
contrast = lambda: 0.1
star = Star(ydeg=ydeg)
for n in range(nspots):
    star.add_spot(longitude(), latitude(), radius(), contrast())
y = star.get_y()

map = starry.Map(ydeg, lazy=False)
map[:, :] = y


fig, ax = plt.subplots(1, 3, figsize=(12, 3))
norm = Normalize(vmin=-0.25, vmax=0.01)

# True map
map[:, :] = y
map.show(ax=ax[0], projection="moll", norm=norm)
ax[0].annotate(
    "true",
    xy=(0.5, 1.0),
    xycoords="axes fraction",
    xytext=(0, 10),
    textcoords="offset points",
    va="bottom",
    ha="center",
    fontsize=16,
)

# image only
map[:, :] = y
map[3::2, :] = 0
map.show(ax=ax[1], projection="moll", norm=norm)
ax[1].annotate(
    "preimage",
    xy=(0.5, 1.0),
    xycoords="axes fraction",
    xytext=(0, 10),
    textcoords="offset points",
    va="bottom",
    ha="center",
    fontsize=16,
)

# Null space only
map[:, :] = y
map[1:3, :] = 0
map[4::2, :] = 0
map.show(ax=ax[2], projection="moll", norm=norm)
ax[2].annotate(
    "null space",
    xy=(0.5, 1.0),
    xycoords="axes fraction",
    xytext=(0, 10),
    textcoords="offset points",
    va="bottom",
    ha="center",
    fontsize=16,
)

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"), bbox_inches="tight"
)

import matplotlib.pyplot as plt
import numpy as np
import os


def RAxisAngle(axis=[0, 1, 0], theta=0):
    """
    Rotate an arbitrary point by an axis and an angle.

    """
    cost = np.cos(theta)
    sint = np.sin(theta)

    return np.reshape(
        [
            cost + axis[0] * axis[0] * (1 - cost),
            axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
            axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
            axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
            cost + axis[1] * axis[1] * (1 - cost),
            axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
            axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
            axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
            cost + axis[2] * axis[2] * (1 - cost),
        ],
        [3, 3],
    )


def get_ortho_latitude_lines(inc=np.pi / 2, obl=0, dlat=np.pi / 6, npts=1000):
    """
    Return the lines of constant latitude on an orthographic projection.

    """
    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Latitude lines
    res = []
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:

        # Figure out the equation of the ellipse
        y0 = np.sin(lat) * si
        a = np.cos(lat)
        b = a * ci
        x = np.linspace(-a, a, npts)
        y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
        y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

        # Mask lines on the backside
        if si != 0:
            if inc > np.pi / 2:
                ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                y1[y1 < ymax] = np.nan
                ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                y2[y2 < ymax] = np.nan
            else:
                ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                y1[y1 > ymax] = np.nan
                ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                y2[y2 > ymax] = np.nan

        # Rotate them
        for y in (y1, y2):
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr, yr))

    return res


def get_ortho_longitude_lines(
    inc=np.pi / 2, obl=0, theta=0, dlon=np.pi / 6, npts=1000
):
    """
    Return the lines of constant longitude on an orthographic projection.

    """
    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Are we (essentially) equator-on?
    equator_on = (inc > 88 * np.pi / 180) and (inc < 92 * np.pi / 180)

    # Longitude grid lines
    res = []
    if equator_on:
        offsets = np.arange(-np.pi / 2, np.pi / 2, dlon)
    else:
        offsets = np.arange(0, 2 * np.pi, dlon)

    for offset in offsets:

        # Super hacky, sorry. This can probably
        # be coded up more intelligently.
        if equator_on:
            sgns = [1]
            if np.cos(theta + offset) >= 0:
                bsgn = 1
            else:
                bsgn = -1
        else:
            bsgn = 1
            if np.cos(theta + offset) >= 0:
                sgns = np.array([1, -1])
            else:
                sgns = np.array([-1, 1])

        for lon, sgn in zip([0, np.pi], sgns):

            # Viewed at i = 90
            y = np.linspace(-1, 1, npts)
            b = bsgn * np.sin(lon - theta - offset)
            x = b * np.sqrt(1 - y ** 2)
            z = sgn * np.sqrt(np.abs(1 - x ** 2 - y ** 2))

            if equator_on:

                pass

            else:

                # Rotate by the inclination
                R = RAxisAngle([1, 0, 0], np.pi / 2 - inc)
                v = np.vstack(
                    (x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1))
                )
                x, y, _ = np.dot(R, v)

                # Mask lines on the backside
                if si != 0:
                    if inc < np.pi / 2:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[: imax + 1] = np.nan
                    else:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[imax:] = np.nan

            # Rotate by the obliquity
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr, yr))

    return res


theta = 15

for inc in [30, 45, 60, 75, 85]:

    # Draw the wireframe
    fig, ax = plt.subplots(1, figsize=(2.5, 2.5))
    x = np.linspace(-1, 1, 10000)
    y = np.sqrt(1 - x ** 2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.plot(x, y, "k-", lw=1, zorder=102)
    ax.plot(x, -y, "k-", lw=1, zorder=102)
    r = 1.035
    x = np.linspace(-r, r, 10000)
    y = np.sqrt(r ** 2 - x ** 2)
    ax.plot(x, y, "w-", lw=1, zorder=103)
    ax.plot(x, -y, "w-", lw=1, zorder=103)
    lat_lines = get_ortho_latitude_lines(inc=inc * np.pi / 180)
    for n, l in enumerate(lat_lines):
        if n == 4:
            ax.plot(l[0], l[1], "k", lw=1, zorder=104)
        else:
            ax.plot(l[0], l[1], "#aaaaaa", lw=0.5, zorder=100)
    lon_lines = get_ortho_longitude_lines(
        inc=inc * np.pi / 180, theta=np.pi + theta * np.pi / 180
    )
    for n, l in enumerate(lon_lines):
        ax.plot(l[0], l[1], "#aaaaaa", lw=0.5, zorder=100)

    # Draw the axis
    if inc < 50:
        ymax = 2.25 / (0.5 + inc / 50)
    else:
        ymax = 1.5
    y = np.linspace(-ymax, ymax, 1000) * np.sin(inc * np.pi / 180)
    ypole = np.sin(inc * np.pi / 180)
    y[(y < ypole) & (y > -1)] = np.nan
    ax.plot(np.zeros_like(y), y, "k-", lw=0.75, zorder=104)

    # Hack a circular arrow indicating the spin
    shrink = 0.4
    offset = 0.45
    x, y = lat_lines[8]
    y = offset + shrink * (y - ypole) + ypole
    x = shrink * x
    ax.plot(x, y, "r", zorder=105)
    x, y = lat_lines[9]
    y = offset + shrink * (y - ypole) + ypole
    x = shrink * x
    ax.plot(x[:100], y[:100], "r", zorder=105)
    ax.plot(x[-100:], y[-100:], "r", zorder=105)
    xa = x[100]
    ya = y[100]
    dx = 100 * (x[100] - x[99])
    dy = 100 * (y[100] - y[99])
    plt.arrow(
        xa,
        ya,
        dx,
        dy,
        length_includes_head=True,
        head_length=np.sqrt(dx ** 2 + dy ** 2),
        head_width=0.1,
        color="r",
        zorder=105,
    )

    ax.axis("off")
    fig.savefig(
        os.path.abspath(__file__).replace(".py", "_{}.pdf".format(inc)),
        bbox_inches="tight",
    )

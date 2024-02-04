"""
Utility functions for displaying image data

Arthur McCray
amccray@anl.gov
"""

from textwrap import dedent

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage
from ipywidgets import interact
from scipy import ndimage as ndi

from .colorwheel import color_im, get_cmap


def show_im(
    image,
    title=None,
    simple=False,
    origin="upper",
    cbar=None,
    cbar_title="",
    scale=None,
    save=None,
    **kwargs,
):
    """Display an image on a new axis.

    Takes a 2D array and displays the image in grayscale with optional title on
    a new axis. In general it's nice to have things on their own axes, but if
    too many are open it's a good idea to close with plt.close('all').

    Args:
        image (2D array): Image to be displayed.
        title (str): (`optional`) Title of plot.
        simple (bool): (`optional`) Default output or additional labels.

            - True, will just show image.
            - False, (default) will show a colorbar with axes labels, and will adjust the
              contrast range for images with a very small range of values (<1e-12).

        origin (str): (`optional`) Control image orientation.

            - 'upper': (default) (0,0) in upper left corner, y-axis goes down.
            - 'lower': (0,0) in lower left corner, y-axis goes up.

        cbar (bool): (`optional`) Choose to display the colorbar or not. Only matters when
            simple = False.
        cbar_title (str): (`optional`) Title attached to the colorbar (indicating the
            units or significance of the values).
        scale (float): Scale of image in nm/pixel. Axis markers will be given in
            units of nanometers.

    Returns:
        None
    """
    image = np.array(image)
    if image.dtype == "bool":
        image = image.astype("int")
    ndim = np.ndim(image)
    if ndim == 2:
        pass
    elif ndim == 3:
        if image.shape[2] not in ([3, 4]):
            if image.shape[0] != 1:
                print(
                    dedent(
                        """\
                    Input image is 3D and does not seem to be a color image.
                    Summing along first axis"""
                    )
                )
            image = np.sum(image, axis=0)
    else:
        print(f"Input image is of dimension {ndim}. Please input 2D image.")
        return
    if cbar is None and simple:
        cbar = False
    elif cbar is None:
        cbar = True
    if simple and title is None:
        # all this to avoid a white border when saving the image
        fig = plt.figure()
        aspect = image.shape[0] / image.shape[1]
        size = kwargs.get("size", (4, 4 * aspect))
        if isinstance(size, (int, float)):
            size = (size, size)
        fig.set_size_inches(size)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
    else:
        _fig, ax = plt.subplots()

    cmap = kwargs.get("cmap", "gray")
    if simple:
        vmin = kwargs.get("vmin", None)
        vmax = kwargs.get("vmax", None)
    else:
        # adjust coontrast range if minimal range detected
        # avoids people thinking 0 phase shift images (E-15) are real
        vmin = kwargs.get("vmin", np.min(image) - 1e-12)
        vmax = kwargs.get("vmax", np.max(image) + 1e-12)

    im = ax.matshow(image, origin=origin, vmin=vmin, vmax=vmax, cmap=cmap)

    if title is not None:
        ax.set_title(str(title))

    if simple:
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        pass

    else:
        plt.tick_params(axis="x", top=False)
        ax.xaxis.tick_bottom()
        ax.tick_params(direction="in")
        if scale is None:
            ticks_label = "pixels"
        else:

            def mjrFormatter(x, pos):
                return f"{scale*x:.3g}"

            fov = scale * max(image.shape[0], image.shape[1])
            if kwargs.get("scale_units", None) is None:
                if fov < 4e3:  # if fov < 4um use nm scale
                    ticks_label = " nm "
                elif fov > 4e6:  # if fov > 4mm use m scale
                    ticks_label = "  m  "
                    scale /= 1e9
                else:  # if fov between the two, use um
                    ticks_label = r" $\mu$m "
                    scale /= 1e3
            else:
                ticks_label = kwargs.get("scale_units")

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

        if origin == "lower":
            ax.text(y=0, x=0, s=ticks_label, rotation=-45, va="top", ha="right")
        elif origin == "upper":  # keep label in lower left corner
            ax.text(
                y=image.shape[0], x=0, s=ticks_label, rotation=-45, va="top", ha="right"
            )

    if cbar:
        plt.colorbar(im, ax=ax, pad=0.02, format="%g", label=str(cbar_title))

    if save:
        # print("saving: ", save)
        dpi = kwargs.get("dpi", 400)
        if simple and title is None:
            plt.savefig(save, dpi=dpi, bbox_inches=0)
        else:
            plt.savefig(save, dpi=dpi, bbox_inches="tight")

    plt.show()
    return


def show_stack(
    images,
    titles=None,
    scale_each=True,
    titletext="",
    origin="upper",
):
    """
    Uses ipywidgets.interact to allow user to view multiple images on the same
    axis using a slider. There is likely a better way to do this, but this was
    the first one I found that works...

    Args:
        images (list): List of 2D arrays. Stack of images to be shown.
        origin (str): (`optional`) Control image orientation.
        title (bool): (`optional`) Try and pull a title from the signal objects.
    Returns:
        None
    """
    _fig, _ax = plt.subplots()
    images = np.array(images)
    if not scale_each:
        vmin = np.min(images)
        vmax = np.max(images)

    N = images.shape[0]
    if titles is not None:
        assert len(titles) == len(images)

    def view_image(i=0):
        if scale_each:
            _im = plt.imshow(
                images[i], cmap="gray", interpolation="nearest", origin=origin
            )
        else:
            _im = plt.imshow(
                images[i],
                cmap="gray",
                interpolation="nearest",
                origin=origin,
                vmin=vmin,
                vmax=vmax,
            )

        if titles is not None:
            plt.title(f"{titletext} {titles[i]}")

    interact(view_image, i=(0, N - 1))
    return


def show_2D(
    mag_x,
    mag_y,
    mag_z=None,
    a=0,
    l=None,
    w=None,
    title=None,
    color=True,
    cmap=None,
    cbar=False,
    origin="upper",
    save=None,
    ax=None,
    rad=None,
    **kwargs,
):
    """Display a 2D vector arrow plot.

    Displays an an arrow plot of a vector field, with arrow length scaling with
    vector magnitude. If color=True, a colormap will be displayed under the
    arrow plot.

    If mag_z is included and color=True, a spherical colormap will be used with
    color corresponding to in-plane and white/black to out-of-plane vector
    orientation.

    Args:
        mag_x (2D array): x-component of magnetization.
        mag_y (2D array): y-component of magnetization.
        mag_z (2D array): optional z-component of magnetization.
        a (int): Number of arrows to plot along the x and y axes. Default 15.
        l (float): Scale factor of arrows. Larger l -> shorter arrows. Default None
            guesses at a good value. None uses matplotlib default.
        w (float): Width scaling of arrows. None uses matplotlib default.
        title (str): (`optional`) Title for plot. Default None.
        color (bool): (`optional`) Whether or not to show a colormap underneath
            the arrow plot. Color image is made from colorwheel.color_im().
        hsv (bool): (`optional`) Only relevant if color == True. Whether to use
            an hsv or 4-fold color-wheel in the color image.
        origin (str): (`optional`) Control image orientation.
        save (str): (`optional`) Path to save the figure.

    Returns:
        fig: Returns the figure handle.
    """
    assert mag_x.ndim == mag_y.ndim
    if mag_x.ndim == 3:
        print("Summing along first axis")
        mag_x = np.sum(mag_x, axis=0)
        mag_y = np.sum(mag_y, axis=0)
        if mag_z is not None:
            mag_z = np.sum(mag_z, axis=0)

    if a > 0:
        # a = ((mag_x.shape[0] - 1) // a) + 1
        a = int(((mag_x.shape[0] - 1) / a) + 1)

    dimy, dimx = mag_x.shape
    X = np.arange(0, dimx, 1)
    Y = np.arange(0, dimy, 1)
    U = mag_x
    V = mag_y

    sz_inches = 3.0
    if color:
        if rad is None:
            rad = mag_x.shape[0] // 16
            rad = max(rad, 16)
            pad = 10  # pixels
            width = np.shape(mag_y)[1] + 2 * rad + pad
            aspect = dimy / width
        elif rad == 0:
            width = np.shape(mag_y)[1]
            aspect = dimy / width
        else:
            pad = 10  # pixels
            width = np.shape(mag_y)[1] + 2 * rad + pad
            aspect = dimy / width
    else:
        aspect = dimy / dimx

    if ax is None:
        if save is not None:  # and title is None: # to avoid white border when saving
            fig = plt.figure()
            size = (sz_inches, sz_inches * aspect)
            # fig.set_size_inches(sz_inches, sz_inches / aspect)
            fig.set_size_inches(size)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            fig.add_axes(ax)
        else:
            fig, ax = plt.subplots()
        ax.set_aspect(aspect)
    if color:
        cmap = get_cmap(cmap, **kwargs)
        im = ax.matshow(
            color_im(
                mag_x,
                mag_y,
                mag_z,
                cmap=cmap,
                rad=rad,
                **kwargs,
            ),
            cmap=cmap,
            origin=origin,
        )
        if cbar:
            # TODO set cbar height, and labels to be 0 - 2Pi
            plt.colorbar(
                im,
                ax=ax,
                pad=0.02,
                format="%g",
                label=str(kwargs.get("cbar_title", "")),
            )
        arrow_color = "white"
        ax.set_xticks([])
        ax.set_yticks([])

    else:
        arrow_color = "black"
    arrow_color = kwargs.get("arrow_color", arrow_color)

    if a > 0:
        ashift = (dimx - 1) % a // 2
        q = ax.quiver(
            X[ashift::a],
            Y[ashift::a],
            U[ashift::a, ashift::a],
            V[ashift::a, ashift::a],
            units="xy",
            scale=l,
            scale_units="xy",
            width=w,
            angles="xy",
            pivot="mid",
            color=arrow_color,
        )

    if not color and a > 0:
        # qk = ax.quiverkey(
        #     q, X=0.95, Y=0.98, U=1, label=r"$Msat$", labelpos="S", coordinates="axes"
        # )
        # qk.text.set_backgroundcolor("w")
        if origin == "upper":
            ax.invert_yaxis()

    if title is not None:
        tr = False
        ax.set_title(title)
    else:
        tr = True

    plt.tick_params(axis="x", labelbottom=False, bottom=False, top=False)
    plt.tick_params(axis="y", labelleft=False, left=False, right=False)
    # ax.set_aspect(aspect)

    if save is not None:
        if not color:
            tr = False
        print(f"Saving: {save}")
        plt.axis("off")
        dpi = kwargs.get("dpi", max(dimy, dimx) * 5 / sz_inches)
        # sets dpi to 5 times original image dpi so arrows are reasonably sharp
        if title is None:  # for no padding
            plt.savefig(save, dpi=dpi, bbox_inches=0, transparent=tr)
        else:
            plt.savefig(save, dpi=dpi, bbox_inches="tight", transparent=tr)

    return

def show_fft(fft, title=None, **kwargs):
    """Display the log of the abs of a FFT

    Args:
        fft (ndarray): 2D image
        title (str, optional): title of image. Defaults to None.
        **kwargs: passed to show_im()
    """
    fft = np.copy(fft)
    nonzeros = np.nonzero(fft)
    fft[nonzeros] = np.log10(np.abs(fft[nonzeros]))
    fft = fft.real
    show_im(fft, title=title, **kwargs)


def show_log(im, title=None, **kwargs):
    """Display the log of an image

    Args:
        im (ndarray): 2D image
        title (str, optional): title of image. Defaults to None.
        **kwargs: passed to show_im()
    """
    im = np.copy(im)
    nonzeros = np.nonzero(im)
    im[nonzeros] = np.log(np.abs(im[nonzeros]))
    show_im(im, title=title, **kwargs)


def show_im_peaks(im=None, peaks=None, peaks2=None, size=None, title=None, **kwargs):
    """
    peaks an array [[y1,x1], [y2,x2], ...]
    """
    _fig, ax = plt.subplots()
    if im is not None:
        ax.matshow(im, cmap="gray", **kwargs)
    if peaks is not None:
        peaks = np.array(peaks)
        ax.plot(
            peaks[:, 1],
            peaks[:, 0],
            c="r",
            alpha=0.9,
            ms=size,
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    if peaks2 is not None and np.size(peaks2) != 0:
        peaks2 = np.array(peaks2)
        ax.plot(
            peaks2[:, 1],
            peaks2[:, 0],
            c="b",
            alpha=0.9,
            ms=size,
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    ax.set_aspect(1)
    if title is not None:
        ax.set_title(str(title), pad=0)
    plt.show()


def get_histo(im, minn=None, maxx=None, numbins=None):
    """
    gets a histogram of a list of datapoints (im), specify minimum value, maximum value,
    and number of bins
    """
    im = np.array(im)
    if minn is None:
        minn = np.min(im)
    if maxx is None:
        maxx = np.max(im)
    if numbins is None:
        numbins = min(np.size(im) // 20, 100)
        print(f"{numbins} bins")
    _fig, ax = plt.subplots()
    ax.hist(im, bins=np.linspace(minn, maxx, numbins))
    plt.show()

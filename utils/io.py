from pathlib import Path

import os

import numpy as np
import tifffile
import torch
from tifffile import TiffFile
import scipy
from SIPRAD_demo.utils.show import show_im


def read_tif(f):
    """Uses Tifffile load an image and read the scale if there is one.

    Args:
        f (str): file to read

    Raises:
        NotImplementedError: If unknown scale type is given, or Tif series is given.
        RuntimeError: If uknown file type is given, or number of pages in tif is wrong

    Returns:
        tuple:  (image, scale), image given as 2D or 3D numpy array, and scale given in
                nm/pixel if scale is found, or None.
    """
    f = Path(f)
    if f.suffix in [".tif", ".tiff"]:
        with TiffFile(f, mode="r") as tif:
            if tif.imagej_metadata is not None and "unit" in tif.imagej_metadata:
                res = tif.pages[0].tags["XResolution"].value
                scale = res[1] / res[0]  # to nm/pixel
                if tif.imagej_metadata["unit"] == "nm":
                    pass
                elif tif.imagej_metadata["unit"] in ["um", "µm"]:
                    scale *= 1000
                else:
                    print("unknown scale type (just need to add it)")
                    raise NotImplementedError
            else:
                scale = None

            if len(tif.series) != 1:
                raise NotImplementedError(
                    "Not sure how to deal with multi-series stack"
                )
            if len(tif.pages) > 1:  # load as stack
                out_im = []
                for page in tif.pages:
                    out_im.append(page.asarray())
                out_im = np.array(out_im)
            elif len(tif.pages) == 1:  # single image
                out_im = tif.pages[0].asarray()
            else:
                raise RuntimeError(
                    f"Found an unexpected number of pages: {len(tif.pages)}"
                )

    elif f.suffix in [".dm3", ".dm4", ".dm5"]:
        raise NotImplementedError(
            "Removed this due to additional ncempy package requirement"
        )
        # with dm.fileDM(f) as im:
        #     im = im.getDataset(0)
        #     assert im["pixelUnit"][0] == im["pixelUnit"][1]
        #     assert im["pixelSize"][0] == im["pixelSize"][1]

        #     if im["pixelUnit"][0] == "nm":
        #         scale = im["pixelSize"][0]
        #     elif im["pixelUnit"][0] == "µm":
        #         scale = im["pixelSize"][0] * 1000
        #     else:
        #         print("unknown scale type (just need to add it)")
        #         raise NotImplementedError
        #     out_im = im["data"]

    else:
        print(
            "If a proper image file is given, then\n"
            "likely just need to implement with ncempy.read or something."
        )
        raise RuntimeError(f"Unknown filetype given: {f.suffix}")

    return out_im, scale


def save_tif(tif, path, scale, v=1, unit="nm", overwrite=True):
    """
    scale in nm/pixel default,
    """
    res = 1 / scale

    if not overwrite:
        path = overwrite_rename(path)

    if v >= 1:
        print("Saving: ", path)

    tifffile.imwrite(
        path,
        tif.astype("float32"),
        imagej=True,
        resolution=(res, res),
        metadata={"unit": unit},
    )
    return


save_tiff = save_tif  # alias

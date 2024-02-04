from pathlib import Path

import os

import numpy as np
import tifffile
import torch
from skimage.restoration import estimate_sigma
from scipy.signal.windows import tukey
from skimage.metrics import structural_similarity
import scipy
from scipy.spatial.transform import Rotation as R


from SIPRAD_demo.utils.show import show_im
import json
from copy import deepcopy


def sim_image(tphi, pscope, defocus, del_px, Amp=None):
    """
    simulate image numpy
    """

    if Amp is None:
        Amp = np.ones_like(tphi)

    ObjWave = Amp * np.exp(1j * tphi)
    (dim, dim) = tphi.shape
    d2 = dim / 2
    line = np.arange(dim) - float(d2)
    [X, Y] = np.meshgrid(line, line)
    qq = np.sqrt(X**2 + Y**2) / float(dim)
    pscope.defocus = defocus
    im = pscope.getImage(ObjWave, qq, del_px)
    return im


def get_TF(pscope, shape, defocus, del_px):
    ny, nx = shape
    ly = (np.arange(ny) - ny / 2) / ny
    lx = (np.arange(nx) - nx / 2) / nx
    [X, Y] = np.meshgrid(lx, ly)
    qq = np.sqrt(X**2 + Y**2)
    pscope.defocus = defocus
    return np.fft.fftshift(pscope.getTransferFunction(qq, del_px))


def log_run(
    pdict,
    loss_parts,
    minloss_iter,
    best_loss_parts,
    cont=False,
):
    runtype = pdict["runtype"]
    savedir_top = pdict["save_home"]
    # add dloc of original file/sim
    if cont:
        raise NotImplementedError("Not sure where to save this")
        # savedir = savedir_top
    else:
        if not isinstance(savedir_top, Path):
            savedir_top = Path(savedir_top)
        if not savedir_top.exists():
            print("Output directory does not exist")
            raise LookupError

        if isinstance(runtype, Path):
            runtype = runtype.stem
        savedir_top2 = savedir_top / runtype
        savedir_top2.mkdir(exist_ok=True)
        savedir = savedir_top2 / pdict["name"]
        savedir = overwrite_rename_dir(savedir)
        savedir.mkdir()

    # save pdict
    with open(savedir / f"{pdict['name']}_pdict.json", 'w') as f:
        pdict2 = deepcopy(pdict)
        for key, val in pdict2.items():
            if isinstance(val, np.ndarray):
                pdict2[key] = val.tolist()
            elif isinstance(val, Path):
                pdict2[key] = str(val)
            elif isinstance(val, torch.device):
                pdict2[key] = str(val)
        json.dump(pdict2, f, sort_keys=True, indent=4)

    # make
    with open(savedir / f"{pdict['name']}_log.txt", "w") as f:
        f.write(f"Run name: {pdict['name']}\n")
        f.write(f"Mode: {pdict['mode']}\n")
        f.write(f"Reconstruction run type: {runtype}\n")
        f.write(f"Number iterations: {pdict['maxiter']}\n")
        f.write(f"Optimizer learning rates (phase_LR, amp_LR): {pdict['LRs'][0]}\n")
        f.write(f"TV learning rates (phase_LR, amp_LR): {pdict['LRs'][1]}\n")
        f.write(f"Start mode: {pdict['start_mode']}\n")
        f.write(f"Del_px: {pdict['del_px']}\n")
        if pdict['mode'] == "sim":
            f.write(f"Zscale: {pdict['zscale']}\n")
            f.write(f"Original data path: {str(pdict['magfile'])}\n")
            f.write(f"Pre_B: {pdict['prefacs'][0]}\n")
            f.write(f"Pre_E: {pdict['prefacs'][1]}\n")
            f.write(f"Guess phi structural similarity, (full, normalized): [0,1]: {(pdict['guess_phi_SS'], pdict['guess_phi_SS_norm'])}\n")
            f.write(f"Min loss phi structural similarity, (full, normalized): [0,1]: {(pdict['best_phi_SS'], pdict['best_phi_SS_norm'])}\n")
            f.write(f"Guess phi accuracy: {pdict['guess_phi_acc']}\n")
            f.write(f"Min loss phi accuracy: {pdict['best_phi_acc']}\n")
            f.write(f"Guess B structural similarity, (Bx, By, Bmag): [0,1]: {(pdict['guess_Bx_SS'], pdict['guess_By_SS'], pdict['guess_Bmag_SS'])}\n")
            f.write(f"Guess B normed structural similarity, (Bx, By, Bmag): [0,1]: {(pdict['guess_Bx_SS_norm'], pdict['guess_By_SS_norm'], pdict['guess_Bmag_SS_norm'])}\n")
            f.write(f"Guess B accuracy, (Bx, By, Bmag): [0,1]: {(pdict['guess_Bx_acc'], pdict['guess_By_acc'], pdict['guess_Bmag_acc'])}\n")
            f.write(f"Min loss B structural similarity, (Bx, By, Bmag): [0,1]: {(pdict['best_Bx_SS'], pdict['best_By_SS'], pdict['best_Bmag_SS'])}\n")
            f.write(f"Min loss B normed structural similarity, (Bx, By, Bmag): [0,1]: {(pdict['best_Bx_SS_norm'], pdict['best_By_SS_norm'], pdict['best_Bmag_SS_norm'])}\n")
            f.write(f"Min loss B accuracy, (Bx, By, Bmag): [0,1]: {(pdict['best_Bx_acc'], pdict['best_By_acc'], pdict['best_Bmag_acc'])}\n")
            f.write(f"Noiser values: {str(pdict['noise_vals'])}\n")

        f.write(f"Loss_parts (MSE, TV): {str(loss_parts)}\n")
        f.write(f"Min loss iteration: {str(minloss_iter)}\n")
        f.write(f"Min loss parts (MSE, TV): {str(best_loss_parts)}\n")
        f.write(f"Amplitude solving: {pdict['solve_amp']}\n")
        f.write(f"Defvals (nm): {str(pdict['defvals'])}\n")
        f.write(f"Tilt [Tx, Ty] (deg): {str(pdict['tilt'])}\n")
        f.write(f"Using DIP: {str(pdict['DIP'])}\n")

        f.write(f"Notes: {str(pdict['notes'])}\n")
    return savedir


def save_guess(
    name,
    savedir,
    guess_phase,
    best_phase,
    guess_amp,
    best_amp,
    inp_im,
    true_phase=None,
    true_amp=None,
    calc_im=None,
    best_im=None,
    pdict=None,
    gphase = None,
    bphase=None,
    gamp = None,
    bamp = None,
    cont=False,
):
    """
    Cuz input is only one image, can save the input im, and true phase

    Save all relevant output from a run, log_run should already have been run
    saves guess_phase as the unsmoothed version, but for images saves gphase and bphase
    which are smoothed

    Args:
        name (str): name of run
        savedir (Path): Path object, outdir for this run
        guess_mags (Tensor): Guess magnetizations from the AD
        calc_im (Tensor or ndarray): Calculated images from the phase
        test_mag (np.ndarray): Numpy array of the input magnetizations used to creat the
            training images. Will only be saved if not already saved.
        inp_ims (np.ndarray): Numpy array of the input images given to the AD algorithm.
            Organized (Tx, TY): [(0,0), (+,0), (-,0), (0,+), (0,-)]
    """
    if not savedir.exists():
        print("savedir does not exist: ")
        print(savedir)
        return

    res = 1 / pdict["del_px"]
    # save guess phase as .pt
    out_phase_name = savedir / f"{name}_outPhase.pt"

    if not out_phase_name.exists():
        torch.save(guess_phase, out_phase_name)

    if pdict['mode'] == "sim":
        title_gphase = f"Reconstructed phase: SS {pdict['guess_phi_SS']*100:.2f}% | Acc {pdict['guess_phi_acc']*100:.2f}%"
        title_bphase = f"Best phase: SS {pdict['best_phi_SS']*100:.2f}% | Acc {pdict['best_phi_acc']*100:.2f}%"
    else:
        title_gphase = "Reconstructed phase"
        title_bphase = "Best phase"

    # and as png
    show_im(
        gphase-gphase.min(),
        scale=pdict["del_px"],
        title=title_gphase,
        save=savedir / "recon_phase.png",
        dpi=600,
    )

    # save best phase as .pt
    best_phase_name = savedir / f"{name}_bestPhase.pt"
    if not best_phase_name.exists():
        torch.save(best_phase, best_phase_name)

    # and as png
    show_im(
        bphase-bphase.min(),
        scale=pdict["del_px"],
        title=title_bphase,
        save=savedir / "bestPhase.png",
        dpi=600,
    )

    # and the phase that generated that input image(s)
    if pdict['mode'] == "sim":
        if isinstance(true_phase, torch.Tensor):
            true_phase = true_phase.cpu().detach().numpy()
        else:
            assert isinstance(true_phase, np.ndarray)
        res = 1 / pdict["del_px"]
        tifffile.imwrite(
            savedir / "true_phase.tif",
            (true_phase - true_phase.min()).astype("float32"),
            imagej=True,
            resolution=(res, res),
            metadata={"unit": "nm"},
        )
        show_im(
            true_phase-true_phase.min(),
            scale=pdict["del_px"],
            title=f"True phase",
            save=savedir / "true_phase.png",
            dpi=600,
        )

    if pdict['solve_amp']:
        # save guess amp as .pt
        out_amp_name = savedir / f"{name}_outamp.pt"

        if not out_amp_name.exists():
            torch.save(guess_amp, out_amp_name)

        if pdict['mode'] == "sim":
            title_gamp = f"Final recon amp: SS {pdict['guess_amp_SS']*100:.2f}% | Acc {pdict['guess_amp_acc']*100:.2f}%"
            title_bamp = f"Best amp: SS {pdict['best_amp_SS']*100:.2f}% | Acc {pdict['best_amp_acc']*100:.2f}%"
        else:
            title_gamp = "Final recon amp"
            title_bamp = "Best amp"

        # and as png
        show_im(
            gamp,
            scale=pdict["del_px"],
            title=title_gamp,
            save=savedir / "recon_amp.png",
            dpi=600,
        )

        # save best amp as .pt
        best_amp_name = savedir / f"{name}_bestAmp.pt"
        if not best_amp_name.exists():
            torch.save(best_amp, best_amp_name)

        # and as png
        show_im(
            bamp,
            scale=pdict["del_px"],
            title=title_bamp,
            save=savedir / "bestAmp.png",
            dpi=600,
        )
        if pdict['mode'] == "sim":
            # and the amp that generated that input image(s)
            if isinstance(true_amp, torch.Tensor):
                true_amp = true_amp.cpu().detach().numpy()
            else:
                assert isinstance(true_amp, np.ndarray)
            tifffile.imwrite(
                savedir / "true_amp.tif",
                true_amp.astype("float32"),
                imagej=True,
                resolution=(res, res),
                metadata={"unit": "nm"},
            )
            show_im(
                true_amp,
                scale=pdict["del_px"],
                title=f"True amp",
                save=savedir / "true_amp.png",
                dpi=600,
            )

    # save calculated image
    if isinstance(calc_im, torch.Tensor):
        calc_im = calc_im.cpu().detach().numpy()
    else:
        assert isinstance(calc_im, np.ndarray)
    tifffile.imwrite(
        savedir / "calc_im.tif",
        calc_im.astype("float32"),
        imagej=True,
        resolution=(res, res),
        metadata={"unit": "nm"},
    )
    if np.ndim(calc_im) == 3:
        calc_im = calc_im[0]
    show_im(
        calc_im,
        scale=pdict["del_px"],
        title=f"Calculated image from reconstructed phase",
        save=savedir / "calc_im.png",
        dpi=600,
    )

    # save best calculated image
    if isinstance(best_im, torch.Tensor):
        best_im = best_im.cpu().detach().numpy()
    else:
        assert isinstance(best_im, np.ndarray)
    tifffile.imwrite(
        savedir / "best_im.tif",
        best_im.astype("float32"),
        imagej=True,
        resolution=(res, res),
        metadata={"unit": "nm"},
    )

    # save input im
    if isinstance(inp_im, torch.Tensor):
        inp_im = inp_im.cpu().detach().numpy()
    else:
        assert isinstance(inp_im, np.ndarray)
    tifffile.imwrite(
        savedir / "input_im.tif",
        inp_im.astype("float32"),
        imagej=True,
        resolution=(res, res),
        metadata={"unit": "nm"},
    )
    if np.ndim(inp_im) == 3:
        inp_im = inp_im[0]
    show_im(
        inp_im,
        scale=pdict["del_px"],
        title=f"Input image",
        save=savedir / "input_im.png",
        dpi=600,
    )


def overwrite_rename(filepath, spacer="_"):
    """Given a filepath, check if file exists already. If so, add numeral 1 to end,
    if already ends with a numeral increment by 1.

    Args:
        filepath (str): filepath to be checked

    Returns:
        str: [description]
    """

    filepath = str(filepath)
    file, ext = os.path.splitext(filepath)
    if os.path.isfile(filepath):
        if file[-1].isnumeric():
            file, num = splitnum(file)
            nname = file + str(int(num) + 1) + ext
            return overwrite_rename(nname)
        else:
            return overwrite_rename(file + spacer + "1" + ext)
    else:
        return Path(filepath)


def overwrite_rename_dir(dirpath, spacer="_"):
    """Given a filepath, check if file exists already. If so, add numeral 1 to end,
    if already ends with a numeral increment by 1.

    Args:
        filepath (str): filepath to be checked

    Returns:
        str: [description]
    """

    dirpath = Path(dirpath)
    if dirpath.is_dir():
        if not any(dirpath.iterdir()): # directory is empty
            return dirpath
        dirname = dirpath.stem
        if dirname[-1].isnumeric():
            dirname, num = splitnum(dirname)
            nname = dirname + str(int(num) + 1) + "/"
            return overwrite_rename_dir(dirpath.parents[0] / nname)
        else:
            return overwrite_rename_dir(dirpath.parents[0] / (dirname + spacer + "1/"))
    else:
        return dirpath

def splitnum(s):
    """split the trailing number off a string. Returns (stripped_string, number)"""
    head = s.rstrip("-.0123456789")
    tail = s[len(head) :]
    return head, tail


def induction_from_phase(phi, del_px):
    """Gives integrated induction in T*nm from a magnetic phase shift

    Args:
        phi (ndarray): 2D numpy array of size (dimy, dimx), magnetic component of the
            phase shift in radians
        del_px (float): in-plane scale of the image in nm/pixel

    Returns:
        tuple: (By, Bx) where each By, Bx is a 2D numpy array of size (dimy, dimx)
            corresponding to the y/x component of the magnetic induction integrated
            along the z-direction. Has units of T*nm (assuming del_px given in units
            of nm/pixel)
    """
    grad_y, grad_x = np.gradient(phi.squeeze(), edge_order=2)
    pre_B = scipy.constants.hbar / (scipy.constants.e * del_px) * 10**18  # T*nm^2
    Bx = pre_B * grad_y
    By = -1 * pre_B * grad_x
    return (By, Bx)

def get_SS(guess_phi, true_phi, scale_invar=False):
    """Get accuracy of phase reconstruction using structural similarity

    Args:
        guess_phi (ndarray): Guess phase shift
        true_phi (ndarray): Ground truth phase shift
        scale_invar (bool): Whether or not to ignore global scaling differences.
            Default False.

    Returns:
        float: Accuracy [-1,1]. 1 corresponds to perfect correlation, 0 to no correlation
            and -1 to anticorrelation.
    """
    if isinstance(guess_phi, torch.Tensor):
        guess_phi = guess_phi.cpu().detach().numpy()
    if isinstance(true_phi, torch.Tensor):
        true_phi = true_phi.cpu().detach().numpy()
    guess_phi = np.copy(guess_phi).astype("double")
    true_phi = np.copy(true_phi).astype("double")
    guess_phi -= np.min(guess_phi)
    true_phi -= np.min(true_phi)
    if scale_invar:
        if np.max(guess_phi) != 0:
            guess_phi /= np.max(guess_phi)
        if np.max(true_phi) != 0:
            true_phi /= np.max(true_phi)
    else:
        # scale = np.max(true_phi)
        # if scale != 0:
        #     guess_phi /= scale
        #     true_phi /= scale
        pass
    data_range = true_phi.ptp()
    return structural_similarity(guess_phi, true_phi, data_range=data_range)


def get_acc(guess_phi, true_phi):
    """Get accuracy of phase reconstruction, does not account for scaling differences
    as long as values are centered around 0

    Args:
        guess_phi (ndarray): Guess phase shift
        true_phi (ndarray): Ground truth phase shift

    Returns:
        float: Accuracy [-1,1]. 1 corresponds to perfect correlation, 0 to no correlation
            and -1 to anticorrelation.
    """
    if isinstance(guess_phi, torch.Tensor):
        guess_phi = guess_phi.cpu().detach().numpy()
    if isinstance(true_phi, torch.Tensor):
        true_phi = true_phi.cpu().detach().numpy()
    guess_phi = guess_phi.astype("double")
    true_phi = true_phi.astype("double")
    acc = (guess_phi * true_phi).sum() / np.sqrt(
        (guess_phi * guess_phi).sum() * (true_phi * true_phi).sum()
    )
    return acc

def get_all_accs(guess_phi, best_phi=None, true_phi=None, pdict=None, guess_amp=None, best_amp=None, true_amp=None):
    """
    returns: ((guess SS, guess SS norm), (best SS, best SS norm)), (guess acc, best acc)
    """
    if isinstance(guess_phi, torch.Tensor):
        guess_phi = guess_phi.cpu().detach().numpy()
    else:
        guess_phi = np.copy(guess_phi)
    if isinstance(best_phi, torch.Tensor):
        best_phi = best_phi.cpu().detach().numpy()
    elif best_phi is not None:
        best_phi = np.copy(best_phi)
    if isinstance(true_phi, torch.Tensor):
        true_phi = true_phi.cpu().detach().numpy()
    elif true_phi is not None:
        true_phi = np.copy(true_phi)
    if isinstance(guess_amp, torch.Tensor):
        guess_amp = guess_amp.cpu().detach().numpy()
    elif guess_amp is not None:
        guess_amp = np.copy(guess_amp)
    if isinstance(best_amp, torch.Tensor):
        best_amp = best_amp.cpu().detach().numpy()
    elif best_amp is not None:
        best_amp = np.copy(best_amp)
    if isinstance(true_amp, torch.Tensor):
        true_amp = true_amp.cpu().detach().numpy()
    elif true_amp is not None:
        true_amp = np.copy(true_amp)

    guess_phi -= guess_phi.mean()
    if best_phi is not None:
        best_phi -= best_phi.mean()
    if true_phi is not None:
        true_phi -= true_phi.mean()

    # if guess_amp is not None:
    #     guess_amp -= guess_amp.min()
    # if best_amp is not None:
    #     best_amp -= best_amp.min()
    # if true_amp is not None:
    #     true_amp -= true_amp.min()

    guess_By, guess_Bx = induction_from_phase(guess_phi, pdict['del_px'])
    true_By, true_Bx = induction_from_phase(true_phi, pdict['del_px'])

    guess_Bmag = np.sqrt(guess_Bx**2 + guess_By**2)
    true_Bmag = np.sqrt(true_Bx**2 + true_By**2)

    pdict['guess_phi_SS'] = get_SS(guess_phi, true_phi)
    pdict['guess_phi_SS_norm'] = get_SS(guess_phi, true_phi, scale_invar=True)
    pdict['guess_phi_acc'] = get_acc(guess_phi, true_phi)
    pdict['guess_Bx_SS'] = get_SS(guess_Bx, true_Bx)
    pdict['guess_Bx_SS_norm'] = get_SS(guess_Bx, true_Bx, scale_invar=True)
    pdict['guess_Bx_acc'] = get_acc(guess_Bx, true_Bx)
    pdict['guess_By_SS'] = get_SS(guess_By, true_By)
    pdict['guess_By_SS_norm'] = get_SS(guess_By, true_By, scale_invar=True)
    pdict['guess_By_acc'] = get_acc(guess_By, true_By)

    pdict['guess_Bave_SS'] = (pdict["guess_Bx_SS"] + pdict["guess_By_SS"]) / 2
    pdict['guess_Bave_SS_norm'] = (pdict["guess_Bx_SS_norm"] + pdict["guess_By_SS_norm"]) / 2
    pdict['guess_Bave_acc'] = (pdict["guess_Bx_acc"] + pdict["guess_By_acc"]) / 2

    pdict['guess_Bmag_SS'] = get_SS(guess_Bmag, true_Bmag)
    pdict['guess_Bmag_SS_norm'] = get_SS(guess_Bmag, true_Bmag, scale_invar=True)
    pdict['guess_Bmag_acc'] = get_acc(guess_Bmag, true_Bmag)
    pdict['guess_phi_FRC_res'] = get_FRC_res(guess_phi, true_phi, pdict['del_px'], cutoff=0.5)[2]
    pdict['guess_Bx_FRC_res'] = get_FRC_res(guess_Bx, true_Bx, pdict['del_px'], cutoff=0.5)[2]
    pdict['guess_By_FRC_res'] = get_FRC_res(guess_By, true_By, pdict['del_px'], cutoff=0.5)[2]
    pdict['guess_Bave_FRC_res'] = (pdict['guess_Bx_FRC_res'] + pdict['guess_By_FRC_res']) / 2
    pdict['guess_Bmag_FRC_res'] = get_FRC_res(guess_Bmag, true_Bmag, pdict['del_px'], cutoff=0.5)[2]


    if best_phi is not None:
        best_By, best_Bx = induction_from_phase(best_phi, pdict['del_px'])
        best_Bmag = np.sqrt(best_Bx**2 + best_By**2)
        pdict['best_phi_SS'] = get_SS(best_phi, true_phi)
        pdict['best_phi_SS_norm'] = get_SS(best_phi, true_phi, scale_invar=True)
        pdict['best_phi_acc'] = get_acc(best_phi, true_phi)
        pdict['best_Bx_SS'] = get_SS(best_Bx, true_Bx)
        pdict['best_Bx_SS_norm'] = get_SS(best_Bx, true_Bx, scale_invar=True)
        pdict['best_Bx_acc'] = get_acc(best_Bx, true_Bx)
        pdict['best_By_SS'] = get_SS(best_By, true_By)
        pdict['best_By_SS_norm'] = get_SS(best_By, true_By, scale_invar=True)
        pdict['best_By_acc'] = get_acc(best_By, true_By)
        pdict['best_Bmag_SS'] = get_SS(best_Bmag, true_Bmag)
        pdict['best_Bmag_SS_norm'] = get_SS(best_Bmag, true_Bmag, scale_invar=True)
        pdict['best_Bmag_acc'] = get_acc(best_Bmag, true_Bmag)
        pdict['best_phi_FRC_res'] = get_FRC_res(best_phi, true_phi, pdict['del_px'], cutoff=0.5)[2]
        pdict['best_Bx_FRC_res'] = get_FRC_res(best_Bx, true_Bx, pdict['del_px'], cutoff=0.5)[2]
        pdict['best_By_FRC_res'] = get_FRC_res(best_By, true_By, pdict['del_px'], cutoff=0.5)[2]
        pdict['best_Bmag_FRC_res'] = get_FRC_res(best_Bmag, true_Bmag, pdict['del_px'], cutoff=0.5)[2]

        pdict['best_Bave_SS'] = (pdict["best_Bx_SS"] + pdict["best_By_SS"]) / 2
        pdict['best_Bave_SS_norm'] = (pdict["best_Bx_SS_norm"] + pdict["best_By_SS_norm"]) / 2
        pdict['best_Bave_acc'] = (pdict["best_Bx_acc"] + pdict["best_By_acc"]) / 2

    if pdict['solve_amp'] and guess_amp is not None:
        pdict['guess_amp_SS'] = get_SS(guess_amp, true_amp)
        pdict['guess_amp_SS_norm'] = get_SS(guess_amp, true_amp, scale_invar=True)
        pdict['guess_amp_acc'] = get_acc(guess_amp, true_amp)
        if best_amp is not None:
            pdict['best_amp_SS'] = get_SS(best_amp, true_amp)
            pdict['best_amp_SS_norm'] = get_SS(best_amp, true_amp, scale_invar=True)
            pdict['best_amp_acc'] = get_acc(best_amp, true_amp)
    return

def add_gaussian_noise(image, percent):
    """
    percent noise is float, percent=10 means 10% noise
    """
    if isinstance(image, torch.Tensor):
        sigma = percent / 200 * torch.mean(image)
        noise = torch.normal(mean=0, std=sigma, size=image.shape, device=image.device)
    else:
        sigma = percent / 200 * np.mean(image)
        noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    return image + noise


def Tukey2D(shape, alpha=0.5, sym=True):
    """
    makes a 2D (rectangular not round) window based on a Tukey signal
    Useful for windowing images before taking FFTs
    """
    dimy, dimx = shape
    ty = tukey(dimy, alpha=alpha, sym=sym)
    filt_y = np.tile(ty.reshape(dimy, 1), (1, dimx))
    tx = tukey(dimx, alpha=alpha, sym=sym)
    filt_x = np.tile(tx, (dimy, 1))
    output = filt_x * filt_y
    return output


def radial_profile(data, center=None, width=1):
    """originally from stackoverflow, modified to allow changing of bins.
    really clever."""
    y, x = np.indices((data.shape))
    if center is None:
        center = np.array(np.shape(data)) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    # because we conver to int, this will give us the proper width of spacing
    r /= width
    # very good way of getting a radius map, better than linspace
    r = np.round(r).astype(np.int)

    # weights the radius map by the values of the data
    tbin = np.bincount(r.ravel(), data.ravel())
    # the weights due to the number of pixels for each radius
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr  # normalizing
    return radialprofile


def FSC(im1, im2, width=2, tukeyAlpha=None):
    """
    Fourier shell correlation of two images
    width = bin width in radial sum
    tukeyAlpha is alpha for tukey window to be applied. alpha=1 works... kinda
    """
    im1 = norm_image(im1)
    im2 = norm_image(im2)

    if tukeyAlpha is not None:
        win = Tukey2D(im1.shape, alpha=tukeyAlpha)
        im1 *= win
        im2 *= win

    if ( np.shape(im1) != np.shape(im2) ) :
        print('input images must have the same dimensions')
    if ( np.shape(im1)[0] != np.shape(im1)[1]) :
        print('input images must be squares')
    fft1 = np.fft.fftshift(np.fft.fft2(im1))
    fft2 = np.fft.fftshift(np.fft.fft2(im2))

    C  = radial_profile(np.real(fft1*np.conj(fft2)), width=width)
    C1 = radial_profile(np.abs(fft1)**2, width=width)
    C2 = radial_profile(np.abs(fft2)**2, width=width)
    denom = np.sqrt(C1 * C2)
    bads = np.where(denom==0)
    denom[bads] = 1
    FSC = C/denom
    FSC[bads] = 0
    FSC = FSC[:-1]
    ny, nx = im2.shape
    bins = np.arange(0, np.sqrt((nx//2)**2 + (ny//2)**2), width) + width/2
    return FSC, bins


def compute_frc_old(
        im1: np.ndarray,
        im2: np.ndarray,
        bin_width: int = 2.0,
        tukey = False,
        tukey_alpha = 1/2,
):
    """ Computes the Fourier Ring/Shell Correlation of two 2-D images
    https://tttrlib.readthedocs.io/en/latest/auto_examples/imaging/plot_imaging_frc.html
    :param image_1:
    :param image_2:
    :param bin_width:
    :return:
    returns bins in 1/pixels, to get to nm:
    size_nm = 1/(rad_px / del_px / max(dim_y, dim_x))
    """
    assert np.shape(im1) == np.shape(im2)
    image_1 = im1 - np.min(im1)
    image_2 = im2 - np.min(im2)
    image_1 = image_1 / np.sum(image_1)
    image_2 = image_2 / np.sum(image_2)

    if tukey:
        win = Tukey2D(image_1.shape, alpha=tukey_alpha)
        image_1 *= win
        image_2 *= win

    f1, f2 = np.fft.fft2(image_1), np.fft.fft2(image_2)
    af1f2 = np.real(f1 * np.conj(f2))
    af1_2, af2_2 = np.abs(f1)**2, np.abs(f2)**2
    nx, ny = af1f2.shape
    x = np.arange(-np.floor(nx / 2.0), np.ceil(nx / 2.0))
    y = np.arange(-np.floor(ny / 2.0), np.ceil(ny / 2.0))
    distances = list()
    wf1f2 = list()
    wf1 = list()
    wf2 = list()
    for xi, yi in np.array(np.meshgrid(x,y)).T.reshape(-1, 2):
        distances.append(np.sqrt(xi**2 + xi**2))
        xi = int(xi)
        yi = int(yi)
        wf1f2.append(af1f2[xi, yi])
        wf1.append(af1_2[xi, yi])
        wf2.append(af2_2[xi, yi])

    bins = np.arange(0, np.sqrt((nx//2)**2 + (ny//2)**2), bin_width)
    f1f2_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1f2
    )
    f12_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1
    )
    f22_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf2
    )


    # f1f2_r = radial_profile(af1f2, width=bin_width)[::-1][2:-1]
    # f12_r = radial_profile(af1_2, width=bin_width)[::-1][2:-1]
    # f22_r = radial_profile(af2_2, width=bin_width)[::-1][2:-1]
    # print(f1f2_r.shape)

    denom = np.sqrt(f12_r * f22_r)

    bads = np.where(denom==0)
    denom[bads] = 1
    density = f1f2_r / denom
    density[bads] = 0

    bins = bin_edges[:-1] + bin_width/2
    return density, bins

def get_cintercept_FRC(frc, bins,  cutoff=0.5):
    # bins in pixels
    cpoints = np.argwhere(frc <= cutoff)
    if np.size(cpoints) > 0:
        un_ind = cpoints[0][0]
    else:
        return bins[-1]
    un_val = frc[un_ind]
    if un_val == cutoff:
        xval = bins[un_ind]
    else:
        ov_val = frc[un_ind-1]

        frac = (ov_val - cutoff) / abs(un_val - ov_val)
        dx = bins[un_ind] - bins[un_ind-1]
        xval = bins[un_ind - 1] + frac * dx

    return xval

def get_FRC_res(im1, im2, del_px, bin_width=2, tukeyAlpha=None, cutoff=0.5):
    if isinstance(im1, torch.Tensor):
        im1 = im1.detach().cpu().numpy()
    if isinstance(im2, torch.Tensor):
        im2 = im2.detach().cpu().numpy()

    frc, frc_bins = FSC(im1, im2, width=bin_width, tukeyAlpha=tukeyAlpha)

    crad1 = 300
    crad2 = 7 # nm
    bins_nm = 1/(frc_bins / del_px / max(im1.shape)) # from 1/pix to 1/nm

    if crad2 is not None:
        cind2 = np.argmax(bins_nm < crad2)
        if cind2 > 0:
            frc_bins = frc_bins[:cind2]
            frc = frc[:cind2]
    if crad1 is not None:
        cind1 = np.argmax(bins_nm < crad1)
        if cind1 > 0:
            frc_bins = frc_bins[cind1:]
            frc = frc[cind1:]

    frc_intercept = get_cintercept_FRC(frc, frc_bins, cutoff)
    assert np.shape(im1) == np.shape(im2)
    maxsize = np.max(im1.shape)
    res = 1 / (frc_intercept / del_px / maxsize) # 1/pix -> nm
    return frc, frc_bins, res





def rot_ang_to_vect(Tx=0, Ty=0, Tz=0, v=[0, 0, 1]):
    """rotates the input vector around x, y, then z.
    input in degrees, rotations around x then y then z axes.
    vector is [x,y,z]
    """
    vx = np.array([1, 0, 0])
    vy = np.array([0, 1, 0])
    vz = np.array([0, 0, 1])

    rx = R.from_rotvec(Tx * vx, degrees=True)
    ry = R.from_rotvec(Ty * vy, degrees=True)
    rz = R.from_rotvec(Tz * vz, degrees=True)
    rtot = rz * ry * rx  # apply x then y then z
    vout = rtot.apply(v)
    return vout


def pad_mags(mags, pad):
    mags = np.array(mags)
    assert np.ndim(mags) == 4
    assert mags.shape[0] == 3
    _, dz, dy, dx = mags.shape

    if isinstance(pad, float):
        padx = round(dx * pad)
        pady = round(dy * pad)
    elif isinstance(pad, int):
        padx = pad
        pady = pad
    elif isinstance(pad, tuple):
        pady, padx = pad

    return np.pad(mags, ((0,0), (0,0), (pady, pady), (padx,padx)), mode='wrap'), (pady, padx)

def pad_im(im, pad):
    im = np.array(im)
    dx = im.shape[-1]
    dy = im.shape[-2]

    if isinstance(pad, float):
        padx = round(dx * pad)
        pady = round(dy * pad)
    elif isinstance(pad, int):
        padx = pad
        pady = pad
    elif isinstance(pad, tuple):
        pady, padx = pad

    return np.pad(im, ((pady, pady), (padx,padx)), mode='wrap'), (pady, padx)


def unpad_im(mags, pad):
    assert np.ndim(mags) >= 2

    if isinstance(pad, int):
        padx = pad
        pady = pad
    elif isinstance(pad, tuple):
        pady, padx = pad

    if padx == 0 and pady == 0:
        return mags
    elif padx == 0:
        return mags[..., pady:-pady, :]
    elif pady == 0:
        return mags[..., padx:-padx]
    else:
        return mags[..., pady:-pady, padx:-padx]


def norm_image(image):
    """Normalize image intensities to between 0 and 1"""
    image = image - np.min(image)
    image = image / np.max(image)
    return image

def sym_image(image):
    """Return 4-fold symmetrized image"""
    dy, dx = image.shape
    return np.pad(image, ((0, dy), (0, dx)), mode='symmetric')

def unsym_image(image):
    dy, dx = image.shape
    return image[:dy//2, :dx//2]
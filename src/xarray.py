""" This submodule contains functions designed to operate o
xarray.object objects.
"""
import xarray as xa
import pymaster as nmt
import numpy as np
import tensorflow as tf

__all__ = ['apply_nmt_flat', 'apply_l2min', 'apply_per_image_standardization', 'resample', 'resample_iterations', 'make_flat_bins', 'make_square_mask']


def _setup_nmt_ufunc_hpix(mask, wsp00=None, wsp02=None, wsp22=None):
    """ This function is a wrapper that supplies a function to take in 
    HEALPix maps, and return a set of power spectra. This is designed
    for use with vectorize routines, when maps exist in a large
    multidimensional arrays.

    Parameters
    ----------
    mask: ndarray
        Array containing an apodized mask, in HEALPix ring-ordering.
    wsp00, wsp02, wsp22: pymaster.NmtWorkspace
        Namaster workspace objects containing pre-computed mode coupling
        matrices. These correspond to various correlations between
        spin-0 and spin-2 maps.

    Returns
    -------
    function
        Function to calculate power spectra of HEALPix maps. 
    """
    def wrapper(arr):
        if arr.shape[0] == 1:
            f0 = nmt.NmtField(mask, arr)
            return wsp00.decouple_cell(nmt.compute_coupled_cell(f0, f0))
        elif arr.shape[0] == 2:
            f2 = nmt.NmtField(mask, arr, purify_b=True, purify_e=True)
            return wsp22.decouple_cell(nmt.compute_coupled_cell(f2, f2))
        elif arr.shape[0] == 3:
            f0 = nmt.NmtField(mask, [arr[0]])
            f2 = nmt.NmtField(mask, arr[1:], purify_b=True, purify_e=True)
            cl00 = wsp00.decouple_cell(nmt.compute_coupled_cell(f0, f0))
            cl02 = wsp02.decouple_cell(nmt.compute_coupled_cell(f0, f2))
            cl22 = wsp22.decouple_cell(nmt.compute_coupled_cell(f2, f2))
            return np.concatenate((cl00, cl02, cl22))
    return wrapper

def apply_nmt_flat(data, dims=['pol', 'x', 'y']):
    """ Function to estimate angular power spectra of flat-sky maps stored
    `xarray` `DataArray`s.

    Parameters
    ----------
    data: xarray.DataArray
        DataArray, must contain dimensions corresponding to those supplied in the
        `dims` argument.
    nx: int
        Pixel dimension
    ang: float
        Angular size of the patch.
    binning: `pymaster.NmtBinFlat`
        Instance of `pymaster.NmtBinFlat` containing the binning scheme to be used. 
    wsp00, wsp02, wsp22: `pymaster.NmtWorkspaceFlat`
        Instances of `pymaster.NmtWorkspaceFlat` objects, depending on the
        combination of spin-0 and spin-2 fields being analyzed.
    dims: list(str)
        List of dimension names containing the data to be analyzed. These should
        be the names of the coordinates of the xarray that containg the Stokes'
        parameter, and then the pixel coordinates.

    Returns
    -------
    `xarray.DataArray`
    """    
    # Make square apodized mask
    mask = make_square_mask(data.res_x, data.ang_y)
    # Get binning scheme
    nmtbin = get_predefined_binning()
    # setup the ufunc to be applied across the relevant xarray dimensions.
    wsp00 = None
    wsp02 = None
    wsp22 = None
    fields = []
    if 't' in data.pol:
        f0 = nmt.NmtFieldFlat(data.ang_y, data.ang_y, mask, np.random.randn(1, data.res_x, data.res_y))
        wsp00 = nmt.NmtWorkspaceFlat()
        wsp00.compute_coupling_matrix(f0, f0, nmtbin)
        fields.append('tt')
    if all(x in data.pol for x in ['q', 'u']):
        f2 = nmt.NmtFieldFlat(data.ang_x, data.ang_y, mask, np.random.randn(2, data.res_x, data.res_y), purify_e=True, purify_b=True)
        wsp22 = nmt.NmtWorkspaceFlat()
        wsp22.compute_coupling_matrix(f2, f2, nmtbin)
        fields.append('ee')
        fields.append('bb')
    if all(x in data.pol for x in ['t', 'q', 'u']):
        f0 = nmt.NmtFieldFlat(data.ang_x, data.ang_y, mask, np.random.randn(1, data.res_x, data.res_y))
        f2 = nmt.NmtFieldFlat(data.ang_x, data.ang_y, mask, np.random.randn(2, data.res_x, data.res_y), purify_e=True, purify_b=True)
        wsp02 = nmt.NmtWorkspaceFlat()
        wsp02.compute_coupling_matrix(f0, f2, nmtbin)
        fields.append('te')

    ufunc_spec = _setup_nmt_ufunc_flat(mask, data.ang_x, nmtbin, wsp00=wsp00, wsp02=wsp02, wsp22=wsp22)
    # apply the ufunc arcoss the specified dimensions. These dimensions are then removed
    # by the `exclude_dims` argument, and additional dimensions added are specified in the
    # `output_core_dims` argument.
    spectra = xa.apply_ufunc(
        ufunc_spec, 
        data, 
        input_core_dims=[dims],
        output_core_dims=[["field", "bandpowers"]],
        exclude_dims = set(dims),
        vectorize=True
    )
    spectra["field"] = fields
    spectra["bandpowers"] = nmtbin.get_effective_ells()
    return spectra

def _setup_nmt_ufunc_flat(mask, ang, binning, wsp00=None, wsp02=None, wsp22=None):
    """ This function is a wrapper that supplies a function to take in 
    2-D maps, and return a set of power spectra. This is designed
    for use with vectorize routines, when maps exist in a large
    multidimensional arrays.

    Parameters
    ----------
    mask: ndarray
        Array containing a 2-D apodized mask.
    wsp00, wsp02, wsp22: pymaster.NmtWorkspaceFlat
        Namaster workspace objects containing pre-computed mode coupling
        matrices. These correspond to various correlations between
        spin-0 and spin-2 maps.

    Returns
    -------
    function
        Function to calculate power spectra of 2-D maps. 
    """
    def wrapper(arr):
        if arr.shape[0] == 1:
            f0 = nmt.NmtFieldFlat(ang, ang, mask, arr)
            return wsp00.decouple_cell(nmt.compute_coupled_cell_flat(f0, f0, binning))
        elif arr.shape[0] == 2:
            f2 = nmt.NmtFieldFlat(ang, ang, mask, arr, purify_b=True, purify_e=True)
            cl22 = wsp22.decouple_cell(nmt.compute_coupled_cell_flat(f2, f2, binning))
            return np.concatenate((cl22[[0]], cl22[[3]]))
        elif arr.shape[0] == 3:
            f0 = nmt.NmtFieldFlat(ang, ang, mask, [arr[0]])
            f2 = nmt.NmtFieldFlat(ang, ang, mask, arr[1:], purify_b=True, purify_e=True)
            cl00 = wsp00.decouple_cell(nmt.compute_coupled_cell_flat(f0, f0, binning))
            cl02 = wsp02.decouple_cell(nmt.compute_coupled_cell_flat(f0, f2, binning))
            cl22 = wsp22.decouple_cell(nmt.compute_coupled_cell_flat(f2, f2, binning))
            return np.concatenate((cl00, cl22[[0, 3]], cl02[[0]]))
    return wrapper


def make_square_mask(nx, ang, margin=0.05, aposize=2, apotype="C1"):
    """ Function to generate an apodized square mask. 

    Parameters
    ----------
    nx: int
        The number of pixels in the x and y dimensions.
    ang: float
        Angular size of the patch in radians.
    margin: float
        Fraction of the map to set to zero at each edge, to allow tapering.
    aposize: float
        Characteristic length scale of the taper in degrees.
    apotype: string
        Type of apodization to apply (see `pymaster.mask_apodization_flat` for
        options.)

    Returns
    -------
    ndarray
        Array containing the apodized mask.
    """
    ny = nx
    margin *= nx
    mask = np.ones((nx, ny))
    xarr = np.ones(nx)[:, None] * np.arange(nx)[None, :]
    yarr = np.ones(ny)[None, :] * np.arange(ny)[:, None]
    
    mask[(xarr < margin) | (xarr > nx - margin)] = 0
    mask[(yarr < margin) | (yarr > ny - margin)] = 0
    apo_mask = nmt.mask_apodization_flat(mask, ang, ang, aposize, apotype)
    return apo_mask


def make_flat_bins(ang, nx, r, **kwargs):
    """ Function to set up the power spectrum binning scheme for 
    analysis of flat maps.

    Parameters
    ----------
    ang: float
        Angular size of patch in radians.
    nx: int
        Number of pixels in each dimension.
    r: int
        Parameter controlling the spacing of bins.

    Returns
    -------
    pymaster.NmtBinFlat
        Instance of the `pymaster.NmtBinFlat` object.
    """
    l0_bins = np.arange(nx / r) * r * np.pi/ang
    lf_bins = (np.arange(nx / r) + 1) * r * np.pi/ang
    b = nmt.NmtBinFlat(l0_bins, lf_bins, **kwargs)
    return b

def get_predefined_bins():
    ell_ini = [100, 150, 200, 250, 300, 400, 550, 700, 850, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    ell_end = [149, 199, 249, 299, 399, 549, 699, 849, 999, 1099, 1199, 1299, 1399, 1499, 1599, 1699]
    return ell_ini, ell_end

def get_predefined_binning():
    ell_ini, ell_end = get_predefined_bins()
    return nmt.NmtBinFlat(ell_ini, ell_end)

def get_predefined_bin_labels():
    ell_ini, ell_end = get_predefined_bins()
    return [f"[{ell_i:d}, {ell_f + 1:d})" for ell_i, ell_f in zip(ell_ini, ell_end)]

def _get_l2_func(model, image):
    """ Wrapper to generate a function to compute the l2 distance
    between the decoded image of a given point in latent space, `x`
    and a given `image`.

    Parameters
    ----------
    model: tensorflow.Model
        Instance of a tensorflow model. Output shape must match `image`.
    image: ndarray
        Array containing the target image.
    """
    def l2(x):
        # set up Gradient Tape, as we will return both the l2 distance
        # and the gradient with respect to the input.
        with tf.GradientTape() as tape:
            tape.watch(x)
            generated_image = model.decode(x)[0, :, :, 0]
            l2_loss = tf.linalg.norm(generated_image - image)
        jac = tape.gradient(l2_loss, x)
        loss_value = tf.reshape(l2_loss, [1])
        return loss_value, jac
    return l2

def _l2_minimization_ufunc(model, initial_position):
    """ 
    """
    def l2min(target):
        l2_func = _get_l2_func(model, target)
        opt = tfp.optimizer.lbfgs_minimize(l2_func, initial_position=initial_position, tolerance=1e-05, max_iterations=2000)
        pred = model.decode(opt.position)[0, :, :, :]
        return np.moveaxis(np.array(pred), -1, 0)
    return l2min


def apply_l2min(data, model, initial_position, dims=['pol', 'x', 'y'], output_core_dims=['pol', 'x', 'y']):
    l2_min_ufunc = _l2_minimization_ufunc(model, initial_position)
    return xa.apply_ufunc(
        l2_min_ufunc, 
        data,
        input_core_dims=[dims],
        output_core_dims=[output_core_dims],
        exclude_dims = set(dims),
        vectorize=True
        )


def _per_image_standardization_ufunc(arr):
    """ Convenience wrapper for tensorflow's `per_image_standardization`
    function, to allow it to be applied to a 2D image.
    """  
    return tf.image.per_image_standardization(arr[..., None])[:, :, 0]


def apply_per_image_standardization(data, dims=['x', 'y']):
    """ Function to apply tensorflow's `per_image_standardization`
    across an `xarray.DataArray`.

    Parameters
    ----------
    data: `xarray.DataArray`
        Instance of an xarray DataArray containing at least the dimensions
        referenced in `dims`.
    dims: list(str) (optional, default=['x', 'y']) 
        List containing the names of dimensions containing the image
        data to be standardized.
    
    Returns
    -------
    xarray.DataArray
        DataArray containing the standardized data.
    """
    return xa.apply_ufunc(
    _per_image_standardization_ufunc, 
    data,
    input_core_dims=[dims],
    output_core_dims=[dims],
    exclude_dims = set(dims),
    vectorize=True
)

def resample(hind, resample_dim):
    """Resample with replacement in dimension ``resample_dim``.

    Args:
        hind (xr.object): input xr.object to be resampled.
        resample_dim (str): dimension to resample along.

    Returns:
        xr.object: resampled along ``resample_dim``.

    """
    to_be_resampled = hind[resample_dim].values
    smp = np.random.choice(to_be_resampled, len(to_be_resampled))
    smp_hind = hind.sel({resample_dim: smp})
    # ignore because then inits should keep their labels
    if resample_dim != 'init':
        smp_hind[resample_dim] = hind[resample_dim].values
    return smp_hind

def resample_iterations(init, iterations, dim='member', dim_max=None, replace=True):
    """Resample over ``dim`` by index ``iterations`` times.

    .. note::
        This gives the same result as `_resample_iterations_idx`. When using dask, the
        number of tasks in `_resample_iterations` will scale with iterations but
        constant chunksize, whereas the tasks in `_resample_iterations_idx` will stay
        constant with increasing chunksize.

    Args:
        init (xa.DataArray, xa.Dataset): Initialized prediction ensemble.
        iterations (int): Number of bootstrapping iterations.
        dim (str): Dimension name to bootstrap over. Defaults to ``'member'``.
        dim_max (int): Number of items to select in `dim`.
        replace (bool): Bootstrapping with or without replacement. Defaults to ``True``.

    Returns:
        xa.DataArray, xa.Dataset: Bootstrapped data with additional dim ```iteration```

    """
    if dim_max is not None and dim_max <= init[dim].size:
        # select only dim_max items
        select_dim_items = dim_max
        new_dim = init[dim].isel({dim: slice(None, dim_max)})
    else:
        select_dim_items = init[dim].size
        new_dim = init[dim]

    if replace:
        idx = np.random.randint(0, init[dim].size, (iterations, select_dim_items))
    elif not replace:
        # create 2d np.arange()
        idx = np.linspace(
            (np.arange(select_dim_items)),
            (np.arange(select_dim_items)),
            iterations,
            dtype='int',
        )
        # shuffle each line
        for ndx in np.arange(iterations):
            np.random.shuffle(idx[ndx])
    idx_da = xa.DataArray(
        idx,
        dims=('iteration', dim),
        coords=({'iteration': range(iterations), dim: new_dim}),
    )
    init_smp = []
    for i in np.arange(iterations):
        idx = idx_da.sel(iteration=i).data
        init_smp2 = init.isel({dim: idx}).assign_coords({dim: new_dim})
        init_smp.append(init_smp2)
    init_smp = xa.concat(init_smp, dim='iteration')
    init_smp['iteration'] = np.arange(1, 1 + iterations)
    return init_smp
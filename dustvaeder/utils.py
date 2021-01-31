import numpy as np
import healpy as hp
import astropy.units as u
import tensorflow as tf

class FlatCutter(object):
    """ Object to control the extraction of flat patches from a given HEALPix 
    map.

    Object is initialized with parameters defining the geometry of the patch:
    its length in degrees, and the number of pixels in each direction. 
    The `rotate_and_interpolate` method defines a grid centered at (0, 0)
    of dimensions corresponding to `xlen`, `ylen`, and rotates it to the 
    point (lon, lat). The value of the map at the resulting grid of longitudes 
    and latitudes is then determined by interpolation. 
    """
    @u.quantity_input
    def __init__(self, ang_x: u.deg, ang_y: u.deg, xres, yres):
        assert type(xres) is int
        assert type(yres) is int
        self.xres = xres
        self.yres = yres

        self.ang_x = ang_x
        self.ang_y = ang_y
        
        # get grid of unit vectors corresponding to flat patch around
        # pole (z = 1). For this we use ang_x in radians, as is appropriate
        # for the implicit small-angle approximation 
        self.xarr = np.linspace(- self.ang_x.to(u.rad).value / 2., 
                                self.ang_x.to(u.rad).value / 2., xres)
        self.yarr = np.linspace(- self.ang_y.to(u.rad).value / 2., 
                                self.ang_y.to(u.rad).value / 2., yres)

        xgrid, ygrid = np.meshgrid(self.xarr, self.yarr)
        xgrid = xgrid.ravel()[None, :]
        ygrid = ygrid.ravel()[None, :]
        zgrid = np.ones_like(ygrid)
        
        # vectors corresponding to cartesian grid around poll
        self.vecs = np.concatenate((xgrid, ygrid, zgrid)).T

        # get the latitude (*not colatitude*) and longitude in degrees
        # of the cartesian grid points around the pole. 
        self.lons, self.lats = hp.vec2ang(self.vecs, lonlat=True)
        self.lats *= u.deg
        self.lons *= u.deg
        return
    
    @u.quantity_input
    def rotate_to_pole_and_interpolate(self, lon: u.deg, lat: u.deg, ma):
        """ Method to rotate the grid at (0, 0) to `rot=(lon, lat)`, and sample
        the map at the grid points by interpolation.

        Parameters
        ----------
        lat, lon: float
            Latitude (*not* colatitude) and longitude of point to be rotated
            to the North pole, in degrees.
        ma: ndarray
            Healpix map from which the interpolation is to be made.
        """
        if hp.pixelfunc.maptype(ma) == 0:  # a single map is converted to a list
            ma = [ma]
        # define a rotation object in terms of the theta_rot and phi_rot angles.
        # This returns a rotator object that can be applied to rotate a given
        # vector by this angle. Since we are interested in rotating some patch
        # to the pole, we actually want to apply the *inverse* rotation operator
        # to the vectors self.co_lats, self.lons.
        lon = lon.to(u.deg)
        lat = lat.to(u.deg)
        rotator = hp.Rotator(rot=[lon.value, lat.value - 90.], deg=True)
        self.inv_lon_grid, self.inv_lat_grid = rotator.I(self.lons.to(u.deg).value, self.lats.to(u.deg).value, lonlat=True)
        # Interpolate the original map to the pixels centers in the new ref frame
        m_rot = [hp.get_interp_val(each, self.inv_lon_grid, self.inv_lat_grid, lonlat=True) for each in ma]

        # Rotate polarization
        if len(m_rot) > 1:
            # Create a complex map from QU  and apply the rotation in psi due to the rotation
            # Slice from the end of the array so that it works both for QU and IQU
            m_rot[-2], m_rot[-1] = spin2rot(m_rot[-2], m_rot[-1], rotator.angle_ref(self.inv_lon_grid, self.inv_lat_grid, lonlat=True))
            m_rot[-2], m_rot[-1] = spin2rot(m_rot[-2], m_rot[-1], self.lons.to(u.rad).value)
        else:
            m_rot = m_rot[0]
        return np.moveaxis(np.array(m_rot).reshape(-1, self.xres, self.yres), 0, -1)

def spin2rot(q, u, phi):
    """ Function to apply rotation by an angle `phi` to the
    spin-2 field defined by `q` and `u`.

    This function calculates and returns:

    P = \exp(2i\phi)(Q + iU)

    Parameters
    ----------
    q, u: ndarray
        Real and imaginary parts of spin-2 field.
    phi: ndarray
        Angle by which to rotate the spin-2 field.

    Returns
    -------
    ndarray
        Rotated spin-2 field.
    """
    p = np.empty(q.shape, dtype=complex)
    p.real = q
    p.imag = u
    p *= np.exp(2j * phi)
    return p.real, p.imag

@u.quantity_input
def get_patch_centers(gal_cut: u.deg, step_size: u.deg):
    """ Function to get the centers of the various patches to be cut out.

    Parameters
    ----------
    gal_cut: float
        We will miss out the region +/- `gal_cut` in Galactic latitude, measured
        in degrees.
    step_size: float
        Stepping distance in Galactic longitude, measured in degrees, between 
        patches.

    Returns
    -------
    list(tuple(float))
        List of two-element tuples containing the longitude and latitude.
    """
    gal_cut = gal_cut.to(u.deg)
    step_size = step_size.to(u.deg)
    assert gal_cut.unit == u.deg
    assert step_size.unit == u.deg
    southern_lat_range = np.arange(-90, (-gal_cut-step_size).value, step_size.value) * u.deg
    northern_lat_range = np.arange((gal_cut + step_size).value, 90, step_size.value) * u.deg
    lat_range = np.concatenate((southern_lat_range, northern_lat_range))

    centers = []
    for t in lat_range:
        step = step_size.value / np.cos(t.to(u.rad).value)
        for i in np.arange(0, 360, step):
            centers.append((i * u.deg, t))
    return centers

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Parameters
    ----------
    x: Image
        Image to be rotated.
    
    Returns
    -------
    Image
        Rotated image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

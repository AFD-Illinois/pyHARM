
import numpy as np
# Need extra broadcasting currently, or this would be scipy.linalg
import numpy.linalg as la
import scipy.optimize as opt

default_met_params = {'a': 0.9375, 'hslope': 0.3, 'r_out': 50.0, 'n1tot': 192,
                      'poly_xt': 0.82, 'poly_alpha': 14.0, 'mks_smooth': 0.5}

class CoordinateSystem(object):
    """ Base class for representing coordinate systems
    Coordinate classes define, as functions of their native coordinates X:
     * r,th,phi in spherical Kerr-Schild coordinates (or just spherical coordinates in the case of Minkowski)
     * A transformation matrix, dxdX, for tensors from one system to the other
     * The forms and determinant of the metric, gcov, gcon, & gdet

     Of these, most are derivable
    """

    def native_startx(cls, met_params):
        raise NotImplementedError

    def native_stopx(cls, met_params):
        raise NotImplementedError

    def r(self, x):
        raise NotImplementedError

    def th(self, x):
        raise NotImplementedError

    def phi(self, x):
        raise NotImplementedError

    def cart_x(self, x):
        raise NotImplementedError

    def cart_y(self, x):
        raise NotImplementedError

    def cart_z(self, x):
        raise NotImplementedError

    def ks_coord(self, x):
        """Return Spherical Kerr-Schild coordinates of a point X in native coordinates"""
        return np.array([self.r(x), self.th(x), self.phi(x)])

    def cart_coord(self, x):
        """Return Cartesian Kerr-Schild coordinates of a point X in native coordinates"""
        return np.array([self.cart_x(x), self.cart_y(x), self.cart_z(x)])

    def correct_small_th(self, theta):
        # Corrections for small theta
        if isinstance(theta, np.ndarray):
            # Perform statically to save lines (/time?)
            # TODO can do with np.clip...
            theta[np.where(np.logical_and(np.abs(theta) < self.small_th, theta >= 0))] = self.small_th
            theta[np.where(np.logical_and(np.abs(theta) < self.small_th, theta < 0))] = -self.small_th

            theta[np.where(np.logical_and(np.abs(np.pi - theta) < self.small_th, theta >= np.pi))] = np.pi + self.small_th
            theta[np.where(np.logical_and(np.abs(np.pi - theta) < self.small_th, theta < np.pi))] = np.pi - self.small_th
        else:
            if np.abs(theta) < self.small_th:
                if theta >= 0:
                    theta = self.small_th
                if theta < 0:
                    theta = -self.small_th

            if np.abs(np.pi - theta) < self.small_th:
                if theta >= np.pi:
                    theta = np.pi + self.small_th
                if theta < np.pi:
                    theta = np.pi - self.small_th
        return theta

    def gcov_ks(self, X):
        gcov_ks = np.zeros([4, 4, *(X.shape[1:])])
        r, th, _ = self.ks_coord(X)

        # TODO Un-C this.
        cth = np.cos(th)
        sth = np.sin(th)

        s2 = sth ** 2
        rho2 = r ** 2 + self.a ** 2 * cth ** 2

        gcov_ks[0, 0] = -1 + 2 * r / rho2
        gcov_ks[0, 1] = 2 * r / rho2
        gcov_ks[0, 3] = -2 * self.a * r * s2 / rho2

        gcov_ks[1, 0] = gcov_ks[0, 1]
        gcov_ks[1, 1] = 1. + 2. * r / rho2
        gcov_ks[1, 3] = -self.a * s2 * (1. + 2. * r / rho2)

        gcov_ks[2, 2] = rho2

        gcov_ks[3, 0] = gcov_ks[0, 3]
        gcov_ks[3, 1] = gcov_ks[1, 3]
        gcov_ks[3, 3] = s2 * (rho2 + self.a ** 2 * s2 * (1. + 2. * r / rho2))

        return gcov_ks

    def gcov(self, X):
        # Apply coordinate transformation to code coordinates X
        gcov_ks = self.gcov_ks(X)
        dxdX = self.dxdX(X)
        return np.einsum("ab...,ac...,bd...->cd...", gcov_ks, dxdX, dxdX)

    def gcon(self, gcov):
        """Return contravariant form of the metric, given the covariant form.
        As with all coordinate functions, the matrix/vector indices are *first*.  Specifically, gcon_func expects
        exactly zero grid indices (i.e. a single 4x4 matrix) or two grid indices (i.e. 4x4xN1xN2).
        """
        # TODO option to take coordinates and auto-call gcon?
        # TODO support 1 & 3 dimensional arrays of input matrices, probably with einsum
        # TODO add option to return gdet for speed
        if len(gcov.shape) == 2:
            return la.inv(gcov)
        elif len(gcov.shape) == 3:
            # Canon ordering is mu,nu,i,j for what I swear are good reasons
            gcov_t = gcov.transpose((2, 0, 1))
            return la.inv(gcov_t).transpose((2, 0, 1))
        elif len(gcov.shape) == 4:
            # Canon ordering is mu,nu,i,j for what I swear are good reasons
            gcov_t = gcov.transpose((2, 3, 0, 1))
            return la.inv(gcov_t).transpose((2, 3, 0, 1))
        elif len(gcov.shape) == 5:
            gcov_t = gcov.transpose((2, 3, 4, 0, 1))
            return la.inv(gcov_t).transpose((3, 4, 0, 1, 2))
        else:
            raise ValueError("Dimensions of gcov are {}.  Should be 4x4 or 4x4xN1xN2".format(gcov.shape))

    def gdet(self, gcov):
        r"""Return the negative root determinant of the metric :math:`\sqrt{-g}`, given the covariant form."""
        # TODO could share more code w/above. Worth it?
        if len(gcov.shape) == 2:
            return np.sqrt(-la.det(gcov))
        elif len(gcov.shape) == 3:
            gcov_t = gcov.transpose((2, 0, 1))
            gdet = np.sqrt(-la.det(gcov_t))
            return gdet
        elif len(gcov.shape) == 4:
            gcov_t = gcov.transpose((2, 3, 0, 1))
            gdet = np.sqrt(-la.det(gcov_t))
            return gdet
        elif len(gcov.shape) == 5:
            gcov_t = gcov.transpose((2, 3, 4, 0, 1))
            gdet = np.sqrt(-la.det(gcov_t))
            return gdet
        else:
            raise ValueError("Dimensions of gcov are {}.  Should be 4x4 or 4x4xN1xN2".format(gcov.shape))

    # TODO Einsum this too
    def conn_func(self, x, delta=1e-5):
        r"""Calculate all connection coefficients :math:`\Gamma^{i}_{j, k}`.
        Returns a 3+N dimensional array conn[i,j,k,...]"""

        conn = np.zeros([4, 4, 4, *(x.shape[1:])])
        tmp = np.zeros_like(conn)

        for mu in range(4):
            xh = np.copy(x)
            xl = np.copy(x)
            xh[mu] += delta
            xl[mu] -= delta
            gh = self.gcov(xh)
            gl = self.gcov(xl)

            # Use the conn array to store metric derivatives (will be replaced)
            conn[:, :, mu] = (gh - gl) / (xh[mu] - xl[mu])

        # Rearrange to find \Gamma_{lam nu mu}
        for lam in range(4):
            for nu in range(4):
                for mu in range(4):
                    tmp[lam, nu, mu] = 0.5 * (conn[nu, lam, mu] + conn[mu, lam, nu] - conn[mu, nu, lam])

        # Raise index to get \Gamma ^ lam_{nu mu}
        gcon = self.gcon(self.gcov(x))
        for lam in range(4):
            for nu in range(4):
                for mu in range(4):
                    conn[lam, nu, mu] = 0
                    for kap in range(4):
                        conn[lam, nu, mu] += gcon[lam, kap] * tmp[kap, nu, mu]
        return conn

    def dxdX(self, x):
        raise NotImplementedError

    # Just take an inverse over first (!) 2 indices
    def dXdx(self, x):
        return np.einsum("...ij->ij...", la.inv(np.einsum("ij...->...ij", self.dxdX(x))))


class Minkowski(CoordinateSystem):
    @classmethod
    def native_startx(cls, met_params):
        # Allow 2 parameter conventions:
        # 1. XNmin/max: simulation convention, resolution-independent
        # 2. startxN/dxN: file convention, uses only parameters present in HARM format
        if 'x1min' in met_params:
            return np.array([0, met_params['x1min'], met_params['x2min'], met_params['x3min']])
        else:
            return np.array([0, met_params['startx1'], met_params['startx2'], met_params['startx3']])

    @classmethod
    def native_stopx(cls, met_params):
        if 'x1max' in met_params:
            return np.array([0, met_params['x1max'], met_params['x2max'], met_params['x3max']])
        else:
            return np.array([0,
                             met_params['startx1'] + (met_params['n1']*met_params['dx1']),
                             met_params['startx2'] + (met_params['n2']*met_params['dx2']),
                             met_params['startx3'] + (met_params['n3']*met_params['dx3'])]
                            )

    # TODO these only make sense for non-negative x
    @classmethod
    def r(cls, x):
        return x[1]**2 + x[2]**2 + x[3]**2

    @classmethod
    def th(cls, x):
        return np.arctan(x[2] / x[1])

    @classmethod
    def phi(cls, x):
        return np.arctan(np.sqrt(x[1] ** 2 + x[2] ** 2) / x[3])

    @classmethod
    def cart_x(cls, x):
        return x[1]

    @classmethod
    def cart_y(cls, x):
        return x[2]

    @classmethod
    def cart_z(cls, x):
        return x[3]

    @classmethod
    def dxdX(cls, x):
        dxdX = np.zeros([4, 4, *x.shape[1:]])
        for i in range(4):
            dxdX[i, i] = 1
        return dxdX

    @classmethod
    def gcov(cls, x):
        gcov = np.zeros([4, 4, *(x.shape[1:])])
        for mu in range(4):
            gcov[mu, mu] = [-1, 1, 1, 1][mu]
        return gcov

    @classmethod
    def gcon(cls, gcov):
        return gcov

    @classmethod
    def gdet(cls, gcov):
        return np.ones([*gcov.shape[2:]])

    @classmethod
    def conn_func(cls, x, delta=1e-5):
        return np.zeros([4, 4, 4, *(x.shape[1:])])

class KS(CoordinateSystem):
    def __init__(self, met_params={'a': 0.9375}):
        self.a = met_params['a']

    def r(self, x):
        return x[1]

    def th(self, x):
        return x[2]

    def phi(self, x):
        return x[3]

    def bl_coord(self, x):
        return self.r(x), self.th(x), self.phi(x)

    def cart_x(self, x):
        return self.r(x)*np.sin(self.th(x))*np.cos(self.phi(x))

    def cart_y(self, x):
        return self.r(x)*np.sin(self.th(x))*np.sin(self.phi(x))

    def cart_z(self, x):
        return self.r(x)*np.cos(self.th(x))

    def dxdX(self, x):
        """Null Transformation"""
        dxdX = np.zeros([4, 4, *x.shape[1:]])
        dxdX[0, 0] = 1
        dxdX[1, 1] = 1
        dxdX[2, 2] = 1
        dxdX[3, 3] = 1
        return dxdX

    def dxdX_cartesian(self, x):
        dxdX = np.zeros([4, 4, *x.shape[1:]])
        r, th, phi = self.bl_coord(x)
        dxdX[0, 0] = 1
        dxdX[1, 1] = np.sin(th)*np.cos(phi)
        dxdX[1, 2] = r*np.cos(th)*np.cos(phi)
        dxdX[1, 3] = -r*np.sin(th)*np.sin(phi)
        dxdX[2, 1] = np.sin(th)*np.sin(phi)
        dxdX[2, 2] = r*np.cos(th)*np.sin(phi)
        dxdX[2, 3] = r*np.sin(th)*np.cos(phi)
        dxdX[3, 1] = np.cos(th)
        dxdX[3, 2] = -r*np.sin(th)
        dxdX[3, 3] = 0
        return dxdX

class MKS(KS):
    def __init__(self, met_params=default_met_params):
        self.a = met_params['a']
        self.hslope = met_params['hslope']

        # For avoiding coordinate singularity
        # We can usually leave this default
        if 'small_theta' in met_params:
            self.small_th = met_params['small_theta']
        else:
            self.small_th = 1.e-20

        # Set radii
        self.r_eh = 1. + np.sqrt(1. - self.a ** 2)
        z1 = 1. + (1. - self.a**2)**(1/3) * ((1. + self.a)**(1/3) + (1. - self.a)**(1. / 3.))
        z2 = np.sqrt(3. * self.a**2 + z1**2)
        self.r_isco = 3. + z2 - (np.sqrt((3. - z1) * (3. + z1 + 2. * z2))) * np.sign(self.a)

    def native_startx(self, met_params):
        # TODO take direct 'startx' from met params?
        if 'startx1' in met_params and 'startx2' in met_params and 'startx3' in met_params:
            return np.array([0, met_params['startx1'], met_params['startx2'], met_params['startx3']])
        elif 'r_in' in met_params:
            # Set startx1 from r_in
            return np.array([0, np.log(met_params['r_in']), 0, 0])
        elif 'n1tot' in met_params and 'r_out' in met_params:
            # Else via a guess, which we propagate back to the originating parameter file
            met_params['r_in'] = np.exp((met_params['n1tot'] * np.log(self.r_eh) / 5.5 - np.log(met_params['r_out'])) /
                                    (-1. + met_params['n1tot'] / 5.5))
            return np.array([0, np.log(met_params['r_in']), 0, 0])
        elif 'n1' in met_params and 'r_out' in met_params:
            # Or a more questionable guess
            met_params['r_in'] = np.exp((met_params['n1'] * np.log(self.r_eh) / 5.5 - np.log(met_params['r_out'])) /
                                    (-1. + met_params['n1'] / 5.5))
            return np.array([0, np.log(met_params['r_in']), 0, 0])
        else:
            print("The only parameters provided to native_startx were: ", met_params)
            raise ValueError("Cannot find or guess startx!")

    def native_stopx(self, met_params):
        if 'r_out' in met_params:
            return np.array([0, np.log(met_params['r_out']), 1, 2*np.pi])
        elif ('startx1' in met_params and 'dx1' in met_params and 'n1' in met_params and
               'startx2' in met_params and 'dx2' in met_params and 'n2' in met_params and
               'startx3' in met_params and 'dx3' in met_params and 'n3' in met_params):
            return np.array([0, met_params['startx1'] + met_params['n1']*met_params['dx1'],
                            met_params['startx2'] + met_params['n2']*met_params['dx2'],
                            met_params['startx3'] + met_params['n3']*met_params['dx3']])
        else:
            raise ValueError("Cannot find or guess stopx!")


    def r(self, x):
        return np.exp(x[1])

    def th(self, x):
        return self.correct_small_th(np.pi*x[2] + ((1. - self.hslope)/2.)*np.sin(2.*np.pi*x[2]))

    def phi(self, x):
        return x[3]

    def cart_x(self, x):
        return self.r(x)*np.sin(self.th(x))*np.cos(self.phi(x))

    def cart_y(self, x):
        return self.r(x)*np.sin(self.th(x))*np.sin(self.phi(x))

    def cart_z(self, x):
        return self.r(x)*np.cos(self.th(x))

    def dxdX(self, x):
        dxdX = np.zeros([4, 4, *x.shape[1:]])
        dxdX[0, 0] = 1
        dxdX[1, 1] = np.exp(x[1])
        dxdX[2, 2] = np.pi - (self.hslope - 1.) * np.pi * np.cos(2. * np.pi * x[2])
        dxdX[3, 3] = 1
        return dxdX


class CMKS(MKS):
    def __init__(self, met_params=default_met_params):
        self.poly_xt = met_params['poly_xt']
        self.poly_alpha = met_params['poly_alpha']
        self.poly_norm = 0.5 * np.pi * 1. / (1. + 1. / (self.poly_alpha + 1.) *
                                             1. / np.power(self.poly_xt, self.poly_alpha))
        super(CMKS, self).__init__(met_params)

    def th(self, x):
        y = 2 * x[2] - 1.
        th_j = self.poly_norm * y * (
                    1. + np.power(y / self.poly_xt, self.poly_alpha) / (self.poly_alpha + 1.)) + 0.5 * np.pi
        return self.correct_small_th(th_j)

    # TODO TODO dxdX


class FMKS(MKS):
    """Funky """
    def __init__(self, met_params=default_met_params):
        super(FMKS, self).__init__(met_params)
        self.poly_xt = met_params['poly_xt']
        self.poly_alpha = met_params['poly_alpha']
        self.mks_smooth = met_params['mks_smooth']
        self.startx1 = self.native_startx(met_params)[1]
        self.poly_norm = 0.5 * np.pi * 1. / (1. + 1. / (self.poly_alpha + 1.) *
                                             1. / np.power(self.poly_xt, self.poly_alpha))

    def th(self, x):
        th_g = np.pi * x[2] + ((1. - self.hslope) / 2.) * np.sin(2. * np.pi * x[2])
        y = 2 * x[2] - 1.
        th_j = self.poly_norm * y * (
                    1. + np.power(y / self.poly_xt, self.poly_alpha) / (self.poly_alpha + 1.)) + 0.5 * np.pi
        return self.correct_small_th(th_g + np.exp(self.mks_smooth * (self.startx1 - x[1])) * (th_j - th_g))

    def dxdX(self, x):
        # TODO evaluate these numerically?
        dxdX = np.zeros([4, 4, *x.shape[1:]])
        dxdX[0, 0] = 1
        dxdX[1, 1] = np.exp(x[1])
        dxdX[2, 1] = -np.exp(self.mks_smooth * (self.startx1 - x[1])) * self.mks_smooth *\
                     (np.pi / 2. - np.pi * x[2] + self.poly_norm * (2. * x[2] - 1.) *
                      (1 + (np.power((-1. + 2 *x[2]) / self.poly_xt, self.poly_alpha)) /
                       (1 + self.poly_alpha)) -
                      1. / 2. * (1. - self.hslope) * np.sin(2. * np.pi * x[2]))
        dxdX[2, 2] = np.pi + (1. - self.hslope) * np.pi * np.cos(2. * np.pi * x[2]) + \
                     np.exp(self.mks_smooth * (self.startx1 - x[1])) * \
                     (-np.pi + 2. * self.poly_norm * (1. + np.power((2. * x[2] - 1.) / self.poly_xt, self.poly_alpha) /
                                                      (self.poly_alpha + 1.)) +
                                                     (2. * self.poly_alpha * self.poly_norm * (2. * x[2] - 1.) *
                                                      np.power((2. * x[2] - 1.) / self.poly_xt, self.poly_alpha - 1.)) /
                                                     ((1. + self.poly_alpha) * self.poly_xt) -
                                                    (1. - self.hslope) * np.pi * np.cos(2. * np.pi * x[2]))
        dxdX[3, 3] = 1
        return dxdX

    def of_ks(self, r, th, phi):
        x1 = np.log(r)
        x3 = phi
        x2 = opt.newton(lambda x2: self.th([0, x1, x2, x3]) - th, 0.5)
        return [0, x1, x2, x3]

    def of_bl(self, r, th, phi):
        pass


class BHAC_MKS(CoordinateSystem):
    # Don't set a default, be careful as we only ever translate existing data with this coordinate system
    def __init__(self, met_params):
        self.a = met_params['a']
        self.hslope = met_params['hslope']

        # For avoiding coordinate singularity
        # We can usually leave this default
        if 'small_theta' in met_params:
            self.small_th = met_params['small_theta']
        else:
            self.small_th = 1.e-20

        # Set radius of horizon
        self.r_eh = 1. + np.sqrt(1. - self.a ** 2)

    def native_startx(self, met_params):
        if 'r_in' in met_params:
            # Set startx1 from r_in
            return np.array([0, np.log(met_params['r_in']), 0, 0])
        else:
            # Else automatically
            return np.array([0,
                             ((met_params['n1tot'] * np.log(self.r_eh) / 5.5 - np.log(met_params['r_out'])) /
                              (1. + met_params['n1tot'] / 5.5)),
                             0, 0])

    def native_stopx(self, met_params):
        return np.array([0, np.log(met_params['r_out']), np.pi, 2*np.pi])

    def r(self, x):
        return np.exp(x[1])

    def th(self, x):
        # BHAC MKS uses 0<X2<pi
        return self.correct_small_th(x[2] + 2*self.hslope/(np.pi**2)*x[2]*(np.pi - 2*x[2])*(np.pi-x[2]))

    def phi(self, x):
        return x[3]

    def cart_x(self, x):
        return self.r(x)*np.sin(self.th(x))*np.cos(self.phi(x))

    def cart_y(self, x):
        return self.r(x)*np.sin(self.th(x))*np.sin(self.phi(x))

    def cart_z(self, x):
        return self.r(x)*np.cos(self.th(x))

    def dxdX(self, x):
        dxdX = np.zeros([4, 4, *x.shape[1:]])
        dxdX[0, 0] = 1
        dxdX[1, 1] = np.exp(x[1])
        dxdX[2, 2] = 1 - 2*self.hslope + 12 * self.hslope * ((x[2] / np.pi)**2 - x[2]/np.pi)
        dxdX[3, 3] = 1
        return dxdX

class BL(CoordinateSystem):
    def __init__(self, met_params={'a': 0.9375}):
        self.a = met_params['a']

    def r(self, x):
        return x[1]

    def th(self, x):
        return x[2]

    def phi(self, x):
        return x[3]

    def bl_coord(self, x):
        return self.r(x), self.th(x), self.phi(x)

    def cart_x(self, x):
        return self.r(x)*np.sin(self.th(x))*np.cos(self.phi(x))

    def cart_y(self, x):
        return self.r(x)*np.sin(self.th(x))*np.sin(self.phi(x))

    def cart_z(self, x):
        return self.r(x)*np.cos(self.th(x))

    def gcov(self, x):
        gcov = np.zeros([4, 4, *(x.shape[1:])])
        r, th, _ = self.bl_coord(x)
        sth = np.abs(np.sin(th))
        s2 = sth * sth
        cth = np.cos(th)
        a2 = self.a**2
        r2 = r**2
        DD = 1. - 2. / r + a2 / r2
        mu = 1. + a2 * cth**2 / r2

        gcov[0, 0] = -(1. - 2. / (r * mu))
        gcov[0, 3] = -2. * self.a * s2 / (r * mu)
        gcov[3, 0] = gcov[0, 3]
        gcov[1, 1] = mu / DD
        gcov[2, 2] = r2 * mu
        gcov[3, 3] = r2 * sth * sth * (1. + a2 / r2 + 2. * a2 * s2 / (r2 * r * mu))

        return gcov

    def dxdX(self, x):
        """Transformation matrix for vectors from BL to KS"""
        dxdX = np.zeros([4, 4, *x.shape[1:]])
        r, _, _ = self.bl_coord(x)

        dxdX[0, 0] = 1
        dxdX[0, 1] = 2. * r / (r**2 - 2.*r + self.a**2)
        dxdX[1, 1] = 1
        dxdX[2, 2] = 1
        dxdX[3, 1] = self.a / (r**2 - 2.*r + self.a**2)
        dxdX[3, 3] = 1
        return dxdX


# OTHER METRICS SUPPORT TBD
    # elif mtype == Met.MKS3:
    #     # TODO take these as params, bring this in line with above w.r.t function name
    #     #if koral_rad: R0=-1.35; H0=0.7; MY1=0.002; MY2=0.02; MP0=1.3
    #     #else: MAD: R0=0; H0=0.6; MY1=0.0025; MY2=0.025; MP0=1.2
    #     #      SANE R0=-2; H0=0.6; MY1=0.0025; MY2=0.025; MP0=1.2
    #     R0, H0 = met_params['R0'], met_params['H0']
    #     MY1, MY2, MP0 = met_params['MY1'], met_params['MY2'], met_params['MP0']
    #
    #     dxdX[1, 1] = 1./(X[1] - R0)
    #     dxdX[2, 1] = -((np.power(2, 1 + MP0) * np.power(X[1], -1 + MP0) * MP0 * (MY1 - MY2) * np.arctan(((-2 * X[2] + np.pi) * np.tan((H0 * np.pi) / 2.)) / np.pi)) /
    #                    (H0 * np.power(np.power(X[1], MP0) * (1 - 2 * MY1) + np.power(2, 1 + MP0) * (MY1 - MY2), 2) * np.pi))
    #     dxdX[2, 2] = ( (-2 * np.power(X[1], MP0) * np.tan((H0 * np.pi) / 2.)) /
    #                    (H0 * (np.power(X[1], MP0) * (-1 + 2 * MY1) +
    #                           np.power(2, 1 + MP0) * (-MY1 + MY2)) * np.pi**2 * (1 + (np.power(-2 * X[2] + np.pi, 2) * np.power(np.tan((H0 * np.pi) / 2.), 2)) /
    #                                                                                                          np.pi**2)))
    #     dxdX[3, 3] = 1.
    # elif mtype == Met.EKS:
    #     dxdX[1, 1] = 1. / X[1]
    #     dxdX[2, 2] = 1. / np.pi
    #     dxdX[3, 3] = 1.

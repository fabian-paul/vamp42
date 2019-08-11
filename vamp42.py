import numpy as np
import scipy
import scipy.optimize
import warnings

__all__ = ['VAMP42']
__author__ = 'Fabian Paul <fapa@uchicago.edu>'

def sigma(x):
    return scipy.special.expit(x)

class VAMP42(object):
    def __init__(self, X, Y, init='linear', publication_gradient=False):
        r'''Find the main kinetic dividing plane with the variational approach to Markov processes (VAMP).
        
        Parameters
        ----------
        X : ndarray((T, 2))
           Initial points form all transition pairs.
        Y : ndarray((T, 2))
           Final points from all transition pairs.
        '''
        # augment X and Y with constant
        T = len(X)
        ones = np.ones(shape=(T, 1))
        self.X = np.hstack((ones, X))
        self.Y = np.hstack((ones, Y))

        if isinstance(init, str) and init=='random':
            # pick proper scale and shift that matches the data
            scales = 1./(np.std(np.concatenate((X, Y)), axis=0) + 1.E-8)
            self.initial = np.zeros(X.shape[1] + 1)
            self.initial[1:] = (np.random.rand(X.shape[1]) - 0.5) * scales
            self.initial[0] = -np.vdot(self.initial[1:], np.mean(np.concatenate((X, Y)), axis=0))  # make initial plane go through the data mean
        elif isinstance(init, str) and init=='linear':
            # find an intial point, by solving the linear problem
            C00 = np.dot(self.X.T, self.X) / T
            C11 = np.dot(self.Y.T, self.Y) / T
            C01 = np.dot(self.X.T, self.Y) / T
            C00_inv = np.linalg.inv(C00)
            C11_inv = np.linalg.inv(C11)
            values, vectors = np.linalg.eig(np.linalg.multi_dot((C00_inv, C01, C11_inv, C01.T)))
            order = np.argsort(values.real)
            principal_vector = vectors[:, order[-2]].real
            self.initial = principal_vector
        else:
            self.initial = init
        self.publication_gradient = publication_gradient

    def selftest(self, delta=1.E-8):
        #err = scipy.optimize.check_grad(self.function, self.gradient, self.initial, epsilon=delta)
        #if err > delta*10:
        #    warnings.warn('Gradient self-test failed.')
        #return err

        x0 = self.initial
        direction = np.random.rand(len(self.initial)) * delta
        f1, grad1 = self.score_function_and_gradient(x0)
        f2 = self.score_function(x0 + direction)
        df = np.dot(grad1, direction)
        #rel_err = np.abs(f2 - f1 - df) / max(abs(f1), abs(f2))
        rel_err = np.abs(f2 - f1 - df) / np.abs(f2 - f1)
        # rel_err is second order error term / first order error term = first order error term
        sign_correct = (np.sign(df) == np.sign(f2 - f1))
        if not sign_correct:
            warnings.warn('Analytical gradient has wrong sign.')
        if rel_err > 0.001:
            warnings.warn('Error %f of analytical gradient seems too large. Expected O(eps).' % rel_err)
            # print('Error %f of analytical gradient seems too large. Expected O(eps).' % rel_err)
        return rel_err, sign_correct

    def run(self, approx=False):
        'Perform optimization of the VAMP score, finding the optimal parameters of the sigmoid.'
        if approx:
            fprime = None
        else:
            self.selftest()
            fprime = self.score_gradient
        xopt, fopt, self._gopt, Bopt, self._func_calls, self._grad_calls, self._warnflag, hist = \
            scipy.optimize.fmin_bfgs(f=self.score_function, x0=self.initial, fprime=fprime, disp=0, full_output=True, retall=True)
        if self._warnflag != 0:
            if self._warnflag == 1:
                warnings.warn('BFGS returned with error: Maximum number of iterations exceeded.')
            elif self._warnflag == 2:
                warnings.warn('BFGS returned with error: Gradient and/or function calls not changing.')
            else:
                warnings.warn('BFGS returned with error code ' + str(self._warnflag))
        self._b = xopt
        self._score = -fopt
        self.hist = -np.array([self.score_function(x) for x in hist])

        # find normalization of singular functions
        self._std_left = np.std(sigma(np.dot(self.X, self._b)))
        self._std_right = np.std(sigma(np.dot(self.Y, self._b)))

        self._var_left = np.var(sigma(np.dot(self.X, self._b)))
        self._var_right = np.var(sigma(np.dot(self.Y, self._b)))

        return self

    @property
    def b(self):
        return self._b

    @property
    def ext_hnf(self):
        'Hesse normal form (normal, -distance) and steepness parameter of the dividing plane.'
        norm = np.linalg.norm(self._b[1:])
        if self._b[0] / norm > 0:
            norm = -norm
        return (self._b[1:] / norm, self._b[0] / norm, norm)

    @property
    def ext_hnf_initial(self):
        'Hesse normal form (normal, -distance) and steepness parameter of the dividing plane.'
        norm = np.linalg.norm(self.initial[1:])
        if self.initial[0] / norm > 0:
            norm = -norm
        return (self.initial[1:] / norm, self.initial[0] / norm, norm)

    @property
    def hnf(self):
        'Hesse normal form (normal, -distance) of the dividing plane.'
        ext_hnf = self.ext_hnf
        return (ext_hnf[0], ext_hnf[1])

    def score_function_and_gradient(self, b, which='both'):
        T = len(self.X)
        sxp = sigma(np.dot(self.X, b))  # sxp: read sigma of x, plus
        sxm = 1. - sxp
        syp = sigma(np.dot(self.Y, b))
        sym = 1. - syp
        sxpm = sxp*sxm
        sypm = syp*sym

        C00 = np.zeros((2, 2))
        C11 = np.zeros((2, 2))
        C01 = np.zeros((2, 2))

        C00[0, 0] = np.vdot(sxp, sxp) / T
        C00[1, 1] = np.vdot(sxm, sxm) / T
        C00[0, 1] = np.vdot(sxp, sxm) / T
        C00[1, 0] = C00[0, 1]

        C11[0, 0] = np.vdot(syp, syp) / T
        C11[1, 1] = np.vdot(sym, sym) / T
        C11[0, 1] = np.vdot(syp, sym) / T
        C11[1, 0] = C11[0, 1]

        C01[0, 0] = np.vdot(sxp, syp) / T
        C01[1, 1] = np.vdot(sxm, sym) / T
        C01[0, 1] = np.vdot(sxp, sym) / T
        C01[1, 0] = np.vdot(sxm, syp) / T
        
        C00_inv = np.linalg.inv(C00)
        C11_inv = np.linalg.inv(C11)

        Kf = np.dot(C00_inv, C01)
        Kr = np.dot(C11_inv, C01.T)

        R = np.einsum('ik,ki', Kf, Kr)
        if which=='function':
            return -R

        # gradient computation starts here
        d = len(b)
        XtXp_sym = np.zeros((2, d, 2))  # XtXP, read: X transpose times X prime
        YtYp_sym = np.zeros((2, d, 2))
        XtYp = np.zeros((2, d, 2))
        YtXp = np.zeros((2, d, 2))

        # XtXp_sym = XtXp + XptX
        XtXp_sym[0, :, 0] = 2*np.dot(self.X.T, sxpm*sxp) / T
        XtXp_sym[1, :, 1] = -2*np.dot(self.X.T, sxpm*sxm) / T
        XtXp_sym[1, :, 0] = np.dot(self.X.T, sxpm*(sxm - sxp)) / T
        #XtXp_sym[1, :, 0] = np.dot(self.X.T, sxpm*(2.*sxm - 1.)) / T
        XtXp_sym[0, :, 1] = XtXp_sym[1, :, 0]

        YtYp_sym[0, :, 0] = 2*np.dot(self.Y.T, sypm*syp) / T
        YtYp_sym[1, :, 1] = -2*np.dot(self.Y.T, sypm*sym) / T
        YtYp_sym[1, :, 0] = np.dot(self.Y.T, sypm*(sym - syp)) / T
        #YtYp_sym[1, :, 0] = np.dot(self.Y.T, sypm*(2.*sym - 1.)) / T
        YtYp_sym[0, :, 1] = YtYp_sym[1, :, 0]

        XtYp[0, :, 0] = np.dot(self.Y.T, sypm*sxp) / T
        XtYp[0, :, 1] = -np.dot(self.Y.T, sypm*sxp) / T
        XtYp[1, :, 0] = np.dot(self.Y.T, sypm*sxm) / T
        XtYp[1, :, 1] = -np.dot(self.Y.T, sypm*sxm) / T
        YptX = np.transpose(XtYp, axes=(2, 1, 0))

        YtXp[0, :, 0] = np.dot(self.X.T, sxpm*syp) / T
        YtXp[0, :, 1] = -np.dot(self.X.T, sxpm*syp) / T
        YtXp[1, :, 0] = np.dot(self.X.T, sxpm*sym) / T
        YtXp[1, :, 1] = -np.dot(self.X.T, sxpm*sym) / T
        XptY = np.transpose(YtXp, axes=(2, 1, 0))

        if not self.publication_gradient:  # gradient rederived by me
            gradient = np.einsum('ij,jk,kni->n', Kf, C11_inv, YtXp + YptX)
            gradient -= np.einsum('ij,jk,knl,li->n', Kf, C11_inv, YtYp_sym, Kr)
            gradient += np.einsum('ij,jk,kni->n', Kr, C00_inv, XptY + XtYp)
            gradient -= np.einsum('ij,jk,knl,li->n', Kr, C00_inv, XtXp_sym, Kf)
        else:  # original formulation form the VAMP paper (as far as I understand it)
            XtXp = np.zeros((2, d, 2))
            YtYp = np.zeros((2, d, 2))
            XtXp[0, :, 0] = np.dot(self.X.T, sxpm*sxp) / T
            XtXp[1, :, 1] = -np.dot(self.X.T, sxpm*sxm) / T
            XtXp[1, :, 0] = np.dot(self.X.T, sxpm*sxm) / T
            XtXp[0, :, 1] = -np.dot(self.X.T, sxpm*sxp) / T
            YtYp[0, :, 0] = np.dot(self.Y.T, sypm*syp) / T
            YtYp[1, :, 1] = -np.dot(self.Y.T, sypm*sym) / T
            YtYp[1, :, 0] = np.dot(self.Y.T, sypm*sym) / T
            YtYp[0, :, 1] = -np.dot(self.Y.T, sypm*syp) / T
            gradient = 2*np.einsum('ij,jk,kni->n', Kf, C11_inv, YtXp)
            gradient -= 2*np.einsum('ij,jk,kl,lni->n', Kf, C11_inv, Kr.T, XtXp)
            gradient = 2*np.einsum('ij,jk,kni->n', Kr, C00_inv, XtYp)
            gradient -= 2*np.einsum('ij,jk,kl,lni->n', Kr, C00_inv, Kf.T, YtYp)

        assert gradient.shape == (d,)

        if which=='both':
            return -R, -gradient
        elif which=='gradient':
            return -gradient
        else:
            raise ValueError('which should one of "function", "gradient", or "both"')

    def score_function(self, b):
        return self.score_function_and_gradient(b, 'function')

    def score_gradient(self, b):
        return self.score_function_and_gradient(b, 'gradient')
        
    def f(self, x):
        'Eigenfunction evaluated at point x.'
        return sigma(np.dot(x, self._b[1:]) + self._b[0])

    def assign(self, x):
        'Crisp assingment of data points to one of the two "states"'
        return (np.dot(x, self._b[1:]) + self._b[0] > 0).astype(int)

    @property
    def eigenvalue(self):
        'The dominant VAMP singular value.'
        return self._score - 1.

    def kinetic_distance(self, a, b, normed=False):
        'Kinetic distance between points a and b.'
        # mean-free property of the singular function is irrelevant here, since we compute the difference
        delta = self.eigenvalue * (sigma(np.dot(a, self._b[1:]) + self._b[0]) - sigma(np.dot(b, self._b[1:]) + self._b[0]))**2
        if normed:
            return delta / self._var_left
        else:
            return delta


if __name__== '__main__':
    dim = 99
    X = np.vstack((np.random.randn(100, dim) + 8*np.ones(dim), 
                   np.random.randn(100, dim)))
    Y = X + np.random.randn(200, dim)    

    print('self test:', VAMP42(X, Y).selftest())
    
    #normal, intercept, steepness = VAMP42(X, Y).run(approx=True).ext_hnf
    #print('normal, intercept, steepness:', normal, intercept, steepness)
    
    vamp = VAMP42(X, Y).run(approx=False)
    normal, intercept, steepness = vamp.ext_hnf
    print('score:', vamp._score)
    #print('normal, intercept, steepness:', normal, intercept, steepness)
    
    

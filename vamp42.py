import numpy as np
import scipy
import scipy.optimize
import warnings

def sigma(x):
    return scipy.special.expit(x)

class VAMP42(object):
    def __init__(self, X, Y):
        # augment X and Y with constant
        ones = np.ones(shape=(len(X), 1))
        self.X = np.hstack((ones, X))
        self.Y = np.hstack((ones, Y))

        # find an intial point, by solving the linear problem
        C00 = np.dot(self.X.T, self.X)
        C11 = np.dot(self.Y.T, self.Y)
        C01 = np.dot(self.X.T, self.Y)
        C00_inv = np.linalg.inv(C00)
        C11_inv = np.linalg.inv(C11)
        values, vectors = np.linalg.eigh(np.linalg.multi_dot((C00_inv, C01, C11_inv, C01.T)))
        order = np.argsort(values)
        principal_vector =  vectors[:, order[-2]]
        # TODO: project all the points to the first IC and find median
        #proj = np.concatenate((np.dot(X, principal_vector), np.dot(Y, principal_vector)))
        #proj = np.dot(X, principal_vector)
        #median_pos = np.argsort(proj)[len(X)//2]]
        #intercept = np.linagl.norm(X[median_pos, :])
        
        self.initial = principal_vector
        print('initial', self.initial)

    def selftest(self, delta=1.E-8):
        #err = scipy.optimize.check_grad(self.function, self.gradient, self.initial, epsilon=delta)
        #if err > delta*10:
        #    warnings.warn('Gradient self-test failed.')
        #return err

        x0 = self.initial
        direction = np.random.rand(len(self.initial)) * delta
        f1, grad1 = self.function_and_gradient(x0)
        f2 = self.function(x0 + direction)
        df = np.dot(grad1, direction)
        #rel_err = np.abs(f2 - f1 - df) / max(abs(f1), abs(f2))
        rel_err = np.abs(f2 - f1 - df) / np.abs(f2 - f1)
        #logging.info('Self-test for VAMP score yields a finite difference of '
        #             '%f and a directional derivative of %f. This corresponds '
        #             'to a relative error of %f.' % (f2-f1, df, err))
        return rel_err

    def run(self, approx=False):
        if approx:
            fprime = None
        else:
            fprime = self.gradient
        xopt, other = scipy.optimize.fmin_bfgs(f=self.function, x0=self.initial, fprime=fprime, disp=1, retall=True)
        print('len', len(other))
        print(other)
        warnflag = 0
        #xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = result
        if warnflag != 0:
            if warnflag == 1:
                warnings.warn('BFGS returned with error: Maximum number of iterations exceeded.')
            elif warnflag == 2:
                warnings.warn('BFGS returned with error: Gradient and/or function calls not changing.')
            else:
                warnings.warn('BFGS returned with error code ' + str(warnflag))
        self._b = xopt
        return self

    @property
    def b(self):
        return self._b

    @property
    def ext_hnf(self):
        norm = np.linalg.norm(self._b[1:])
        return (self._b[1:] / norm, self._b[0] / norm, norm)

    @property
    def hnf(self):
        return (self.ext_hnf[0], self.ext_hnf[1])

    def function_and_gradient(self, b, which='both'):
        sxp = sigma(np.dot(self.X, b))
        sxm = 1. - sxp  # sigma(-np.dot(self.X, b))
        syp = sigma(np.dot(self.Y, b))
        sym = 1. - syp  # sigma(-np.dot(self.Y, b))
        sxpm = sxp*sxm
        sypm = syp*sym

        C00 = np.zeros((2, 2))
        C11 = np.zeros((2, 2))
        C01 = np.zeros((2, 2))

        C00[0, 0] = np.vdot(sxp, sxp)
        C00[1, 1] = np.vdot(sxm, sxm)
        C00[0, 1] = np.vdot(sxp, sxm)
        C00[1, 0] = C00[0, 1]

        C11[0, 0] = np.vdot(syp, syp)
        C11[1, 1] = np.vdot(sym, sym)
        C11[0, 1] = np.vdot(syp, sym)
        C11[1, 0] = C11[0, 1]

        C01[0, 0] = np.vdot(sxp, syp)
        C01[1, 1] = np.vdot(sxm, sym)
        C01[0, 1] = np.vdot(sxp, sym)
        C01[1, 0] = np.vdot(sxm, syp)
        
        C00_inv = np.linalg.inv(C00)
        C11_inv = np.linalg.inv(C11)

        Kf = np.dot(C00_inv, C01)
        Kr = np.dot(C11_inv, C01.T)

        R = np.einsum('ik,ki', Kf, Kr)
        if which=='function':
            return R

        # gradient computation starts here
        d = len(b)
        XtXp_sym = np.zeros((2, d, 2))
        YtYp_sym = np.zeros((2, d, 2))
        XtYp = np.zeros((2, d, 2))
        YtXp = np.zeros((2, d, 2))

        XtXp_sym[0, :, 0] = 2*np.dot(self.X.T, sxpm*sxp)
        XtXp_sym[1, :, 1] = -2*np.dot(self.X.T, sxpm*sxm)
        XtXp_sym[1, :, 0] = np.dot(self.X.T, sxpm*(sxm - sxp))
        #XtXp_sym[1, :, 0] = np.dot(self.X.T, sxpm*(2.*sxm - 1.))
        XtXp_sym[0, :, 1] = XtXp_sym[1, :, 0]

        YtYp_sym[0, :, 0] = 2*np.dot(self.Y.T, sypm*syp)
        YtYp_sym[1, :, 1] = -2*np.dot(self.Y.T, sypm*sym)
        YtYp_sym[1, :, 0] = np.dot(self.Y.T, sypm*(sym - syp))
        #YtYp_sym[1, :, 0] = np.dot(self.Y.T, sypm*(2.*sym - 1.))
        YtYp_sym[0, :, 1] = YtYp_sym[1, :, 0]

        XtYp[0, :, 0] = np.dot(self.Y.T, sypm*sxp)
        XtYp[0, :, 1] = -np.dot(self.Y.T, sypm*sxp)
        XtYp[1, :, 0] = np.dot(self.Y.T, sypm*sxm)
        XtYp[1, :, 1] = -np.dot(self.Y.T, sypm*sxm)
        YptX = np.transpose(XtYp, axes=(2, 1, 0))

        YtXp[0, :, 0] = np.dot(self.X.T, sxpm*syp)
        YtXp[0, :, 1] = -np.dot(self.X.T, sxpm*syp)
        YtXp[1, :, 0] = np.dot(self.X.T, sxpm*sym)
        YtXp[1, :, 1] = -np.dot(self.X.T, sxpm*sym)
        XptY = np.transpose(YtXp, axes=(2, 1, 0))

        gradient = np.einsum('ij,jk,kni->n', Kf, C11_inv, YtXp + YptX)
        gradient -= np.einsum('ij,jk,knl,li->n', Kf, C11_inv, YtYp_sym, Kr)
        gradient += np.einsum('ij,jk,kni->n', Kr, C00_inv, XptY + XtYp)
        gradient -= np.einsum('ij,jk,knl,li->n', Kr, C00_inv, XtXp_sym, Kf)

        assert gradient.shape == (d,)

        if which=='both':
            return R, gradient
        elif which=='gradient':
            return gradient
        else:
            raise ValueError('which should one of "function", "gradient", or "both"')

    def function(self, b):
        return self.function_and_gradient(b, 'function')

    def gradient(self, b):
        return self.function_and_gradient(b, 'gradient')

    def kinetic_distance(self, x, y):
        # get the eigenvalue, which is 1 - the optimal vamp score
        # scale the eigenvector (var ev == 1)
        pass

if __name__== '__main__':
    dim = 2
    X = np.vstack((np.random.randn(100, dim) + 8*np.ones(dim), 
                   np.random.randn(100, dim)))
    Y = X + np.random.randn(200, dim)    

    print('self test:', VAMP42(X, Y).selftest())
    
    #normal, intercept, steepness = VAMP42(X, Y).run(approx=True).ext_hnf
    #print('normal, intercept, steepness:', normal, intercept, steepness)
    
    normal, intercept, steepness = VAMP42(X, Y).run(approx=False).ext_hnf
    print('normal, intercept, steepness:', normal, intercept, steepness)
    
    

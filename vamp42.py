import numpy as np
import scipy
import scipy.optimize
import warnings

__all__ = ['VAMP42']
__author__ = 'Fabian Paul <fapa@uchicago.edu>'

def sigma(x):
    return scipy.special.expit(x)

class VAMP42(object):
    def __init__(self, X, Y, init='linear'):
        # augment X and Y with constant
        T = len(X)
        ones = np.ones(shape=(T, 1))
        self.X = np.hstack((ones, X))
        self.Y = np.hstack((ones, Y))

        if init=='random':
            self.initial = np.random.rand(X.shape[1] + 1)
        elif init=='linear':
            # find an intial point, by solving the linear problem
            C00 = np.dot(self.X.T, self.X) / T
            C11 = np.dot(self.Y.T, self.Y) / T
            C01 = np.dot(self.X.T, self.Y) / T
            C00_inv = np.linalg.inv(C00)
            C11_inv = np.linalg.inv(C11)
            #print('dets', np.linalg.det(C00), np.linalg.det(C11))
            values, vectors = np.linalg.eig(np.linalg.multi_dot((C00_inv, C01, C11_inv, C01.T)))
            #vamp._vectors[:, 2] 
            order = np.argsort(values)
            principal_vector = vectors[:, order[-2]]
            #principal_vector = np.dot(scipy.linalg.sqrtm(C11_inv), vectors[:, order[-2]])
            norm = np.linalg.norm(principal_vector[1:])
            # TODO: shift my median and not by mean
            # TODO: set steepness to some meaningful value (not too high!)
            # TODO: project all the points to the first IC and find median
            #proj = np.concatenate((np.dot(X, principal_vector), np.dot(Y, principal_vector)))
            #proj = np.dot(X, principal_vector)
            #median_pos = np.argsort(proj)[len(X)//2]]
            #intercept = np.linagl.norm(X[median_pos, :])
            self.initial = principal_vector / norm
            stdn = np.dot(np.dot(C00[1:, 1:], principal_vector[1:]), principal_vector[1:])**2  # lala TODO TODO
            #self._values = values
            #self._vectors = vectors
            #self._values2, self._vectors2 = np.linalg.eig(np.linalg.multi_dot((C11_inv, C01.T, C00_inv, C01)))
            #self._C00 = C00
            #self._C11 = C11
            #self._sqrtC00_inv = scipy.linalg.sqrtm(C00_inv)
            #self._sqrtC11_inv = scipy.linalg.sqrtm(C11_inv)
            #print('initial', self.initial)
        else:
            self.initial = init

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
        # rel_err is second order error term / first order error term = first order error term
        sign_correct = (np.sign(df) == np.sign(f2 - f1))
        if not sign_correct:
            warnings.warn('Analytical gradient has wrong sign.')
            print('Analytical gradient has wrong sign.')
        if rel_err > delta*1000:
            warnings.warn('Error %f of analytical gradient seems too large. Expected O(eps).' % rel_err)
            print('Error %f of analytical gradient seems too large. Expected O(eps).' % rel_err)
        #logging.info('Self-test for VAMP score yields a finite difference of '
        #             '%f and a directional derivative of %f. This corresponds '
        #             'to a relative error of %f.' % (f2-f1, df, err))
        return rel_err, sign_correct

    def run(self, approx=False, disp=0):
        if approx:
            fprime = None
        else:
            self.selftest()
            fprime = self.gradient
        xopt, fopt, self._gopt, Bopt, self._func_calls, self._grad_calls, self._warnflag, hist = \
            scipy.optimize.fmin_bfgs(f=self.function, x0=self.initial, fprime=fprime, disp=0, full_output=True, retall=True)
        #print('len', len(other))
        #print(other)
        #print('hist', hist)
        #warnflag = 0
        # = result
        if self._warnflag != 0:
            if self._warnflag == 1:
                warnings.warn('BFGS returned with error: Maximum number of iterations exceeded.')
            elif self._warnflag == 2:
                warnings.warn('BFGS returned with error: Gradient and/or function calls not changing.')
            else:
                warnings.warn('BFGS returned with error code ' + str(self._warnflag))
        self._b = xopt
        self._score = fopt
        self.hist = -np.array([self.function(x) for x in hist])

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
        T = len(self.X)
        sxp = sigma(np.dot(self.X, b))
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
        XtXp_sym = np.zeros((2, d, 2))
        YtYp_sym = np.zeros((2, d, 2))
        XtYp = np.zeros((2, d, 2))
        YtXp = np.zeros((2, d, 2))

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

        gradient = np.einsum('ij,jk,kni->n', Kf, C11_inv, YtXp + YptX)
        gradient -= np.einsum('ij,jk,knl,li->n', Kf, C11_inv, YtYp_sym, Kr)
        gradient += np.einsum('ij,jk,kni->n', Kr, C00_inv, XptY + XtYp)
        gradient -= np.einsum('ij,jk,knl,li->n', Kr, C00_inv, XtXp_sym, Kf)

        assert gradient.shape == (d,)

        if which=='both':
            return -R, -gradient
        elif which=='gradient':
            return -gradient
        else:
            raise ValueError('which should one of "function", "gradient", or "both"')

    def function(self, b):  # -> score function
        return self.function_and_gradient(b, 'function')

    def gradient(self, b):  # -> score gradient
        return self.function_and_gradient(b, 'gradient')
        
    def f(self, X):
        T = len(X)
        ones = np.ones(shape=(T, 1))
        X_aug = np.hstack((ones, X))
        return sigma(np.dot(X_aug, self.b))

    def kinetic_distance(self, x, y):
        # get the eigenvalue, which is 1 - the optimal vamp score
        # scale the eigenvector (var ev == 1)
        pass

if __name__== '__main__':
    dim = 5
    X = np.vstack((np.random.randn(100, dim) + 8*np.ones(dim), 
                   np.random.randn(100, dim)))
    Y = X + np.random.randn(200, dim)    

    print('self test:', VAMP42(X, Y).selftest())
    
    #normal, intercept, steepness = VAMP42(X, Y).run(approx=True).ext_hnf
    #print('normal, intercept, steepness:', normal, intercept, steepness)
    
    vamp = VAMP42(X, Y).run(approx=False)
    normal, intercept, steepness = vamp.ext_hnf
    print('score:', vamp._score)
    print('normal, intercept, steepness:', normal, intercept, steepness)
    
    

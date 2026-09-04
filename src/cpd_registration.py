from builtins import super
import numpy as np
import numbers
from src.em_alignment import EMRegistration


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)


class RigidRegistration(EMRegistration):
    """
    Rigid registration.

    Attributes
    ----------
    R: numpy array (semi-positive definite)
        DxD rotation matrix. Any well behaved matrix will do,
        since the next estimate is a rotation matrix.

    t: numpy array
        1xD initial translation vector.

    s: float (positive)
        scaling parameter.

    A: numpy array
        Utility array used to calculate the rotation matrix.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    YPY: float
        Denominator value used to update the scale factor.
        Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    X_hat: numpy array
        Centered target point cloud.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    """

    def __init__(self, R=None, t=None, s=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Warning!! Nestor's interference at 11/10/2021
        # if self.D != 2 and self.D != 3:
        # raise ValueError(
        #    'Rigid registration only supports 2D or 3D point clouds. Instead got {}.'.format(self.D))

        if R is not None and (
                R.ndim != 2 or R.shape[0] != self.D or R.shape[1] != self.D or not is_positive_semi_definite(R)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(
                    self.D, self.D, R))

        if t is not None and (t.ndim != 2 or t.shape[0] != 1 or t.shape[1] != self.D):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(
                    self.D, t))

        if s is not None and (not isinstance(s, numbers.Number) or s <= 0):
            raise ValueError(
                'The scale factor must be a positive number. Instead got: {}.'.format(s))

        self.R = np.eye(self.D) if R is None else R
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.s = 1 if s is None else s
        # Warning!!! Nestor's addition at (01/11/2021): it creates a flag objects, that enables or disables update of one of the three registration components R,t,s.
        self.flag_in = [1, 1, 1]
        if 'flag_in' in kwargs:
            self.flag_in = kwargs['flag_in']
        self.adj_mask = None
        if 'adj_mask' in kwargs:
            self.adj_mask = kwargs['adj_mask']

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # target point cloud mean
        # muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0),
        # self.Np)
        muX = np.zeros((self.D))  # [0,0,0]
        # source point cloud mean
        # muY = np.divide(
        # np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)
        muY = np.zeros((self.D))  # [0,0,0]

        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        # centered source point cloud
        Y_hat = self.Y - np.tile(muY, (self.M, 1))
        # To take off soon ...
        # self.X_hat = self.X/max(np.max(self.X), np.max(self.Y)) #- np.tile(muX, (self.N, 1))
        # Y_hat = self.Y/max(np.max(self.X), np.max(self.Y))  #- np.tile(muY, (self.M, 1))

        self.YPY = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(Y_hat, Y_hat), axis=1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)
        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D,))
        C[self.D - 1] = np.linalg.det(np.dot(U, V))
        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.flag_in[0] != -1:
            self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.flag_in[2] != -1:
            self.s = np.trace(np.dot(np.transpose(self.A),
                                     np.transpose(self.R))) / self.YPY
        if self.flag_in[1] != -1:
            self.t = np.transpose(muX) - self.s * \
                     np.dot(np.transpose(self.R), np.transpose(muY))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the rigid transformation.

        """
        if Y is None:
            self.TY = self.s * np.dot(self.Y, self.R) + self.t
            return
        else:
            return self.s * np.dot(Y, self.R) + self.t

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAR = np.trace(np.dot(self.A, self.R))
        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X_hat, self.X_hat), axis=1))
        self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / \
                 (2 * self.sigma2) + self.D * self.Np / 2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)
        self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the rigid transformation parameters.

        """
        return self.s, self.R, self.t


class NonRigidRegistration(EMRegistration):
    """
    Non-rigid registration.

    Attributes
    ----------
    beta: float (positive)
        Width of the Gaussian kernel.

    lambda_reg: float (positive)
        Regularization parameter.

    W: numpy array
        MxD matrix of displacement field weights.

    G: numpy array
        MxM Gaussian kernel matrix.

    """

    def __init__(self, beta=None, lambda_reg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = 2.0 if beta is None else beta
        self.lambda_reg = 2.0 if lambda_reg is None else lambda_reg
        self.W = np.zeros((self.M, self.D))
        self.G = self.compute_gaussian_kernel(self.Y, self.Y, self.beta)
        self.adj_mask = None
        if 'adj_mask' in kwargs:
            self.adj_mask = kwargs['adj_mask']

    def compute_gaussian_kernel(self, Y1, Y2, beta):
        diff = Y1[:, None, :] - Y2[None, :, :]
        dist = np.sum(diff ** 2, axis=2)
        return np.exp(-dist / (2 * beta ** 2))

    def update_transform(self):
        """
        Calculate a new estimate of the non-rigid transformation.
        """
        # P1 is Mx1, need diagonal matrix of P1 for P1_diag
        P1_diag = np.diag(self.P1)
        # Solve linear system: (P1_diag * G + lambda_reg * sigma2 * I) * W = P * X - P1_diag * Y
        A = np.dot(P1_diag, self.G) + self.lambda_reg * self.sigma2 * np.eye(self.M)
        B = np.dot(self.P, self.X) - np.dot(P1_diag, self.Y)
        self.W = np.linalg.solve(A, B)

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the non-rigid transformation.
        """
        if Y is None:
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            G_new = self.compute_gaussian_kernel(Y, self.Y, self.beta)
            return Y + np.dot(G_new, self.W)

    def update_variance(self):
        """
        Update the variance of the mixture model.
        """
        qprev = self.q

        # Standard non-rigid CPD variance update:
        # sigma2 = (||X - TY||^2 + lambda_reg * trace(W^T * G * W)) / (N*D)

        # Need TY = Y + GW
        TY = self.Y + np.dot(self.G, self.W)

        # Calculate ||X - TY||^2
        # This part is complex due to probabilistic weights P.
        # Following standard CPD:
        # sigma2 = (sum(sum(P * ||X - TY||^2))) / (Np * D)

        # But wait, EMRegistration does sigma2 update.
        # The base EMRegistration calls this update_variance.
        # Let's simplify and use the common form.

        # ||X - TY||^2 weighted by P
        diff = self.X[None, :, :] - TY[:, None, :]
        sq_diff = np.sum(diff ** 2, axis=2)

        # P is MxN
        # ||X - TY||^2 weighted by P
        self.sigma2 = np.sum(self.P * sq_diff) / (self.Np * self.D)

        # Regularization term (trace(W^T * G * W))
        reg = self.lambda_reg * np.trace(np.dot(self.W.T, np.dot(self.G, self.W)))
        self.sigma2 += reg / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the non-rigid transformation parameters.
        """
        return self.W, self.beta, self.lambda_reg

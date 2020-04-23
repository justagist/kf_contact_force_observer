"""
Kalman filter based FT sensor smoothing and correction [NOT TESTED]
as described in:
Li, C., Zhang, Z., Xia, G., Xie, X. and Zhu, Q., 2018. Efficient force control learning system for industrial robots based on variable impedance control. Sensors, 18(8), p.2539.
"""

import numpy as np


class SystemModelParams(object):
    """
        In practice, the covariance
        matrices Q and R can automatically be calibrated based on the offline experimental data [1].
        It should be mentioned that the larger the weights of Q are chosen, the more the observer will
        rely on the measurements. Larger diagonal elements in Q result in faster response time of the
        corresponding estimates, but this also results in increased noise amplification. 

        [1] https://ieeexplore.ieee.org/document/7914641/

        :type K: float (range (0,1) inclusive)
        :type Q: np.ndarray (shape: [12,12])
        :type R: np.ndarray (shape: [6,6])
    """
    def __init__(self,K, Q, R):
        
        self._K = K # kalman gain
        self._Q = Q # process noise covariance matrix
        self._R = R # measurement noise covariance matrix

    @classmethod
    def use_default_values(cls):
        return cls(K=0.5, Q=np.eye(12), R=np.eye(6))



def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

class KalmanContactForceObserver(object):

    """
    R_j_i : Rotation matrix of frame j with respect to frame i
    r_ij_k: vector from i to j in frame k

    E: origin of end-effector tip frame (where contact occurs)
    C: origin of centre of mass of end-effector attachment
    S: origin of sensor measurement frame (eg. wrist)

    :param m: mass of end-effector
    :param g: gravity vector in world frame
    """

    def __init__(self, R_S_E, R_C_E, R_S_C,
                 m, r_CS_C, r_CE_C, g=np.array([0.,0.,-9.81]),
                 kalman_params=SystemModelParams.use_default_values()):

        self._R_C_E = R_C_E

        self._model_params = kalman_params

        self._x_prev = np.zeros([12,1])
        self._P_prev = np.zeros(self._model_params._Q.shape)
        self._m = m
        self._g = np.asarray(g).reshape([3,1])

        # -- precompute constants
        self._A = np.block([[np.eye(6), np.eye(6)],
                            [np.zeros([6,6]), np.eye(6)]])

        self._H = np.block([[R_S_E,                                                         np.zeros(3,9)],
                            [np.dot(np.dot(R_C_E,(skew(r_CS_C)-skew(r_CE_C))),R_S_C), R_S_E, np.zeros(3,6)]]) 
        self._prod_val = np.dot(R_C_E, skew((r_CE_C)))


    def get_observations(self, ft_vals, R_W_E):
        """
            Get filtered observations by performing one-step Kalman update.

            :param ft_vals: measured force-torque readings
            :param R_W_E: rotation matrix describing base frame with respect to end-effector tip E.

            TODO: remove argument for R_W_E, replace with R_S_W (easier to obtain). Can use
               known rotations to obtain R_W_E and R_W_C for computation

        """

        R_W_C = np.dot(R_W_E, self._R_C_E.T) # --- not 100% sure

        D = self._m * np.block([[-R_W_E],
                                [np.dot(self._prod_val, R_W_C)]])

        # -- prediction step (eq 6)
        x_bar = np.dot(self._A, self._x_prev)
        P_bar = np.dot(np.dot(self._A,self._P_prev),self._A.T) + self._model_params._Q

        # -- correction step (eq 7)
        x = x_bar + self._model_params._K * (ft_vals - np.dot(self._H,x_bar))

        brack_val = np.linalg.inv(np.dot(np.dot(self._H,P_bar),self._H)+self._model_params._R)
        self._model_params._K = np.dot(np.dot(P_bar,self._H.T),brack_val)

        self._P_prev = np.dot(np.eye(12)-self._model_params._K*self._H, P_bar)
        self._x_prev = x

        # -- return observation using descretised observation model (eq 5)
        return np.dot(self._H,x) + np.dot(D,self._g) + np.random.normal(0,self._model_params._R)






import numpy as np
import matplotlib.pyplot as plt

TEST_TIME = 100
DT = 0.1

# x_k = F*x_(k-1) + B*u
F = np.array([
    [1, DT],
    [0, 1]
])

B = np.array([
    [(DT**2)/2],
    [DT]
])

H = np.array([
    [1, 0],
    [0, 1]
])

ACCEL = 1

Q = np.diag([0.5, 0.5])
R = np.diag([0.5, 0.5])

# 0.5 is variance (sigma^2)
OBS_NOISE = np.diag([0.5, 0.5])

plt.axis([0, 6, 0, 20])

def observation(xTrue, u):
    newxTrue = motion_model(xTrue, u)
    newObs = observation_model(xTrue) + OBS_NOISE @ np.random.randn(2, 1)
    return newxTrue, newObs

def motion_model(x, u):
    newx = F @ x + B * u
    return newx

def observation_model(x):
    xObs = H @ x
    return xObs

def kf(oldxEst, oldPEst, xObs, u):
    # Perdiction
    xEstDash = motion_model(oldxEst, u)
    PEstDash = F @ oldPEst @ F.T + Q

    # Kalman Gain
    gain = PEstDash @ H.T @ np.linalg.inv(H*PEstDash*H.T + R)
    xEst = xEstDash + gain @ (xObs - H@xEstDash)

    # Calculate covariance for next iteration
    PEst = (np.eye(len(xEst)) - gain @ H) @ PEstDash
    return xEst, PEst


def main():
    print(__file__ + " start!!")

    time = 0.0
    hxTrue = np.zeros((2,1))
    hxObs = np.zeros((2,1))
    hxEst = np.zeros((2,1))
    hPEst = np.zeros((2,1))

    xTrue = np.zeros((2,1))
    xObS = np.zeros((2,1))
    xEst = np.zeros((2,1))
    PEst = np.zeros((2,2))

    while (time <= TEST_TIME):
        xTrue, xObS = observation(xTrue, ACCEL)
        xEst, PEst = kf(xEst, PEst, xObS, ACCEL)

        hxTrue = np.hstack((hxTrue, xTrue))
        hxObs = np.hstack((hxObs, xObS))
        hxEst = np.hstack((hxEst, xEst))
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-b")
        plt.plot(hxObs[0, :].flatten(), hxObs[1, :].flatten(), "og")
        plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-r")
        plt.pause(0.1)
        time += DT

if __name__ == '__main__':
    main()
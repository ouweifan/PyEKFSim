import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time as tm

TEST_TIME = 10
DT = 0.1

ACCEL = 1

OBS_NOISE = np.diag([0.1, 0.1])

plt.axis([0, 6, 0, 20])

def observation(xTrue, u):
    newxTrue = motion_model(xTrue, u)
    newObs = observation_model(xTrue) + OBS_NOISE @ np.random.randn(2, 1)
    return newxTrue, newObs

def motion_model(xTrue, u):
    Fk = np.array([
        [1, DT],
        [0, 1]
    ])
    Bk = np.array([
        [(DT**2)/2],
        [DT]
    ])
    newxTrue = Fk @ xTrue + Bk * u
    return newxTrue

def observation_model(xTrue):
    xObs = xTrue
    return xObs

def main():
    print(__file__ + " start!!")

    time = 0.0
    hxTrue = np.zeros((2,1))
    hxEst = np.zeros((2,1))

    xTrue = np.zeros((2,1))
    xEst = np.zeros((2,1))

    while (time <= TEST_TIME):
        xTrue, xEst = observation(xTrue, ACCEL)
        hxTrue = np.hstack((hxTrue, xTrue))
        hxEst = np.hstack((hxEst, xEst))
        plt.cla()
        plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-b")
        plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "og")
        plt.pause(0.1)
        time += DT

if __name__ == '__main__':
    main()
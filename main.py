import numpy as np
from scipy.signal import lsim, TransferFunction
import matplotlib.pyplot as plt
import pandas as pd
from plot_mp import plot_mp, deviation


def simulate():
    kv_l = 0.83
    ka_l = 0.1
    kp_l = 200
    ki_l = 1.5
    kd_l = 0
    kf_v_l = 0  # position feedforward
    kf_p_l = 0  # velocity feedforward

    kv_r = 0.85
    ka_r = 0.11
    kp_r = kp_l
    ki_r = ki_l
    kd_r = kd_l
    kf_v_r = 0
    kf_p_r = 0

    # create system model
    left = TransferFunction([kd_l+kf_v_l,kp_l+kf_p_l,ki_l],[ka_l,kd_l+kv_l,kp_l,ki_l])
    right = TransferFunction([kd_r+kf_v_r,kp_r+kf_p_r,ki_r],[ka_r,kd_r+kv_r,kp_r,ki_r])

    # read in profile files
    left_profile = prepare_profile('C:/Users/Sam Ehrenstein/IdeaProjects/robot2017/calciferLeftRedLeftProfile.csv')
    right_profile = prepare_profile('C:/Users/Sam Ehrenstein/IdeaProjects/robot2017/calciferRightRedLeftProfile.csv')

    dt = left_profile[0, 2]
    t_rr = np.arange(0, left_profile.shape[0]*dt, dt)  # time on RoboRIO (time for setpoint updates)
    t = np.linspace(0, t_rr[-1], 1000*dt*left_profile.shape[0])  # time for simulation

    # staircased trajectories for using in the simulation
    u_left = staircase(left_profile, t, dt)
    u_right = staircase(right_profile, t, dt)

    # interpolated trajectories for error analysis
    u_left_c = np.interp(t, t_rr, left_profile[:, 0])
    u_right_c = np.interp(t, t_rr, right_profile[:, 0])

    tout_l, y_l, x_l = lsim(left, u_left, t)
    tout_r, y_r, x_r = lsim(right, u_right, t)

    err_pid_l = y_l-u_left  # raw error (the one the PID sees)
    err_pid_r = y_r-u_right
    err_lerp_l = y_l-u_left_c   # error based off of the interpolated trajectory
    err_lerp_r = y_r-u_right_c

    prof_traj = plot_mp(u_left_c, u_right_c, 0.001)    # path from profile
    actual_traj = plot_mp(y_l, y_r, 0.001)  # actual path followed
    dev = deviation(prof_traj, actual_traj)
    print(dev)

    plt.figure(1)
    plt.subplot(221)
    sp_l, = plt.plot(t, u_left, label='Setpoint')
    act_l, = plt.plot(t, y_l, label='Actual')
    plt.legend(handles=[sp_l, act_l])
    plt.title('Left Setpoint and Position')
    plt.subplot(222)
    l_err, = plt.plot(t, err_lerp_l, label='Error')
    plt.plot(t, np.zeros_like(t), color='black')
    plt.legend(handles=[l_err])
    plt.title('Left Error')
    plt.subplot(223)
    sp_r, = plt.plot(t, u_right, label='Setpoint')
    act_r, = plt.plot(t, y_r, label='Actual')
    plt.legend(handles=[sp_r, act_r])
    plt.title('Right Setpoint and Position')
    plt.subplot(224)
    r_err, = plt.plot(t, err_lerp_r, label='Error')
    plt.plot(t, np.zeros_like(t), color='black')
    plt.legend(handles=[r_err])
    plt.title('Right Error')

    plt.subplots_adjust(hspace=0.4)

    plt.figure(2)
    prof_left, = plt.plot(prof_traj[:, 1], prof_traj[:, 2], label='Predicted Left')
    prof_right, = plt.plot(prof_traj[:, 3], prof_traj[:, 4], label='Predicted Right')
    actual_left, = plt.plot(actual_traj[:,1], actual_traj[:,2], label='Actual Left')
    actual_right, = plt.plot(actual_traj[:,3],actual_traj[:,4], label='Actual Right')
    plt.legend(handles=[prof_left, prof_right, actual_left, actual_right])

    # plt.figure(3)
    # l_dev, = plt.plot(t, dev[0], label='Left deviation')
    # r_dev, = plt.plot(t, dev[1], label='Right deviation')
    # plt.title('Deviations from expected x,y position')
    # plt.legend(handles=[l_dev, r_dev])

    plt.show()


# Reads in a profile and removes the first row (since it's just the number of lines)
def prepare_profile(filename):
    prof = pd.read_csv(filename, skiprows=[0], header=None).values
    prof = prof[:, 0:3]
    return prof


# Expands the profile to be the length of t such that u(t) has the last value that would have been sent
# This is used instead of interpolating because the setpoint is updated slower than the robot's response time
def staircase(profile, t, dt):
    u = np.zeros_like(t)
    for i in range(u.shape[0]):
        u[i] = profile[int(np.ceil(t[i]/dt)), 0]
    return u

simulate()

# % Copyright (c) 2018 Sam Ehrenstein.
# %
# % Permission is hereby granted, free of charge, to any person obtaining a copy
# % of this software and associated documentation files (the "Software"), to deal
# % in the Software without restriction, including without limitation the rights
# % to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# % copies of the Software, and to permit persons to whom the Software is
# % furnished to do so, subject to the following conditions:
# %
# % The above copyright notice and this permission notice shall be included in all
# % copies or substantial portions of the Software.
# %
# % THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# % IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# % FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# % AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# % LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# % OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# % SOFTWARE.

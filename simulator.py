import numpy as np
import pandas as pd
from plot_mp import plot_mp, deviation
import bokeh.plotting as bp
from bokeh.layouts import gridplot, widgetbox, column
from bokeh.models.widgets import Panel, Tabs, TextInput, Button, Div
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
import matplotlib.pyplot as plt

# Copyright (c) 2018 Sam Ehrenstein. The full copyright notice is at the bottom of this file.

def prepare_profile(filename):
    prof = pd.read_csv(filename, skiprows=[0], header=None).values
    prof = prof[:, 0:3]
    return prof

# Expands the profile to be the length of t such that u(t) has the last value that would have been sent
# This is used instead of interpolating because the setpoint is updated slower than the robot's response time
def staircase(profile, t, dt, dt_prof):
    u = np.zeros((t.shape[0], 3))
    ratio = dt_prof/dt
    for i in range(profile.shape[0]):
        u[int(i*ratio):int((i+1)*ratio)] = profile[i]
    return u

def simulate_motor(kp, ki, kd, kf, ka, kv, profile, t):
    pos = np.zeros_like(t)
    vel = np.zeros_like(t)
    err = np.zeros_like(t)
    integral = 0
    vc = 12 - 3    # replace 3 with the intercept voltage
    for i in range(1, t.shape[0]):
        dt = t[i]-t[i-1]
        c2 = ka/kv*(vc/kv-vel[i-1])
        c1 = -c2 + pos[i-1]
        pos[i] = c1 + vc/kv*dt + c2*np.exp(-kv/ka*dt)
        vel[i] = vc/kv-c2*kv/ka*np.exp(-kv/ka*dt)
        err[i] = profile[i, 0]-pos[i]
        dedt = (err[i]-err[i-1])/dt
        integral += err[i]*dt
        vc = kp*err[i] + ki*integral + kd*dedt + kf*profile[i, 1]
        if vc > 9:
            vc = 9
    return pos, err

def simulate(args):
    kv_l = args['kv_l']
    ka_l = args['ka_l']
    kv_r = args['kv_r']
    ka_r = args['ka_r']
    kp = args['kp']
    ki = args['ki']
    kd = args['kd']
    kf = args['kf']

    dt = 0.001
    dt_prof = 0.05

    left_profile = prepare_profile('demoLeft.csv')
    right_profile = prepare_profile('demoRight.csv')
    # print('left profile', left_profile.shape)
    t = np.arange(0, left_profile.shape[0]*dt_prof-dt, dt)
    print('t shape', t.shape)
    u_left = staircase(left_profile, t, dt, dt_prof)
    u_right = staircase(right_profile, t, dt, dt_prof)

    pos_l, err_l = simulate_motor(kp, ki, kd, kf, ka_l, kv_l, u_left, t)
    pos_r, err_r = simulate_motor(kp, ki, kd, kf, ka_r, kv_r, u_right, t)

    print(t.shape)
    plt.plot(t, u_left[:,0], t, pos_l)
    plt.show()

k_args = {'kp':0.8,
'ki':0,
'kd':0,
'kf':0.8,
'kv_l':0.83,
'ka_l':0.11,
'kv_r':0.85,
'ka_r':0.1
}
simulate(k_args)

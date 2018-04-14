import numpy as np
from scipy.signal import lsim, TransferFunction
import pandas as pd
from plot_mp import plot_mp, deviation
import bokeh.plotting as bp
from bokeh.layouts import gridplot, widgetbox, column
from bokeh.models.widgets import Panel, Tabs, TextInput, Button
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource


# Copyright (c) 2018 Sam Ehrenstein. The full copyright notice is at the bottom of this file.




def simulate(args):
    # Left side constants
    kv_l = 0.83         # Kv
    ka_l = 0.1          # Ka
    kp_l = args['kp']   # Kp
    ki_l = 0            # Ki
    kd_l = 0            # Kd
    kf_v_l = 0          # position feedforward
    kf_p_l = 0          # velocity feedforward

    # Right side constants
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
    left_profile = prepare_profile('demoLeft.csv')
    right_profile = prepare_profile('demoRight.csv')

    dt = left_profile[0, 2]
    dt_sim = 0.001
    t_rr = np.arange(0, left_profile.shape[0]*dt, dt)  # time on RoboRIO (time for setpoint updates)
    t = np.linspace(0, t_rr[-1], np.floor(1/dt_sim*dt*left_profile.shape[0]))  # time for simulation

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

    prof_traj = plot_mp(u_left_c, u_right_c, dt_sim)    # path from profile
    actual_traj = plot_mp(y_l, y_r, dt_sim)  # actual path followed
    dev = deviation(prof_traj, actual_traj)

    # Create ColumnDataSources for each plot
    u_time = ColumnDataSource(data=dict(t=t, l=u_left, r=u_right))
    y_time = ColumnDataSource(data=dict(t=t, l=y_l, r=y_r))
    err_time = ColumnDataSource(data=dict(t=t, l=err_lerp_l, r=err_lerp_r))
    pt_data = ColumnDataSource(data=dict(lx=prof_traj[:,1], ly=prof_traj[:,2], rx=prof_traj[:,3], ry=prof_traj[:,4]))
    at_data = ColumnDataSource(data=dict(lx=actual_traj[:,1], ly=actual_traj[:,2], rx=actual_traj[:,3], ry=actual_traj[:,4]))
    dev_data = ColumnDataSource(data=dict(t=t, l=dev[0], r=dev[1]))
    return u_time, y_time, err_time, pt_data, at_data, dev_data


def update_sim(attrname, old, new):
    kp = float(kp_input.value)

    args = {'kp': kp}
    simulate(args)


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


u, y, err, pt, at, dev = simulate({'kp': 1.5})

# Plot predicted and actual robot motion in x-y coordinates
p1 = bp.figure(plot_width=500, plot_height=500, x_range=(-1, 11), y_range=(-10, 2))
p1.line('lx', 'ly', source=pt, line_width=2, line_color='navy', legend='Predicted left')
p1.line('rx', 'ry', source=pt, line_width=2, line_color='orange', legend='Predicted right')
p1.line('lx', 'ly', source=at, line_width=2, line_color='green', legend='Actual left')
p1.line('rx', 'ry', source=at, line_width=2, line_color='red', legend='Actual right')

# Plot deviation
dev_plot = bp.figure(plot_width=500, plot_height=500)
dev_plot.line('t', 'l', source=dev, line_color='green', legend='Left deviation')
dev_plot.line('t', 'r', source=dev, line_color='red', legend='Right deviation')
dev_tab = Panel(child=dev_plot, title='Deviation')

# Plot analytics for each profile
left_sp = bp.figure(plot_width=250, plot_height=250, title='Left Setpoint and Position')
left_sp.line('t', 'l', source=u, line_color='blue', legend='Setpoint')
left_sp.line('t', 'l', source=y, line_color='orange', legend='Actual')
right_sp = bp.figure(plot_width=250, plot_height=250, title='Right Setpoint and Position')
right_sp.line('t', 'r', source=u, line_color='blue', legend='Setpoint')
right_sp.line('t', 'r', source=y, line_color='orange', legend='Actual')
left_err = bp.figure(plot_width=250, plot_height=250, title='Left Error')
left_err.line('t', 'l', source=err, line_color='blue', legend='Error')
right_err = bp.figure(plot_width=250, plot_height=250, title='Right Error')
right_err.line('t', 'r', source=err, line_color='blue', legend='Error')
analytics = gridplot([[left_sp, right_sp], [left_err, right_err]])

kp_input = TextInput(title='Kp', value='1.5')
submit = Button(label='Simulate', button_type='success')

inputs = widgetbox(kp_input, submit)
p_tab = Panel(child=column(inputs, p1), title='Path')

bp.show(Tabs(tabs=[p_tab, dev_tab, Panel(child=analytics, title='Analytics')]))


# This file is part of MP-Sim.
#
# This program is free software; you can redistribute it and / or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
# Fifth Floor, Boston, MA 02110 - 1301 USA

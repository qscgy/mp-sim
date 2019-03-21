import numpy as np
import pandas as pd
from plot_mp import plot_mp, deviation
import matplotlib.pyplot as plt
from util import *

# Copyright (c) 2018-2019 Sam Ehrenstein. The full copyright notice is at the bottom of this file.


class Motor():
    """Represents a motor or motor cluster."""
    def __init__(self, ka, kv):
        self.ka = ka
        self.kv = kv

    def next(self, last_pos, last_vel, vc, dt):
        c2 = self.ka/self.kv*(vc/self.kv-last_vel)
        c1 = -c2 + last_pos
        pos = c1 + vc/self.kv*dt + c2*np.exp(-self.kv/self.ka*dt)
        vel = vc/self.kv-c2*self.kv/self.ka*np.exp(-self.kv/self.ka*dt)
        return pos, vel


class PIDMotor():
    def __init__(self, kp, ki, kd, motor, profile, t):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.motor = motor
        self.profile = profile
        self.t = t
        self.error = np.zeros_like(t)
        self.pos = np.zeros_like(t)
        self.vel = np.zeros_like(t)
        self.ig = 0
    
    def next(self, vc, i, dt):
        self.pos[i], self.vel[i] = self.motor.next(self.pos[i-1], self.vel[i-1], vc, dt)
        self.error[i] = self.profile[i,0]-self.pos[i]
        dedt = (self.error[i]-self.error[i-1])/dt
        self.ig += self.error[i]*dt
        vc = self.kp*self.error[i] + self.ki*self.ig + self.kd*dedt + self.motor.kv*self.profile[i,1]
        if vc > 9:
            vc = 9
        return self.pos[i], self.vel[i], vc


class TankDrive():
    def __init__(self, lmotor, rmotor, wb):
        self.lmotor = lmotor
        self.rmotor = rmotor
        self.wb = wb
    
    def simulate(self, kp, ki, kd, kpa, kia, kda, lprofile, rprofile, t):
        left = PIDMotor(kp, ki, kd, self.lmotor, lprofile, t)
        right = PIDMotor(kp, ki, kd, self.rmotor, rprofile, t)
        pose = np.zeros((t.shape[0], 6))
        theta = np.zeros_like(t)
        theta_ig = 0
        e_theta = np.zeros_like(t)
        corr = 0
        lvc = 9
        rvc = 9
        dt = t[1]-t[0]
        dt_rr = lprofile[0, 3]      # dt of roboRIO
        ratio = int(dt_rr/dt)
        for i in range(1, t.shape[0]):
            pose[i, 0], pose[i, 2], lvc = left.next(lvc-corr, i, dt) # run left for dt and get PID output
            pose[i, 3], pose[i,5], rvc = right.next(rvc+corr, i, dt)

            # The roboRIO runs slower than the Talons, so it only applies angular correction on its updates.
            if i % ratio == 0:
                delta_l = pose[i,0]-pose[i-ratio,0]
                delta_r = pose[i,3]-pose[i-ratio,3]
                tp = (delta_r-delta_l)/self.wb
                theta[i] = theta[i-ratio] + tp
                corr, e_theta[i], theta_ig = next_pid(kpa, kia, kda, 0, theta[i], 0, e_theta[i-ratio],
                    theta_ig, np.deg2rad(lprofile[i, 4]), 0, dt_rr)
            else:
                theta[i] = theta[i-1]
                e_theta[i] = e_theta[i-1]

        pose[:,1] = left.error
        pose[:,4] = right.error
        return pose, theta


def simulate(args):
    kv_l = args['kv_l']
    ka_l = args['ka_l']
    kv_r = args['kv_r']
    ka_r = args['ka_r']
    kp = args['kp']
    ki = args['ki']
    kd = args['kd']
    kpa = args['kpa']
    kia = args['kia']
    kda = args['kda']
    wb = args['wb']
    left_file = args['leftprof']
    right_file = args['rightprof']

    dt = 0.001
    dt_prof = 0.05

    left_profile = prepare_profile(left_file)
    right_profile = prepare_profile(right_file)
    t = np.arange(0, left_profile.shape[0]*dt_prof-dt, dt)
    u_left = staircase(left_profile, t, dt, dt_prof)
    u_right = staircase(right_profile, t, dt, dt_prof)

    left = Motor(ka_l, kv_l)
    right = Motor(ka_r, kv_r)
    drive = TankDrive(left, right, wb)
    pose, theta = drive.simulate(kp, ki, kd, kpa, kia, kda, u_left, u_right, t)
    pos_l = pose[:,0]
    err_l = pose[:,1]
    vel_l = pose[:,2]
    pos_r = pose[:,3]
    err_l = pose[:,4]
    vel_l = pose[:,5]

    trajectories = plot_mp(pos_l, pos_r, wb, dt)
    trajectories_prof = plot_mp(u_left[:,0], u_right[:,0], wb, dt)

    plt.figure()
    plt.plot(trajectories_prof[:,1], trajectories_prof[:,2], label='intended left')
    plt.plot(trajectories_prof[:,3], trajectories_prof[:,4], label='intended right')
    plt.plot(trajectories[:,1], trajectories[:,2], label='actual left')
    plt.plot(trajectories[:,3], trajectories[:,4], label='actual right')
    plt.title('Intended vs. Actual Trajectories')
    plt.xlabel('x displacement (ft)')
    plt.ylabel('y displacement (ft')
    plt.legend()

    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.plot(t, u_left[:,0], label='left profile')
    plt.plot(t, pos_l, label='actual left distance')
    plt.title('Left side path')
    plt.xlabel('time (s)')
    plt.ylabel('distance (ft)')
    plt.legend()
    plt.subplot(212)
    plt.plot(t, u_right[:,0], label='right profile')
    plt.plot(t, pos_r, label='actual right distance')
    plt.title('Right side path')
    plt.xlabel('time (s)')
    plt.ylabel('distance (ft)')
    plt.legend()

    plt.figure()
    plt.plot(t, u_left[:,4], label='intended theta')
    plt.plot(t, theta*180/np.pi, label='actual theta')
    plt.legend()
    plt.title('Intended vs. Actual Theta')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')

    plt.show()

k_args = {'kp':7,
'ki':0,
'kd':0.3,
'kv_l':0.83,
'ka_l':0.1,
'kv_r':0.85,
'ka_r':0.11,
'kpa':0,
'kia':0,
'kda':0,
'wb':26/12,
'leftprof':'demoLeft2.csv',
'rightprof':'demoRight2.csv'
}
simulate(k_args)

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
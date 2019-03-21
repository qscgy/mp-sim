import numpy as np
import pandas as pd

def prepare_profile(filename):
    prof = pd.read_csv(filename, skiprows=[0], header=None).values
    prof = prof[:, 0:5]
    return prof

# Expands the profile to be the length of t such that u(t) has the last value that would have been sent
# This is used instead of interpolating because the setpoint is updated slower than the robot's response time
def staircase(profile, t, dt, dt_prof):
    """
    Given a profile with some time step, returns a new profile with the same value
    at any given time, but with a different time step. In other words, a zeroth-order
    interpolation of the profile with a new time step.
    
    Args:
        profile (np.array): the profile to staircase
        t (np.array): the array times at which the returned profile should have values
        dt (float): the time step of t;  should divide into dt_prof
        dt_prof (float): the time step of profile

    Returns:
        np.array: the new profile
    """
    u = np.zeros((t.shape[0], 5))
    ratio = dt_prof/dt
    for i in range(profile.shape[0]):
        u[int(i*ratio):int((i+1)*ratio)] = profile[i]
    return u

def next_motor(vc, ka, kv, last_pos, last_vel, dt):
    """
    Given the constants for a system moved by a motor under PID control and 
    information about its state at the start of a time slice, returns its 
    state after some time `dt` has passed.

    Args:
        vc (float): the voltage at the start of the time slice
        ka (float): the kA for the motor, see README
        kv (float): the kV for the motor, see README
        last_pos (float): distance traveled by the system at the start of the time slice
        last_vel (float): velocity of the system at the start of the time slice
        dt (float): the length of the time slice

    Returns:
        np.array: an array of the form [position, error, velocity]
        float: the voltage at the end of 
    """    
    c2 = ka/kv*(vc/kv-last_vel)
    c1 = -c2 + last_pos
    pos = c1 + vc/kv*dt + c2*np.exp(-kv/ka*dt)
    vel = vc/kv-c2*kv/ka*np.exp(-kv/ka*dt)
    return pos, vel

def next_pid(kp, ki, kd, kv, pos, vel, last_err, ig, xset, vset, dt):
    """
    Computes the next supplied voltage (our process variable) of a motor-system under position PID control
    with a velocity feedforward.

    Args:
        kp (float): P value in the PID
        ki (float): I value in the PID
        kd (float): D value in the PID
        kv (float): kV of the system, see README
        pos (float): the current linear position
        vel (float): the current linear velocity
        last_err (float): the last value of the position error
        ig (float): the integral of the position error
        xset (float): the current position setpoint
        vset (float): the current velocity setpoint
        dt (float): the time step of the simulation

    Returns:
        float: the next output
        float: the current position error
        float: the current integral of the error
    """    
    err = xset-pos
    dedt = (err-last_err)/dt
    ig += err*dt
    vc = kp*err + ki*ig + kd*dedt + kv*vset
    if vc > 9:
        vc = 9
    return vc, err, ig

def simulate_tank(kp, ki, kd, lka, lkv, rka, rkv, lprofile, rprofile, t, wb, kpa, kia, kda):
    """
    Simulates a tank-drive robot following motion profiles under position PID contro
    with a velocity feedforward, cascaded with an outer PID loop on angular heading.

    Args:
        kp (float): P value in the position PID
        ki (float): I value in the position PID
        kd (float): D value in the position PID
        lka (float): left side kA, see README
        lkv (float): left side kV, see README
        rka (float): right side kA, see README
        rkv (float): right side kV, see README
        lprofile (np.array): left side profile, columns are position, velocity, acceleration, dt, heading
        rprofile (np.array): right side profile, same format as left side
        t (np.array): 1D array of the times at which the profile is evaluated
        wb (float): effective wheelbase diameter
        kpa (float): P in angular PID
        kia (float): I in angular PID
        kda (float): D in angular PID

    Returns:
        np.array: an array with as many rows as t and 6 columns containing the pose,
        in the form left pos, left err, left vel, right pos, right err, right vel
        np.array : an array with the same shape as t, containing the heading at each time
    """    
    pose = np.zeros((t.shape[0], 6))
    l_ig = 0
    r_ig = 0
    theta = np.zeros_like(t)
    theta_ig = 0
    e_theta = np.zeros_like(t)
    corr = 0
    lvc = 9
    rvc = 9
    for i in range(1, t.shape[0]):
        dt = t[i]-t[i-1]
        pose[i, 0], pose[i, 2] = next_motor(lvc-corr, lka, lkv, pose[i-1,0],
            pose[i-1,2], dt)
        lvc, pose[i, 1], l_ig = next_pid(kp, ki, kd, lkv, pose[i, 0], pose[i,2], pose[i-1,1],l_ig,
            lprofile[i,0], lprofile[i,1], dt)
        pose[i, 3], pose[i,5] = next_motor(rvc+corr, rka, rkv, pose[i-1,3], 
            pose[i-1,5], dt)
        rvc, pose[i, 4], r_ig = next_pid(kp, ki, kd, rkv, pose[i, 3], pose[i,5],pose[i-1,4], r_ig,
            rprofile[i,0], rprofile[i,1], dt)
        delta_l = pose[i,0]-pose[i-1,0]
        delta_r = pose[i,3]-pose[i-1,3]
        tp = (delta_r-delta_l)/wb
        theta[i] = theta[i-1] + tp
        corr, e_theta[i], theta_ig = next_pid(kpa, kia, kda, 0, theta[i], 0, e_theta[i-1],
            theta_ig, np.deg2rad(lprofile[i, 4]), 0, dt)

    return pose, theta

    # pos = np.zeros_like(t)
    # vel = np.zeros_like(t)
    # err = np.zeros_like(t)
    # integral = 0
    # vc = 9
    # for i in range(1, t.shape[0]):
    #     dt = t[i]-t[i-1]
    #     c2 = ka/kv*(vc/kv-vel[i-1])
    #     c1 = -c2 + pos[i-1]
    #     pos[i] = c1 + vc/kv*dt + c2*np.exp(-kv/ka*dt)
    #     vel[i] = vc/kv-c2*kv/ka*np.exp(-kv/ka*dt)
    #     err[i] = profile[i, 0]-pos[i]
    #     dedt = (err[i]-err[i-1])/dt
    #     integral += err[i]*dt
    #     vc = kp*err[i] + ki*integral + kd*dedt + kv*profile[i, 1]
    #     if vc > 9:
    #         vc = 9
    # return pos, err, vel
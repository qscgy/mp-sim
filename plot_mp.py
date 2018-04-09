import numpy as np

def plot_mp(left, right, dt):
    wheelbase_dia = 26.0/12
    startingCenter = (-10.3449+(29./2.+3.25)/12., (27./2.+3.25)/12.)

    out = np.zeros((left.shape[0], 5))
    out[0] = np.array([0, startingCenter[1], startingCenter[0]+wheelbase_dia/2,startingCenter[1],startingCenter[1]-wheelbase_dia/2])
    for i in range(1, out.shape[0]):
        perpendicular = angle_between(out[i-1, 1], out[i-1, 2], out[i-1, 3], out[i-1, 4])-np.pi/2
        out[i, 0] = out[i-1, 0] + dt
        delta_left = left[i]-left[i-1]
        delta_right = right[i]-right[i-1]

        theta = (delta_left-delta_right)/wheelbase_dia
        if theta == 0:
            out[i, 1] = out[i-1, 1] + delta_left*np.cos(perpendicular)
            out[i, 2] = out[i-1, 2] + delta_left*np.sin(perpendicular)
            out[i, 3] = out[i-1, 3] + delta_right*np.cos(perpendicular)
            out[i, 4] = out[i-1, 4] + delta_right*np.sin(perpendicular)
        else:
            rightR = (wheelbase_dia/2)*(delta_left+delta_right)/(delta_left-delta_right)-wheelbase_dia/2
            leftR = rightR + wheelbase_dia
            vectorTheta = (np.pi-theta)/2 - (np.pi/2-perpendicular)
            vectorDistanceWithoutR = np.sin(theta)/np.sin((np.pi-theta)/2)

            out[i, 1] = out[i-1, 1] + vectorDistanceWithoutR*leftR*np.cos(vectorTheta)
            out[i, 2] = out[i-1, 2] + vectorDistanceWithoutR*leftR*np.sin(vectorTheta)
            out[i, 3] = out[i-1, 3] + vectorDistanceWithoutR*rightR*np.cos(vectorTheta)
            out[i, 4] = out[i-1, 4] + vectorDistanceWithoutR*rightR*np.sin(vectorTheta)

    return out


def angle_between(lx, ly, rx, ry):
    delta_x = lx-rx
    delta_y = ly-ry
    angle = 0
    if delta_x == 0:
        angle = np.pi/2
    else:
        angle = np.arctan(delta_y/delta_x)

    if delta_y>0:
        if delta_x>0:
            return angle
        else:
            return np.pi-angle
    else:
        if delta_x>0:
            return -angle
        else:
            return angle-np.pi

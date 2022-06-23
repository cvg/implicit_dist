import numpy as np
import matplotlib.pyplot as plt

def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

def angle(v1,v2):
    return np.rad2deg(np.arccos(np.clip(v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))

def skew(v):
    return np.array([[0,-v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def plot_1d_radial_reprojs(R,t,pp,points3D,width,height,**kwargs):
    
    for X in points3D:        
        z = (R @ X + t)[0:2]

        # pp + alpha * z = d
        # lambda = (d-pp)/z
        alphas = [-pp[0]/z[0], -pp[1]/z[1], (width-pp[0])/z[0], (height-pp[1])/z[1]]
        alpha = min([alpha for alpha in alphas if alpha > 0])

        plt.plot([pp[0], pp[0]+alpha*z[0]], [pp[1], pp[1]+alpha*z[1]],**kwargs)
        

def apply_opencv_distortion(Xcam,params):
    x = []

    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    k = params[4:]

    K = np.array([[fx,0,cx], [0, fy, cy], [0,0,1]])

    for Z in Xcam:

        r = np.linalg.norm(Z[0:2])
        z = Z[2]
                
        if(r > 1e-8):
            theta = np.arctan2(r,z)
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta6 * theta2        
            theta_d = theta * (1.0 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8)
            x_proj = np.r_[(theta_d/r) * Z[0:2], 1.0]
        else:
            x_proj = np.r_[Z[0:2], 1.0]
        
        x_proj = K @ x_proj
        x.append(x_proj[0:2] / x_proj[2])        
    return np.array(x)

def plot_camera(ax,q,t,scale=1, color='r'):
    R = qvec2rotmat(q)
    c = -R.T @ t
    ax.quiver(c[0],c[1],c[2], scale*R[0,0], scale*R[0,1], scale*R[0,2], color=color)
    ax.quiver(c[0],c[1],c[2], scale*R[1,0], scale*R[1,1], scale*R[1,2], color=color)
    ax.quiver(c[0],c[1],c[2], scale*R[2,0], scale*R[2,1], scale*R[2,2], color=color)


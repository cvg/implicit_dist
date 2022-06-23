#include "camera_pose.h"

namespace implicit_dist {
void CameraPose::relative_pose(const CameraPose& pose_i, 
                        const CameraPose& pose_j, 
                        CameraPose& pose_ij) {
    pose_ij.q(pose_j.q() * pose_i.q().conjugate());
    pose_ij.t = pose_j.t - pose_ij.q() * pose_i.t;
}

}; // implicit_dist

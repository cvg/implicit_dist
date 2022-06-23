#ifndef IMPLICIT_DIST_CAMERA_POSE_H_
#define IMPLICIT_DIST_CAMERA_POSE_H_

#include <Eigen/Dense>

namespace implicit_dist {

struct CameraPose {
    Eigen::Vector4d q_vec; // q.w, q.x, q.y, q.z
    Eigen::Vector3d t;

    // Constructors (Defaults to identity camera)
    CameraPose() : q_vec(0.0, 0.0, 0.0, 0.0), t(0.0, 0.0, 0.0) {}

    CameraPose(const Eigen::Vector4d &qq, const Eigen::Vector3d &tt) : q_vec(qq), t(tt) {}
    CameraPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &tt) : t(tt) {
        Eigen::Quaternion<double> qq(R);
        q_vec[0] = qq.w();
        q_vec[1] = qq.x();
        q_vec[2] = qq.y();
        q_vec[3] = qq.z();

    }

    // Helper functions
    inline Eigen::Matrix3d R() const { return q().toRotationMatrix(); }
    inline Eigen::Matrix<double, 3, 4> Rt() const {
        Eigen::Matrix<double, 3, 4> tmp;
        tmp.block<3, 3>(0, 0) = q().toRotationMatrix();
        tmp.col(3) = t;
        return tmp;
    }
    
    void q(Eigen::Quaternion<double> qq) {
        q_vec[1] = qq.x();
        q_vec[2] = qq.y();
        q_vec[3] = qq.z();
        q_vec[0] = qq.w();
    }
    
    inline Eigen::Quaternion<double> q() const { return Eigen::Quaternion<double>(q_vec[0], q_vec[1], q_vec[2], q_vec[3]); };

    inline Eigen::Vector3d rotate(const Eigen::Vector3d &p) const { return q() * p; }
    inline Eigen::Vector3d derotate(const Eigen::Vector3d &p) const { return q().conjugate() * t; }
    inline Eigen::Vector3d apply(const Eigen::Vector3d &p) const { return rotate(p) + t; }
    inline CameraPose inv() const { CameraPose result; CameraPose::relative_pose(*this, CameraPose(), result); return result; }

    // calculate the pose R_ij: x_j = R_ij * x_i
    static void relative_pose(const CameraPose& pose_i, 
                                const CameraPose& pose_j, 
                                CameraPose& pose_ij);

};

};

#endif // IMPLICIT_DIST_CAMERA_POSE_H_
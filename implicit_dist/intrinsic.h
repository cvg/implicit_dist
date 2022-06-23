#ifndef IMPLICIT_DIST_INTRINSIC_H_
#define IMPLICIT_DIST_INTRINSIC_H_

#include <Eigen/Dense>
#include "camera_pose.h"
#include <vector>
#include "cost_matrix.h"
namespace implicit_dist {

    // The intrinsic calibration is represented by the principal point
    // and pairs of (r_i, f_i)
    struct IntrinsicCalib {
        std::vector<std::pair<double,double>> r_f; // For undistortion we have pairs of r and f (sorted by r)
        std::vector<std::pair<double,double>> theta_r; // For distortion we have pairs of theta and r  (sorted by theta)
        Eigen::Vector2d pp;
    };

    // Adaptively select lambda to balance radial/tangential component of reprojection errors
    // We truncate errors where larger than max_error (component-wise)
    IntrinsicCalib calibrate(const std::vector<Eigen::Vector2d> &points2D, 
                                const std::vector<Eigen::Vector3d> &points3D,
                                const CostMatrix &cost_matrix, const Eigen::Vector2d &pp,
                                const CameraPose &pose,
                                const double max_error = 2.0);

    IntrinsicCalib calibrate_multi(const std::vector<std::vector<Eigen::Vector2d>> &points2D, 
                                const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                const CostMatrix &cost_matrix, const Eigen::Vector2d &pp,
                                const std::vector<CameraPose> &pose,
                                const double max_error = 2.0);
    
    // Solve the calibration with a given lambda (trade-off parameter)
    IntrinsicCalib calibrate_fix_lambda(const std::vector<Eigen::Vector2d> &points2D, 
                                const std::vector<Eigen::Vector3d> &points3D,
                                const CostMatrix &cost_matrix, const Eigen::Vector2d &pp,
                                const CameraPose &poses, double lambda);

    IntrinsicCalib calibrate_fix_lambda_multi(const std::vector<std::vector<Eigen::Vector2d>> &points2D, 
                                const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                const CostMatrix &cost_matrix, const Eigen::Vector2d &pp,
                                const std::vector<CameraPose> &poses, double lambda);

    std::vector<Eigen::Vector3d> undistort(const std::vector<Eigen::Vector2d> &points2D, const IntrinsicCalib& calib);
    std::vector<Eigen::Vector2d> distort(const std::vector<Eigen::Vector3d> &points3D, const IntrinsicCalib& calib);
}

#endif
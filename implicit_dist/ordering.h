#ifndef IMPLICIT_DIST_ORDERING_H_
#define IMPLICIT_DIST_ORDERING_H_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "camera_pose.h"
#include "cost_matrix.h"
#include <limits>
#include <vector>


namespace implicit_dist {

CameraPose camposeco_iccv15_estimate_t3(const std::vector<Eigen::Vector2d> &points2D, 
                                const std::vector<Eigen::Vector3d> &points3D,
                                const CameraPose &initial_pose,
                                const size_t max_pairs = 120);


CameraPose camposeco_iccv15_optimize(const std::vector<Eigen::Vector2d> &points2D, 
                                const std::vector<Eigen::Vector3d> &points3D,
                                const CameraPose &initial_pose,
                                const size_t max_pairs = 120);

std::vector<CameraPose> camposeco_iccv15_optimize_multi(
                                const std::vector<std::vector<Eigen::Vector2d>> &points2D, 
                                const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                const std::vector<CameraPose> &initial_pose,
                                const size_t max_pairs = 120);

void get_pairs(std::vector<std::tuple<double, double, double, int>>& r_rho_z,
                                std::vector<std::tuple<double, size_t, size_t>>& rs_pairs,
                                double min_rho,
                                size_t max_pairs);
}

#endif

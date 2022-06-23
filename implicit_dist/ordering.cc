#include "ordering.h"
#include "cost_functions.h"
#include <tuple>
namespace implicit_dist {

CameraPose camposeco_iccv15_estimate_t3(const std::vector<Eigen::Vector2d> &points2D, 
                                const std::vector<Eigen::Vector3d> &points3D,
                                const CameraPose &initial_pose,
                                const size_t max_pairs) {
    
    const double min_rho = 0.001; // TODO ?
    

    // Setup correspondences
    std::vector<std::tuple<double, double, double, int>> r_rho_z;
    r_rho_z.reserve(points2D.size());
    for (size_t k = 0; k < points2D.size(); ++k) {
        Eigen::Vector3d Z = initial_pose.apply(points3D[k]);
        const double r = points2D[k].norm();
        const double rho = Z.topRows<2>().norm();
        r_rho_z.emplace_back(std::make_tuple(r, rho, Z[2], r_rho_z.size()));
    }
    

    // Create all valid pairs
    std::vector<std::tuple<double, size_t, size_t>> rs_pairs;
    get_pairs(r_rho_z, rs_pairs, max_pairs, min_rho);

    size_t max_k = std::min(rs_pairs.size(), max_pairs);

    std::vector<double> I_left;
    std::vector<double> I_right;
    std::vector<double> candidate_t3;

    for (size_t k = 0; k < max_k; ++k) {        
        const std::tuple<double, double, double, int> &v_i = r_rho_z[std::get<1>(rs_pairs[k])];
        const std::tuple<double, double, double, int> &v_j = r_rho_z[std::get<2>(rs_pairs[k])];
        
        const double r_i = std::get<0>(v_i), rho_i = std::get<1>(v_i), z_i = std::get<2>(v_i);
        const double r_j = std::get<0>(v_j), rho_j = std::get<1>(v_j), z_j = std::get<2>(v_j);
        
        const double I = (z_j * rho_i - z_i * rho_j) / (rho_i - rho_j);

        if ( (r_i - r_j) * (rho_i - rho_j) > 0) {
            I_left.push_back(I);
        } else {
            I_right.push_back(I);
        }
        candidate_t3.push_back(I);
    }


    double best_score = 100000.0;
    double best_t3 = 0.0;

    for (double t3 : candidate_t3) {
        double score = 0.0;
        for (double I : I_left) {
            if (t3 > I) {
                score += t3 - I;
            }
        }
        for (double I : I_right) {
            if (t3 < I) {
                score += I - t3;
            }
        }
        if (score < best_score) {
            best_score = score;
            best_t3 = t3;
        }
    }

    CameraPose new_pose = initial_pose;
    new_pose.t(2) -= best_t3;
    return new_pose;
}



CameraPose camposeco_iccv15_optimize(const std::vector<Eigen::Vector2d> &points2D, 
                                const std::vector<Eigen::Vector3d> &points3D,
                                const CameraPose &initial_pose,
                                const size_t max_pairs) {
    
    std::vector<CameraPose> output;
    const std::vector<CameraPose> initial_poses = {initial_pose};
    
    output = camposeco_iccv15_optimize_multi({points2D}, {points3D}, initial_poses, max_pairs);
    return output[0];

}


std::vector<CameraPose> camposeco_iccv15_optimize_multi(
                                const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                const std::vector<CameraPose> &initial_poses,
                                const size_t max_pairs) {
    const size_t num_cams = points2D.size();
    const double min_rho = 0.001; // TODO ?


    size_t num_total_pts = 0;
    for (size_t cam_k = 0; cam_k < num_cams; ++cam_k) {
        num_total_pts += points2D[cam_k].size();
    }

    // Setup correspondences
    std::vector<std::tuple<double, double, double, int>> r_rho_z;
    std::vector<size_t> cam_idx;
    r_rho_z.reserve(num_total_pts);
    cam_idx.reserve(num_total_pts);
    
    for (size_t cam_k = 0; cam_k < num_cams; ++cam_k) {
        const CameraPose &pose = initial_poses[cam_k];
        for (size_t pt_k = 0; pt_k < points2D[cam_k].size(); ++pt_k) {
            Eigen::Vector3d Z = pose.apply(points3D[cam_k][pt_k]);
            const double r = points2D[cam_k][pt_k].norm();
            const double rho = Z.topRows<2>().norm();
            r_rho_z.emplace_back(std::make_tuple(r, rho, Z[2], r_rho_z.size()));
            cam_idx.push_back(cam_k);
        }
    }
    

    // Create all valid pairs
    std::vector<std::tuple<double, size_t, size_t>> rs_pairs;
    get_pairs(r_rho_z, rs_pairs, max_pairs, min_rho);

    // // Sort them according to the radius difference
    // std::sort(rs_pairs.begin(), rs_pairs.end());
    size_t max_k = std::min(rs_pairs.size(), max_pairs);

    ceres::Problem problem;
    ceres::LossFunction *loss = new ceres::HuberLoss(0.001);
    std::vector<double> t3(num_cams);
    for (size_t k = 0; k < num_cams; ++k) {
        t3[k] = 0.0;
    }

    for (size_t k = 0; k < max_k; ++k) {
        const std::tuple<double, double, double, int> &v_i = r_rho_z[std::get<1>(rs_pairs[k])];
        const std::tuple<double, double, double, int> &v_j = r_rho_z[std::get<2>(rs_pairs[k])];
        const size_t cam0 = cam_idx[std::get<1>(rs_pairs[k])];
        const size_t cam1 = cam_idx[std::get<2>(rs_pairs[k])];
        
        const double r_i = std::get<0>(v_i), rho_i = std::get<1>(v_i), z_i = std::get<2>(v_i);
        const double r_j = std::get<0>(v_j), rho_j = std::get<1>(v_j), z_j = std::get<2>(v_j);
        
        // Constraint is given by the sign of
        //  I = (z_j - t3[cam1]) * rho_i - (z_i - t3[cam0]) * rho_j  

        // if we should have I < 0 and penalize I > 0
        bool less_than_zero = (r_i - r_j) * (rho_i - rho_j) > 0;

        if (cam0 == cam1) {
            ceres::CostFunction *cost = OrderingCostSingleCamera::CreateCost(z_i, rho_i, z_j, rho_j, less_than_zero);
            problem.AddResidualBlock(cost, loss, &(t3[cam0]));
        } else {
            ceres::CostFunction *cost = OrderingCostTwoCameras::CreateCost(z_i, rho_i, z_j, rho_j, less_than_zero);
            problem.AddResidualBlock(cost, loss, &(t3[cam0]), &(t3[cam1]));
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true; // true if you want more debug output
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::vector<CameraPose> output;
    for (size_t k = 0; k < num_cams; ++k) {
        CameraPose pose;
        // pose.R = initial_poses[k].R;
        pose.q(initial_poses[k].q());
        pose.t = initial_poses[k].t;
        pose.t(2) -= t3[k];
        output.push_back(pose);
    }
    return output;
}

void get_pairs(std::vector<std::tuple<double, double, double, int>>& r_rho_z,
                                std::vector<std::tuple<double, size_t, size_t>>& rs_pairs, 
                                double min_rho,
                                size_t max_pairs) {
    size_t num_total_pts = r_rho_z.size();

    std::sort(r_rho_z.begin(), r_rho_z.end());
    std::set<std::tuple<double, size_t, size_t>> rs_pairs_set;
    for (int i = 0; i < num_total_pts; i++) {
        int smallerIdx = std::get<3>(r_rho_z[i]);
        for (int j = i + 1; j < num_total_pts; j++) {
            int largerIdx = std::get<3>(r_rho_z[j]);

            if (std::abs(std::get<1>(r_rho_z[smallerIdx]) - std::get<1>(r_rho_z[largerIdx])) <= min_rho)
                continue;

            double diff = std::get<0>(r_rho_z[largerIdx]) - std::get<0>(r_rho_z[smallerIdx]);

            if (rs_pairs_set.size() >= max_pairs && std::get<0>(*rs_pairs_set.rbegin()) <= diff)
                break;

            rs_pairs_set.insert(std::make_tuple(diff, smallerIdx, largerIdx));
            if (rs_pairs_set.size() > max_pairs) {
                auto ite = rs_pairs_set.end();
                ite--;
                rs_pairs_set.erase(ite);
            }
        }
    }

    rs_pairs.resize(rs_pairs_set.size());
    int counter = 0;
    for (auto ite = rs_pairs_set.begin(); ite != rs_pairs_set.end(); ++ite) {
        rs_pairs[counter] = *ite;
        ++counter;
    }
}

}
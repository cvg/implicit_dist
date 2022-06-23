#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <ceres/ceres.h>

#include "cost_functions.h"
#include "cost_matrix.h"
#include "bundle_adjustment.h"
#include "intrinsic.h"
#include "utils.h"

namespace implicit_dist {
    
ceres::LossFunction* setup_loss_function_ba(BundleAdjustmentOptions::LossFunction loss_func, double loss_scale) {
    switch(loss_func) {
        case BundleAdjustmentOptions::HUBER:
            return new ceres::HuberLoss(loss_scale);                
        case BundleAdjustmentOptions::CAUCHY:
            return new ceres::CauchyLoss(loss_scale);                
        case BundleAdjustmentOptions::TRIVIAL:
        default:
            return nullptr;                
    }
}

void bundle_adjustment(std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                std::vector<Eigen::Vector3d> &points3D,
                                std::vector<std::vector<int>> &pointsInd,
                                CostMatrix &cost_matrix, const CostMatrixOptions& cm_opt,
                                std::vector<CameraPose> &poses, Eigen::Vector2d &pp, 
                                BundleAdjustmentOptions ba_opt) {

    if (ba_opt.upgrade_result) {
        std::cout << "pose upgrade for " << points2D.size() << " images" << std::endl;
        // First, upgrade the pose estimation
        int num_cams = points2D.size();
        // convert to suitable data structure for pose refinement
        std::vector<std::vector<Eigen::Vector3d>> points3D_sep(num_cams);
        for (int i = 0; i < num_cams; i++) {
            points3D_sep[i].resize(points2D[i].size());
            for (int j = 0; j < points2D[i].size(); j++) {
                points3D_sep[i][j] = points3D[pointsInd[i][j]];
            }
        }

        poses = pose_refinement_multi(points2D, points3D_sep, cost_matrix, pp, poses, ba_opt);
    }

    if (ba_opt.filter_result) {
        std::cout << "ba_opt.filter_result" << std::endl;
        filter_result_ba(points2D, points3D, pointsInd, poses, pp, ba_opt);
    }

    // exlucde points with too little occurrence in images
    std::vector<int> occur_count(points3D.size(), 0);
    for (int i = 0; i < pointsInd.size(); i++) {
        for (int j = 0; j < pointsInd[i].size(); j++) {
            occur_count[pointsInd[i][j]] += 1;
        }
    }

    int counter = 0;
    for (int i = 0; i < pointsInd.size(); i++) {
        auto ite_2D = points2D[i].begin();
        auto ite_ind = pointsInd[i].begin();
        while (ite_ind != pointsInd[i].end()) {
            if (occur_count[*ite_ind] < ba_opt.min_curr_num) {
                points2D[i].erase(ite_2D);
                pointsInd[i].erase(ite_ind);
                counter++;
            } else {
                ite_2D++;
                ite_ind++;
            }
        }
        for (int j = 0; j < pointsInd[i].size(); j++) {
            if (occur_count[pointsInd[i][j]] < ba_opt.min_curr_num)
                std::cout << "exlusion error!" << std::endl;
        }
    }
    
    cost_matrix = build_cost_matrix_multi(points2D, cm_opt, pp);
    
    std::vector<Eigen::Quaterniond> qs;
    std::vector<Eigen::Vector3d> ts;

    int n_img = points2D.size();
    for (size_t k = 0; k < n_img; ++k) {
        qs.emplace_back(poses[k].q());
        ts.emplace_back(poses[k].t);
    }
    
    std::vector<std::vector<Eigen::Vector2d>> points2D_center = points2D;
    for (size_t cam_k = 0; cam_k < n_img; ++cam_k) {
        for (size_t i = 0; i < points2D[cam_k].size(); ++i) {
            points2D_center[cam_k][i] -= pp;
        }
    }

    std::cout << "excluded point number: " << counter << std::endl;

    std::cout << "start bundle adjustment for " << points2D.size() << " images" << std::endl;
    // Then, do bundle adjustment
    int ite = 0;
    while (ite < ba_opt.max_ite_num) {
        
        std::cout << "ite number: " << ite << std::endl;
        double ratio = bundle_adjustment_inner(points2D_center, points3D, pointsInd, cost_matrix, qs, ts, ba_opt);
        ite++;

        std::cout << "optimize_projection done, decrease ratio: " << ratio << std::endl;
        if (ratio < ba_opt.stop_ratio) {
            std::cout << "ratio < " << ba_opt.stop_ratio << ", BA terminated" << std::endl;
            break;
        }
    }
    
    for (size_t k = 0; k < n_img; ++k) {
        poses[k].q(qs[k]);
        poses[k].t = ts[k];
    }
}


double bundle_adjustment_inner(const std::vector<std::vector<Eigen::Vector2d>> &points2D_center,
                                std::vector<Eigen::Vector3d> &points3D,
                                const std::vector<std::vector<int>> &pointsInd,
                                const CostMatrix &cost_matrix,
                                std::vector<Eigen::Quaterniond> &qs, std::vector<Eigen::Vector3d> &ts,
                                BundleAdjustmentOptions ba_opt) {

    size_t n_img = points2D_center.size();
    std::vector<Eigen::Vector3d> points3D_new = points3D;


    ceres::Problem problem;

    // Radial reprojection error
    ceres::LossFunction* loss_function_radial = setup_loss_function_ba(ba_opt.loss_radial, ba_opt.loss_scale_radial);    

    for (size_t cam_k = 0; cam_k < n_img; ++cam_k) {
        for (size_t i = 0; i < points2D_center[cam_k].size(); ++i) {
            ceres::CostFunction* reg_cost = BARadialReprojError::CreateCost(points2D_center[cam_k][i]);
            problem.AddResidualBlock(reg_cost, loss_function_radial, qs[cam_k].coeffs().data(), ts[cam_k].data(), points3D_new[pointsInd[cam_k][i]].data());
        }
    }

    // This is used for setting up the dynamic parameter vectors
    // (workaround for having duplicate parameters in the blocks)
    std::vector<std::vector<double*>> params(cost_matrix.pt_index.size());
    
    ceres::LossFunction* loss_function_dist = setup_loss_function_ba(ba_opt.loss_dist, ba_opt.loss_scale_dist);        

    // Implicit distortion cost (the cost matrix, regularization)
    for (size_t i = 0; i < cost_matrix.pt_index.size(); ++i) {
        ceres::CostFunction* reg_cost = BACostMatrixRowCost::CreateCost(
            points2D_center, pointsInd, points3D, points3D_new, cost_matrix.pt_index[i], cost_matrix.cam_index[i], cost_matrix.values[i], qs, ts, params[i]);

        problem.AddResidualBlock(reg_cost, loss_function_dist, params[i]);
    }

    // Setup parameterizations and constant parameter blocks
    for (size_t k = 0; k < n_img; ++k) {
        double *q = qs[k].coeffs().data();

        problem.SetParameterization(q, new ceres::EigenQuaternionParameterization());

        if (pointsInd[k].size() != points2D_center[k].size()) {
            std::cout << "size error" << std::endl;
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 5;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = ba_opt.verbose; // true if you want more debug output
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    points3D = points3D_new;

    // return the decrease ratio
    return (summary.initial_cost - summary.final_cost) / double(summary.initial_cost);
}

void filter_result_ba(std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                std::vector<Eigen::Vector3d> &points3D,
                                std::vector<std::vector<int>> &pointsInd,
                                std::vector<CameraPose>& poses, Eigen::Vector2d &pp, 
                                BundleAdjustmentOptions ba_opt) {
    
    std::vector<std::vector<Eigen::Vector3d>> points3D_sep(points2D.size());
    for (int i = 0; i < points2D.size(); i++) {
        points3D_sep[i].resize(points2D[i].size());
        for (int j = 0; j < points2D[i].size(); j++) {
            points3D_sep[i][j] = points3D[pointsInd[i][j]];
        }
    }
    
    std::vector<std::vector<double>> fs_diff;
    calculate_fmed_diff(points2D, points3D_sep, poses, pp, fs_diff);

    int counter = 0;    
    for (int i = 0; i < points2D.size(); i++) {
        int ori_size = points2D[i].size();
        exclude_outliers(fs_diff[i], points2D[i], pointsInd[i], true, ba_opt.filter_thres);
        counter += ori_size - points2D[i].size();
    }
    std::cout << "filtering number: " << counter << std::endl;
}

}
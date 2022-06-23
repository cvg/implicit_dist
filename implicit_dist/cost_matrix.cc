#include "cost_matrix.h"
#include <random>
#include <iostream>
namespace implicit_dist {

int get_next(const std::vector<std::tuple<double, int, int>> &rs, int i, const CostMatrixOptions &opt) {
    int j = i+1;
    if (j >= rs.size()) {
        return -1;
    }
    double dr = std::abs(std::get<0>(rs[i]) - std::get<0>(rs[j]));
    while (dr < opt.min_delta_r) {
        j++;
        if (j >= rs.size())
            return -1;        
        dr = std::abs(std::get<0>(rs[i]) - std::get<0>(rs[j])); 
    }    
    if (dr > opt.max_delta_r)
        return -1;
    return j;
}

int get_previous(const std::vector<std::tuple<double, int, int>> &rs, int i, const CostMatrixOptions &opt) {
    int j = i-1;
    if (j < 0) {
        return -1;
    }    

    double dr = std::abs(std::get<0>(rs[i]) - std::get<0>(rs[j]));
    while (dr < opt.min_delta_r) {
        j--;
        if (j < 0)
            return -1;
        dr = std::abs(std::get<0>(rs[i]) - std::get<0>(rs[j]));         
    }
    if (dr > opt.max_delta_r)
        return -1;

    return j;
}

std::vector<int> get_neighbors(const std::vector<std::tuple<double, int, int>> &rs, int i, int k, const CostMatrixOptions &opt) {
    std::vector<int> ind;
    
    int k_left = std::floor(k / 2.0);
    int k_right = std::ceil(k / 2.0);

    int i0 = i;
    int i_left = i;
    for (size_t j = 0; j < k_left; ++j) {
        i0 = get_previous(rs, i0, opt);
        i_left = i0;
        if (i0 == -1) {
            break; // not enough neighbors
        }
        if (i0 > 0) {
            i0--;                
        }
        ind.push_back(i0);
    }
    i0 = i;

    // if left side is not enough, take more from the right side
    k_right = k - ind.size();
    for (size_t j = 0; j < k_right; ++j) {
        i0 = get_next(rs, i0, opt);
        if (i0 == -1) {
            break; // not enough neighbors
        }
        if (i0 < rs.size()-1) {
            i0++; 
        }
        ind.push_back(i0);
    }

    // if right side is not enough, take more from the left side
    if (ind.size() < k) {
        i0 = i_left;
        k_left = k - ind.size();
        for (size_t j = 0; j < k_left; ++j) {
            i0 = get_previous(rs, i0, opt);
            if (i0 == -1) {
                break; // not enough neighbors
            }
            if (i0 > 0) {
                i0--;                
            }
            ind.push_back(i0);
        }
    }

    return ind;
}

CostMatrix build_cost_matrix(const std::vector<Eigen::Vector2d> &pts,
                        const CostMatrixOptions &options,
                        const Eigen::Vector2d &pp) {
    return build_cost_matrix_multi({pts}, options, pp);
}
CostMatrix build_cost_matrix_multi(
                        const std::vector<std::vector<Eigen::Vector2d>> &pts,
                        const CostMatrixOptions &options,
                        const Eigen::Vector2d &pp) {
    srand(0);
    CostMatrix cm;
    const size_t num_cams = pts.size();
    std::vector<std::tuple<double, int, int>> rs;
    
    for (size_t cam_k = 0; cam_k < num_cams; cam_k++) {
        for (size_t pt_k = 0; pt_k < pts[cam_k].size(); pt_k++) {
            const double r = (pts[cam_k][pt_k]-pp).norm();
            rs.emplace_back(r, pt_k, cam_k);
        }
    }
    std::sort(rs.begin(), rs.end());
    
    if (options.use_subset) {
        std::vector<std::tuple<double, int, int>> rs_sub;
        
        // start with a little distance from image center
        int idx_small = std::upper_bound(rs.begin(), rs.end(), std::make_tuple(options.min_delta_r, 0, 0)) - rs.begin();

        double curr_r = std::get<0>(rs[idx_small + 1]);
        rs_sub.push_back(rs[idx_small + 1]);
        for (int i = idx_small + 1; i < rs.size(); i++) {
            if (std::get<0>(rs[i]) - curr_r > options.subset_interval) {
                curr_r = std::get<0>(rs[i]);
                rs_sub.push_back(rs[i]);                
            }
        }
        rs = rs_sub;
    }

    cm.pt_index.clear();
    cm.cam_index.clear();
    cm.values.clear();
    
    cm.pt_index.reserve(rs.size());
    cm.cam_index.reserve(rs.size());
    cm.values.reserve(rs.size());

    for (int k = 0; k < rs.size(); ++k) {            
        std::vector<int> knn;
        knn = get_neighbors(rs, k, options.poly_knn, options);

        if (knn.size() < 2) {
            continue; // not enough points to fit polynomial
        }

        // establish matrix for solving least square
        Eigen::MatrixXd A(knn.size() + 1, 2);
        Eigen::VectorXd rvec(2);
        A.col(0).setOnes(); 
        rvec(0) = 1;
        for (int d = 1; d < 2; d++) {
            for (int j = 0; j < knn.size(); j++) {
                A(j + 1, d) = A(j + 1, d - 1) * std::get<0>(rs.at(knn.at(j)));
            }
            A(0, d) = A(0, d - 1) * std::get<0>(rs[k]); 
            rvec(d) = rvec(d - 1)  * std::get<0>(rs[k]);
        }
        
        Eigen::MatrixXd AtA = (A.transpose() * A);
        Eigen::VectorXd coeffs = rvec.transpose() * AtA.inverse() * A.transpose();

        cm.pt_index.push_back({});
        cm.cam_index.push_back({});
        cm.values.push_back({});
        
        // Add point of interest
        cm.pt_index.back().push_back(std::get<1>(rs[k]));
        cm.cam_index.back().push_back(std::get<2>(rs[k]));
        cm.values.back().push_back(coeffs(0) - 1.0);
        
        // Add other points
        for (size_t i = 0; i < knn.size(); ++i) {
            cm.pt_index.back().push_back(std::get<1>(rs.at(knn.at(i))));
            cm.cam_index.back().push_back(std::get<2>(rs.at(knn.at(i))));
            cm.values.back().push_back(coeffs(i + 1));
        }
    
    }
    return cm;
}


}

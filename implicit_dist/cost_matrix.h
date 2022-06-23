#ifndef IMPLICIT_DIST_COST_MATRIX_H_
#define IMPLICIT_DIST_COST_MATRIX_H_

#include <limits>
#include <vector>
#include <tuple>
#include <Eigen/Dense>

namespace implicit_dist {


    struct CostMatrixOptions {
        double min_delta_r = 1.0;
        double max_delta_r = 100.0;

        // number of neighbors to include in the poly fitting
        int poly_knn = 5;

        // if we should resample the image points so that it results in a more reasonable intrinsics calibration
        bool use_subset = false;
        double subset_interval = 0.5; // the distance between two sampling points

        CostMatrixOptions clone() const {
            CostMatrixOptions copy = *this;
            return copy;
        }
    };



    class CostMatrix {
    public:        
        // pt_index:    index of the keypoint
        // cam_index:   contains the index of the camera
        // values:      concrete values stored in C (at the position index)
        std::vector<std::vector<int>> pt_index;
        std::vector<std::vector<int>> cam_index;
        std::vector<std::vector<double>> values;
    };

    CostMatrix build_cost_matrix(const std::vector<Eigen::Vector2d> &pts,
                                 const CostMatrixOptions &options,
                                 const Eigen::Vector2d &pp = Eigen::Vector2d(0.0,0.0));

    CostMatrix build_cost_matrix_multi(const std::vector<std::vector<Eigen::Vector2d>> &pts,
                                      const CostMatrixOptions &options,
                                      const Eigen::Vector2d &pp = Eigen::Vector2d(0.0,0.0));
        

}

#endif
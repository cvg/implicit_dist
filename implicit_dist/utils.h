#ifndef IMPLICIT_DIST_UTILS_H_
#define IMPLICIT_DIST_UTILS_H_

#include <vector>
#include <Eigen/Dense>
#include "camera_pose.h"

namespace implicit_dist {

    double median(std::vector<double> v);

    void calculate_fmed_diff(const std::vector<std::vector<Eigen::Vector2d>> &points2D, 
                                const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                const std::vector<CameraPose>& poses, 
                                const Eigen::Vector2d &pp,
                                std::vector<std::vector<double>>& fs_diff,
                                int med_sz = 5);

    // items to be points2D, points3D, pointsInd
    template <typename T1, typename T2>
    double exclude_outliers(std::vector<double> dist, std::vector<T1>& pointsItem1, std::vector<T2>& pointsItem2, bool use_threshold = false, double min_thres = 10) {
        double thres = 1.4826 * 3 * median(dist); // 3 times scaled MAD (according to MATLAB)
        if (use_threshold)
            thres = std::max(thres, min_thres); // set the maximum distance to be at at least some value

        auto ite1 = pointsItem1.begin();
        auto ite2 = pointsItem2.begin();
        int counter = 0;
        while (ite1 != pointsItem1.end()) {
            if (dist[counter] > thres) {
                pointsItem1.erase(ite1);
                pointsItem2.erase(ite2);
            }
            else {
                ite1++;
                ite2++;
            }

            counter++;
        }
        return thres;
    }

}

#endif
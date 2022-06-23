
#include <vector>
#include <algorithm>

#include "utils.h"
#include <Eigen/Core>

namespace implicit_dist {
    double median(std::vector<double> v) {
        if (v.size() == 0) {
            return 0.0;
        }
        auto n = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + n, v.begin() + v.size());
        auto med = v[n];
        if (!(v.size() & 1)) { //If the set size is even
            auto max_it = std::max_element(v.begin(), v.begin() + n);
            med = (*max_it + med) / 2.0;
        }
        return med;
    }

    void calculate_fmed_diff(const std::vector<std::vector<Eigen::Vector2d>> &points2D, 
                                const std::vector<std::vector<Eigen::Vector3d>> &points3D,           
                                const std::vector<CameraPose>& poses, 
                                const Eigen::Vector2d &pp,
                                std::vector<std::vector<double>>& fs_diff,
                                int med_sz) {
        
        int num_cams = points2D.size();
        // Calculate the median of the focal length 
        std::vector<std::tuple<double, double, int>> r_f;
            
        // Compute pointwise focal lengths
        int counter = 0;
        for (size_t cam_ind = 0; cam_ind < num_cams; ++cam_ind) {
            for (size_t pt_ind = 0; pt_ind < points2D[cam_ind].size(); ++pt_ind) {
                Eigen::Vector2d z = points2D[cam_ind][pt_ind] - pp;
                Eigen::Vector3d Z = poses[cam_ind].apply(points3D[cam_ind][pt_ind]);
                double f = (z.squaredNorm() * Z[2]) / (Z.topRows<2>().dot(z));
                double r = z.norm();
                r_f.emplace_back(r, f, counter);
                counter++;
            }
        }
        
        std::sort(r_f.begin(), r_f.end());

        int num_pts = r_f.size();

        // apply median filter
        std::vector<double> med_f;
        med_f.reserve(num_pts);

        const int m = med_sz / 2; // we assume med_z is odd

        // TODO: this is an ugly solution but I can't cba to fix it now
        std::vector<double> fs;
        for (int k = 0; k < num_pts; ++k) {
            fs.clear();
            for (int i = std::max(k-m, 0); i < std::min(num_pts, k+m+1); ++i) {
                fs.push_back(std::get<1>(r_f[i]));
            }
            std::sort(fs.begin(), fs.end());

            if (fs.size() % 2 == 0) { // even number of points (do average of center)
                med_f.push_back((fs[fs.size()/2 - 1] + fs[fs.size()/2]) / 2.0);
            } else { // odd number of points (take center)
                med_f.push_back(fs[fs.size()/2]);
            }
        }

        for (size_t k = 0; k < num_pts; ++k) {
            std::get<1>(r_f[k]) = med_f[k];
        }

        // restore the order of r_f
        std::sort(r_f.begin(), r_f.end(), [](std::tuple<double, double, int> a, std::tuple<double, double, int> b) { return std::get<2>(a) < std::get<2>(b); });

        fs_diff.resize(num_cams);
        counter = 0;
        for (int i = 0; i < num_cams; i++) {
            fs_diff[i].resize(points2D[i].size());
            std::vector<Eigen::Vector3d> points3D_cam(points2D[i].size());

            for (int j = 0; j < points2D[i].size(); j++) {
                points3D_cam[j] = poses[i].apply(points3D[i][j]);
            }

            for (int j = 0; j < points2D[i].size(); j++) {
                Eigen::Vector2d x_ph = points3D_cam[j].topRows<2>() / points3D_cam[j][2];
                double f_obs = (points2D[i][j] - pp).norm() / x_ph.dot(points2D[i][j].normalized());
                double f_calc = std::get<1>(r_f[counter]);

                fs_diff[i][j] = std::abs(f_obs - f_calc);
                counter++;
            }
        }
    }

}
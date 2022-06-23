#ifndef IMPLICIT_DIST_COST_FUNCTIONS_H_
#define IMPLICIT_DIST_COST_FUNCTIONS_H_

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <vector>
#include "intrinsic.h"


namespace implicit_dist {

// error for the radial projection (keep x, X fixed, change R, t and principal point)
struct RadialReprojError { 
    RadialReprojError(const Eigen::Vector2d& point2D, const Eigen::Vector3d& point3D) : x(point2D), X(point3D) {}

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec_xy, T* residuals) const {
        Eigen::Quaternion<T> q;
        q.coeffs() << qvec[0], qvec[1], qvec[2], qvec[3];

        Eigen::Matrix<T, 3, 1> Z = q.toRotationMatrix() * X;

        Eigen::Matrix<T, 2, 1> t;
        t << tvec_xy[0], tvec_xy[1];

        Eigen::Matrix<T, 2, 1> z = (Z.template topRows<2>() + t).normalized();

        T alpha = z.dot(x.cast<T>());
        
        residuals[0] = alpha * z(0) - T(x(0));
        residuals[1] = alpha * z(1) - T(x(1));
        return true;
    }

    // Factory function
    static ceres::CostFunction* CreateCost(const Eigen::Vector2d &x, const Eigen::Vector3d &X) {
        return (new ceres::AutoDiffCostFunction<RadialReprojError, 2, 4, 3>(new RadialReprojError(x,X)));
    }

private:
    const Eigen::Vector2d& x;
    const Eigen::Vector3d& X;
};

// cost for a row of cost matrix
struct CostMatrixRowCost {
    typedef ceres::DynamicAutoDiffCostFunction<CostMatrixRowCost, 1> CostMatrixRowCostFunction;

    CostMatrixRowCost(const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                      const std::vector<std::vector<Eigen::Vector3d>> &points3D, 
                      const std::vector<int> &pt_idx,
                      const std::vector<int> &cam_idx,
                      const std::vector<int> &qt_idx,
                      const std::vector<double> &coeffs) : 
                    xs(points2D), Xs(points3D), pt_index(pt_idx), cam_index(cam_idx),
                    weights(coeffs), qt_index(qt_idx) {}

    template <typename T>
    bool operator()(T const* const* qtvec,
                    T* residuals) const {
        
        // qtvec[2*i] = q.coeffs().data(); qtvec[2*i + 1] = t.data()
        residuals[0] = T(0.0);

        for (size_t k = 0; k < pt_index.size(); ++k) {
            // compute fs
            size_t pt_ind = pt_index[k];
            size_t cam_ind = cam_index[k];
            size_t qt_ind = qt_index[k];

            Eigen::Quaternion<T> q;
            q.coeffs() << qtvec[2*qt_ind][0], qtvec[2*qt_ind][1], qtvec[2*qt_ind][2], qtvec[2*qt_ind][3];
            Eigen::Matrix<T,3,1> t;
            t << qtvec[2*qt_ind+1][0], qtvec[2*qt_ind+1][1], qtvec[2*qt_ind+1][2];

            Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
            Eigen::Matrix<T, 3, 1> Z = R * Xs[cam_ind][pt_ind] + t;

            double nx = xs[cam_ind][pt_ind].squaredNorm();
            T f = T(nx) * Z(2) / xs[cam_ind][pt_ind].cast<T>().dot(Z.template topRows<2>());

            residuals[0] += weights[k] * f;
        }
        return true;
    }


    // Factory function
    static ceres::CostFunction* CreateCost(const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                           const std::vector<std::vector<Eigen::Vector3d>> &points3D, 
                                           const std::vector<int> &pt_idx,
                                           const std::vector<int> &cam_idx,
                                           const std::vector<double> &coeffs,
                                           std::vector<Eigen::Quaterniond> &qvec,
                                           std::vector<Eigen::Vector3d> &tvec,
                                           std::vector<double*> &params) {
        
        size_t num_cams = 0;
        std::vector<int> qt_index;
        for (size_t k = 0; k < pt_idx.size(); ++k) {
            // Figure out if this camera has been used before
            bool new_camera = true;
            for (size_t i = 0; i < k; ++i) {
                if (cam_idx[i] == cam_idx[k]) {
                    qt_index.push_back(qt_index[i]);
                    new_camera = false;
                    break;
                }
            }

            if (new_camera) {
                qt_index.push_back(num_cams);
                params.push_back(qvec[cam_idx[k]].coeffs().data());
                params.push_back(tvec[cam_idx[k]].data());
                num_cams++;
            }
        }
        
        CostMatrixRowCostFunction* cost_function = new CostMatrixRowCostFunction(
                new CostMatrixRowCost(points2D, points3D, pt_idx, cam_idx, qt_index, coeffs));

        for (int i = 0; i < num_cams; ++i) {
            cost_function->AddParameterBlock(4);
            cost_function->AddParameterBlock(3);
        }
        cost_function->SetNumResiduals(1);
        return cost_function;
    }

private:
    const std::vector<std::vector<Eigen::Vector2d>> &xs;
    const std::vector<std::vector<Eigen::Vector3d>> &Xs;
    const std::vector<int> &pt_index;
    const std::vector<int> &cam_index;
    const std::vector<double> &weights;
    const std::vector<int> qt_index;
};


// Same as CostMatrixRowCost but optimizing over pointwise focal length
// Cost function for the regularization in task of calibration
struct FocalCostMatrixRowCost { 
    typedef ceres::DynamicAutoDiffCostFunction<FocalCostMatrixRowCost, 1> FocalCostMatrixRowCostFunction;

    FocalCostMatrixRowCost(const std::vector<double> &coeffs) : weights(coeffs) {}

    template <typename T>
    bool operator()(T const* const* fvec,
                    T* residuals) const {
                
        residuals[0] = T(0.0);

        for (size_t k = 0; k < weights.size(); ++k) {            
            residuals[0] += T(weights[k]) * (*(fvec[k]));
        }

        return true;
    }


    // Factory function
    static ceres::CostFunction* CreateCost(const std::vector<int> &pt_idx,
                                           const std::vector<int> &cam_idx,
                                           const std::vector<double> &coeffs,
                                           std::vector<std::vector<double>> &fvec,
                                           std::vector<double*> &params) {
        
        // In contrast to the case of (R,t) optimization, we cannot have duplicate parameters
        // here. So we can simply assume that they are given in the same order as params
        
        for (size_t k = 0; k < pt_idx.size(); ++k) {
            // Figure out if this camera has been used before            
            size_t p_idx = pt_idx[k];
            size_t c_idx = cam_idx[k];
            params.push_back(&(fvec[c_idx][p_idx]));                        
        }
        FocalCostMatrixRowCostFunction* cost_function = new FocalCostMatrixRowCostFunction(new FocalCostMatrixRowCost(coeffs));

        for (size_t k = 0; k < pt_idx.size(); ++k) {
            cost_function->AddParameterBlock(1);
        }
        cost_function->SetNumResiduals(1);

        return cost_function;
    }

private:
    const std::vector<double> &weights;
};

// Observation error in calibration
struct FocalDataCost {
    FocalDataCost(const double pointwise_focal) : fi(pointwise_focal) {}

    template <typename T>
    bool operator()(T const* f,
                    T* residuals) const {
                
        residuals[0] = *f - T(fi);
        return true;
    }

    // Factory function
    static ceres::CostFunction* CreateCost(const double pointwise_focal) {
        return (new ceres::AutoDiffCostFunction<FocalDataCost, 1, 1>(new FocalDataCost(pointwise_focal)));
    }

private:
    const double fi;
};


// Re-implementation of Camposeco 15
struct OrderingCostSingleCamera {
    OrderingCostSingleCamera(const double zi, const double rhoi,
                             const double zj, const double rhoj,
                             bool negative) : 
                             z_i(zi), rho_i(rhoi), z_j(zj), rho_j(rhoj),
                            less_than_zero(negative) {}

    template <typename T>
    bool operator()(T const* t3, T* residuals) const {

        //  I = (z_j - t3[cam1]) * rho_i - (z_i - t3[cam0]) * rho_j  
        T I = (T(z_j) - *t3) * T(rho_i) - (T(z_i) - *t3) * T(rho_j);

        if (less_than_zero) {
            // if we should have I < 0 and penalize I > 0
            residuals[0] = std::max(I, T(0.0));
        } else {
            // otherwise we should have I > 0 and penalize I < 0
            residuals[0] = std::max(-I, T(0.0));
        }
        
        return true;
    }


    // Factory function
    static ceres::CostFunction* CreateCost(const double zi, const double rhoi,
                                           const double zj, const double rhoj,
                                           bool negative) {
        return (new ceres::AutoDiffCostFunction<OrderingCostSingleCamera, 1, 1>(
                new OrderingCostSingleCamera(zi,rhoi,zj,rhoj,negative)));
    }

private:
    const double z_i, rho_i, z_j, rho_j;
    const bool less_than_zero;
};


struct OrderingCostTwoCameras {
    OrderingCostTwoCameras(const double zi, const double rhoi,
                             const double zj, const double rhoj,
                             bool negative) : 
                             z_i(zi), rho_i(rhoi), z_j(zj), rho_j(rhoj),
                            less_than_zero(negative) {}

    template <typename T>
    bool operator()(T const* t_i, T const* t_j, T* residuals) const {

        //  I = (z_j - t3[cam1]) * rho_i - (z_i - t3[cam0]) * rho_j  
        T I = (T(z_j) - *t_j) * T(rho_i) - (T(z_i) - *t_i) * T(rho_j);

        if (less_than_zero) {
            // if we should have I < 0 and penalize I > 0
            residuals[0] = std::max(I, T(0.0));
        } else {
            // otherwise we should have I > 0 and penalize I < 0
            residuals[0] = std::max(-I, T(0.0));
        }
        
        return true;
    }


    // Factory function
    static ceres::CostFunction* CreateCost(const double zi, const double rhoi,
                                           const double zj, const double rhoj,
                                           bool negative) {
        return (new ceres::AutoDiffCostFunction<OrderingCostTwoCameras, 1, 1, 1>(
                new OrderingCostTwoCameras(zi,rhoi,zj,rhoj,negative)));
    }

private:
    const double z_i, rho_i, z_j, rho_j;
    const bool less_than_zero;
};



// // error for the radial projection (keep x, X fixed, change R, t and principal point)
// struct NonParametricReprojError {
//     NonParametricReprojError(const Eigen::Vector2d& point2D, const Eigen::Vector3d& point3D,
//                              const IntrinsicCalib &intrinsic_calib) :
//                              x(point2D), X(point3D), calib(intrinsic_calib), index(0) {}


//     template<typename T>
//     void update_index(T theta) const {
//         // Move to the right
//         while (T(calib.theta_r[index+1].first) < theta && index < calib.theta_r.size()-1) {
//             index++;
//         }
//         // Move to the left
//         while (T(calib.theta_r[index].first) > theta && index > 0) {
//             index--;
//         }
//     }


//     template <typename T>
//     bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
//         Eigen::Quaternion<T> q;
//         q.coeffs() << qvec[0], qvec[1], qvec[2], qvec[3];
//         Eigen::Matrix<T, 3, 1> t;
//         t << tvec[0], tvec[1], tvec[2];

//         Eigen::Matrix<T, 3, 1> Z = q.toRotationMatrix() * X.cast<T>() + t;
//         Eigen::Matrix<T, 2, 1> z(Z(0),Z(1));
//         T rho = z.norm();
//         T theta = ceres::atan2(rho, Z(2));

//         update_index(theta);

//         T theta1 = T(calib.theta_r[index].first);
//         T r1 = T(calib.theta_r[index].second);

//         T theta2 = T(calib.theta_r[index+1].first);
//         T r2 = T(calib.theta_r[index+1].second);

//         T d_theta1 = theta - theta1;
//         T d_theta2 = theta2 - theta;

//         T r = (d_theta1 * r2 + d_theta2 * r1) / (d_theta1 + d_theta2);

//         T alpha = r / rho;

//         residuals[0] = alpha * z(0) + T(calib.pp(0)) - T(x(0));
//         residuals[1] = alpha * z(1) + T(calib.pp(1)) - T(x(1));
//         return true;
//     }

//     // Factory function
//     static ceres::CostFunction* CreateCost(const Eigen::Vector2d &x, const Eigen::Vector3d &X, const IntrinsicCalib &calib) {
//         return (new ceres::AutoDiffCostFunction<NonParametricReprojError, 2, 4, 3>(new NonParametricReprojError(x,X,calib)));
//     }

// private:
//     const Eigen::Vector2d& x;
//     const Eigen::Vector3d& X;
//     const IntrinsicCalib &calib;
//     mutable size_t index;
// };

// Cost for bundle adjustment
// error for the radial projection (keep x, change R, t and X)
struct BARadialReprojError { 
    BARadialReprojError(const Eigen::Vector2d& point2D) : x(point2D) {}

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec_xy, const T* const Xvec, T* residuals) const {
        Eigen::Quaternion<T> q;
        q.coeffs() << qvec[0], qvec[1], qvec[2], qvec[3];

        Eigen::Matrix<T, 3, 1> X;
        X << Xvec[0], Xvec[1], Xvec[2];
        Eigen::Matrix<T, 3, 1> Z = q.toRotationMatrix() * X;

        Eigen::Matrix<T, 2, 1> t;
        t << tvec_xy[0], tvec_xy[1];

        Eigen::Matrix<T, 2, 1> z = (Z.template topRows<2>() + t).normalized();

        T alpha = z.dot(x.cast<T>());
        
        residuals[0] = alpha * z(0) - T(x(0));
        residuals[1] = alpha * z(1) - T(x(1));
        return true;
    }

    // Factory function
    static ceres::CostFunction* CreateCost(const Eigen::Vector2d &x) {
        return (new ceres::AutoDiffCostFunction<BARadialReprojError, 2, 4, 3, 3>(new BARadialReprojError(x)));
    }

private:
    const Eigen::Vector2d& x;
};


// regularization cost for joint optimization (keep x, X fixed, change R, t)
struct BACostMatrixRowCost {
    typedef ceres::DynamicAutoDiffCostFunction<BACostMatrixRowCost, 1>
        BACostMatrixRowCostFunction;
    
    BACostMatrixRowCost(const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                    const std::vector<Eigen::Vector3d> &points3D,
                    const std::vector<std::vector<int>> &pointsInd,
                    const std::vector<int> &pt_idx,
                    const std::vector<int> &cam_idx,
                    const std::vector<int> &qt_idx,
                    const std::vector<double> &coeffs,
                    int num_cams) : 
                    xs(points2D), Xs(points3D), Xs_ind(pointsInd), pt_index(pt_idx), cam_index(cam_idx),
                    weights(coeffs), qt_index(qt_idx), num_cams(num_cams) {};

    template <typename T>
    bool operator()(T const* const* qtvec,
                    T* residuals) const {
        
        // qtvec[2*i] = q.coeffs().data(); qtvec[2*i + 1] = t.data(); qtvec[2*num_cams] = X.data()
        residuals[0] = T(0.0);

        for (size_t k = 0; k < pt_index.size(); ++k) {
            // compute fs
            size_t pt_ind = pt_index[k];
            size_t cam_ind = cam_index[k];
            size_t qt_ind = qt_index[k];

            Eigen::Quaternion<T> q;
            q.coeffs() << qtvec[2*qt_ind][0], qtvec[2*qt_ind][1], qtvec[2*qt_ind][2], qtvec[2*qt_ind][3];
            Eigen::Matrix<T,3,1> t;
            t << qtvec[2*qt_ind+1][0], qtvec[2*qt_ind+1][1], qtvec[2*qt_ind+1][2];

            Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
            Eigen::Matrix<T, 3, 1> Z = R * Xs[Xs_ind[cam_ind][pt_ind]] + t;

            if (k == 0) {
                Eigen::Matrix<T, 3, 1> X;
                X << qtvec[2*num_cams][0], qtvec[2*num_cams][1], qtvec[2*num_cams][2];
                Z = R * X + t;
            }

            double nx = xs[cam_ind][pt_ind].squaredNorm();
            T f = T(nx) * Z(2) / xs[cam_ind][pt_ind].cast<T>().dot(Z.template topRows<2>());

            residuals[0] += weights[k] * f;
        }
        return true;
    }


    // Factory function
    static ceres::CostFunction* CreateCost(const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                            const std::vector<std::vector<int>> &pointsInd,
                                            const std::vector<Eigen::Vector3d> &points3D,
                                            std::vector<Eigen::Vector3d> &points3D_new,
                                            const std::vector<int> &pt_idx,
                                            const std::vector<int> &cam_idx,
                                            const std::vector<double> &coeffs,
                                            std::vector<Eigen::Quaterniond> &qvec,
                                            std::vector<Eigen::Vector3d> &tvec,
                                            std::vector<double*> &params) {
        
        size_t num_cams = 0;
        std::vector<int> qt_index;
        for (size_t k = 0; k < pt_idx.size(); ++k) {
            // Figure out if this camera has been used before
            bool new_camera = true;
            for (size_t i = 0; i < k; ++i) {
                if (cam_idx[i] == cam_idx[k]) {
                    qt_index.push_back(qt_index[i]);
                    new_camera = false;
                    break;
                }
            }

            if (new_camera) {
                qt_index.push_back(num_cams);
                params.push_back(qvec[cam_idx[k]].coeffs().data());
                params.push_back(tvec[cam_idx[k]].data());
                num_cams++;
            }
        }
        // put the point of concern into the parameters
        size_t pt_ind = pt_idx[0];
        size_t cam_ind = cam_idx[0];
        params.push_back(points3D_new[pointsInd[cam_ind][pt_ind]].data());
        
        BACostMatrixRowCostFunction* cost_function = new BACostMatrixRowCostFunction(
                new BACostMatrixRowCost(points2D, points3D, pointsInd, pt_idx, cam_idx, qt_index, coeffs, num_cams));

        for (int i = 0; i < num_cams; ++i) {
            cost_function->AddParameterBlock(4);
            cost_function->AddParameterBlock(3);
        }

        // X
        cost_function->AddParameterBlock(3);
        cost_function->SetNumResiduals(1);
        return cost_function;
    }

private:
    const std::vector<std::vector<Eigen::Vector2d>> &xs;
    const std::vector<Eigen::Vector3d> &Xs;
    const std::vector<std::vector<int>> &Xs_ind;
    const std::vector<int> &pt_index;
    const std::vector<int> &cam_index;
    const std::vector<double> &weights;
    const std::vector<int> qt_index;

    int num_cams;

};
}

#endif

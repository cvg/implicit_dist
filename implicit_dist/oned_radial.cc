#include "oned_radial.h"
#include <ceres/ceres.h>

namespace implicit_dist {

// error for the radial projection (keep x, X fixed, change R, t and principal point)
struct RadialReprojError { 
    RadialReprojError(const Eigen::Vector2d& x, const Eigen::Vector3d& X) : x_(x), X_(X) {}

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const ppvec, T* residuals) const {
        Eigen::Quaternion<T> q;
        q.coeffs() << qvec[0], qvec[1], qvec[2], qvec[3];

        Eigen::Matrix<T, 3, 1> Z = q.toRotationMatrix() * X_;

        Eigen::Matrix<T, 2, 1> t;
        t << tvec[0], tvec[1];

        Eigen::Matrix<T, 2, 1> z = (Z.template topRows<2>() + t).normalized();

        Eigen::Matrix<T, 2, 1> xc;
        xc << T(x_(0)) - ppvec[0], T(x_(1)) - ppvec[1];

        T alpha = z.dot(xc);
        T lambda = ceres::log(T(1.0) + xc.norm());
        
        residuals[0] = lambda * (alpha * z(0) - xc(0));
        residuals[1] = lambda * (alpha * z(1) - xc(1));
        return true;
    }

    // Factory function
    static ceres::CostFunction* CreateCost(const Eigen::Vector2d &x, const Eigen::Vector3d &X) {
        return (new ceres::AutoDiffCostFunction<RadialReprojError, 2, 4, 2, 2>(new RadialReprojError(x, X)));
    }

private:
    const Eigen::Vector2d& x_;
    const Eigen::Vector3d& X_;
};



void pose_refinement_1D_radial(const std::vector<Eigen::Vector2d> &points2D,
                            const std::vector<Eigen::Vector3d> &points3D,
                            CameraPose *pose, Eigen::Vector2d *pp, 
                            PoseRefinement1DRadialOptions opt) {

    std::vector<CameraPose> poses;
    poses.push_back(*pose);
    joint_pose_refinement_1D_radial({points2D}, {points3D}, &poses, pp, opt);
    *pose = poses[0];
}



void joint_pose_refinement_1D_radial(const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                            const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                            std::vector<CameraPose> *poses, Eigen::Vector2d *pp, 
                            PoseRefinement1DRadialOptions opt) {

    const size_t num_cams = points2D.size();
    std::vector<Eigen::Quaterniond> qs;
    std::vector<Eigen::Vector2d> ts;

    for (size_t k = 0; k < num_cams; ++k) {
        qs.emplace_back(poses->at(k).q());
        ts.push_back(poses->at(k).t.topRows<2>());
    }

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(6.0);

    for (size_t cam_k = 0; cam_k < num_cams; ++cam_k) {
        for (size_t i = 0; i < points2D[cam_k].size(); ++i) {
            ceres::CostFunction* reg_cost = RadialReprojError::CreateCost(points2D[cam_k][i], points3D[cam_k][i]);
            problem.AddResidualBlock(reg_cost, loss_function, qs[cam_k].coeffs().data(), ts[cam_k].data(), pp->data());
        }
    }
    
    for (size_t cam_k = 0; cam_k < num_cams; ++cam_k) {
        problem.SetParameterization(qs[cam_k].coeffs().data(), new ceres::EigenQuaternionParameterization());
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = opt.verbose;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (size_t cam_k = 0; cam_k < num_cams; ++cam_k) {
        poses->at(cam_k).q(qs[cam_k]);
        poses->at(cam_k).t.topRows<2>() = ts[cam_k];
    }
}


}

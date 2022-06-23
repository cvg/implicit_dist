#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <map>

#include "implicit_dist/oned_radial.h"
#include "implicit_dist/cost_matrix.h"
#include "implicit_dist/pose_refinement.h"
#include "implicit_dist/ordering.h"
#include "implicit_dist/intrinsic.h"
#include "implicit_dist/bundle_adjustment.h"
#include "implicit_dist/utils.h"


static std::string to_string(const Eigen::MatrixXd& mat){
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

template<typename T>
static std::string to_string(const std::vector<T>& vec){
    std::stringstream ss;
    ss << "[";
    if (vec.size() > 0) {
        for (size_t k = 0; k < vec.size() - 1; ++k) {
            ss << vec.at(k) << ", ";
        }
        ss << vec.at(vec.size()-1);
    }
    ss << "]";
    return ss.str();
}

template<typename T>
static std::string vec_to_string(const std::vector<std::vector<T>>& vec){
    std::stringstream ss;
    ss << "[";
    if (vec.size() > 0) {
        for (size_t k = 0; k < vec.size() - 1; ++k) {
            ss << to_string(vec.at(k)) << ", ";
        }  
        ss << to_string(vec.at(vec.size()-1));
    }
    ss << "]";
    return ss.str();
}


namespace py = pybind11;

using namespace implicit_dist;

CameraPose pose_refinement_wrapper(const std::vector<Eigen::Vector2d> points2D, 
                               const std::vector<Eigen::Vector3d> points3D,
                               const CostMatrix cost_matrix, const Eigen::Vector2d pp,
                               const CameraPose initial_pose, PoseRefinementOptions refinement_opt) {
 
    CameraPose output = pose_refinement(points2D, points3D, cost_matrix, pp, initial_pose, refinement_opt);
    return output;
}

std::vector<CameraPose> pose_refinement_multi_wrapper(const std::vector<std::vector<Eigen::Vector2d>> points2D, 
                    const std::vector<std::vector<Eigen::Vector3d>> points3D,
                    const CostMatrix cost_matrix, const Eigen::Vector2d pp,
                    const std::vector<CameraPose> initial_poses, PoseRefinementOptions refinement_opt) {
    std::vector<CameraPose> output = pose_refinement_multi(points2D, points3D, cost_matrix, pp, initial_poses, refinement_opt);
    return output;
}

CameraPose non_parametric_pose_refinement_wrapper(const std::vector<Eigen::Vector2d> points2D, 
                               const std::vector<Eigen::Vector3d> points3D,
                               const IntrinsicCalib calib,
                               const CameraPose initial_pose, PoseRefinementOptions refinement_opt) {
 
    CameraPose output = non_parametric_pose_refinement(points2D, points3D, calib, initial_pose, refinement_opt);
    return output;
}

py::dict pose_refinement_1D_radial_wrapper(const std::vector<Eigen::Vector2d> points2D, std::vector<Eigen::Vector3d> points3D, CameraPose initial_pose, Eigen::Vector2d initial_pp, const PoseRefinement1DRadialOptions &opt) {
 
    CameraPose pose = initial_pose;
    Eigen::Vector2d pp = initial_pp;
    pose_refinement_1D_radial(points2D, points3D, &pose, &pp, opt);

    py::dict output_dict;
    output_dict["pose"] = pose;
    output_dict["pp"] = pp;

    return output_dict;
}

py::dict joint_pose_refinement_1D_radial_wrapper(const std::vector<std::vector<Eigen::Vector2d>> points2D, 
                        std::vector<std::vector<Eigen::Vector3d>> points3D, 
                        std::vector<CameraPose> initial_poses, 
                        Eigen::Vector2d initial_pp, 
                        const PoseRefinement1DRadialOptions &opt) {

    std::vector<CameraPose> poses = initial_poses;
    Eigen::Vector2d pp = initial_pp;
    joint_pose_refinement_1D_radial(points2D, points3D, &poses, &pp, opt);

    py::dict output_dict;
    output_dict["poses"] = poses;
    output_dict["pp"] = pp;

    return output_dict;
}

py::dict filter_result_pose_refinement_wrapper(std::vector<Eigen::Vector2d> points2D, 
                        std::vector<Eigen::Vector3d> points3D, 
                        CameraPose pose,
                        Eigen::Vector2d pp,
                        const PoseRefinementOptions refinement_opt) {
      
    filter_result_pose_refinement(points2D, points3D, pose, pp, refinement_opt);

    py::dict output_dict;
    output_dict["points2D"] = points2D;
    output_dict["points3D"] = points3D;

    return output_dict;
}

py::dict filter_result_pose_refinement_multi_wrapper(std::vector<std::vector<Eigen::Vector2d>> points2D,
                        std::vector<std::vector<Eigen::Vector3d>> points3D,
                        std::vector<CameraPose> poses, Eigen::Vector2d pp,
                        const PoseRefinementOptions refinement_opt) {
      
    filter_result_pose_refinement_multi(points2D, points3D, poses, pp, refinement_opt);

    py::dict output_dict;
    output_dict["points2D"] = points2D;
    output_dict["points3D"] = points3D;

    return output_dict;
}

py::dict bundle_adjustment_wrapper(std::vector<std::vector<Eigen::Vector2d>> &points2D,
                        std::vector<Eigen::Vector3d> &points3D,
                        std::vector<std::vector<int>> &pointsInd,
                        CostMatrix &cost_matrix, const CostMatrixOptions& cm_opt,
                        std::vector<CameraPose> &poses, Eigen::Vector2d &pp, 
                        BundleAdjustmentOptions ba_opt) {
 
    bundle_adjustment(points2D, points3D, pointsInd, cost_matrix, cm_opt, poses, pp, ba_opt);

    py::dict output_dict;
    output_dict["poses"] = poses;
    output_dict["points2D"] = points2D;
    output_dict["points3D"] = points3D;
    output_dict["pointsInd"] = pointsInd;
    output_dict["cost_matrix"] = cost_matrix;

    return output_dict;
}

IntrinsicCalib calibrate_wrapper(const std::vector<Eigen::Vector2d> points2D, 
                        const std::vector<Eigen::Vector3d> points3D,
                        const CostMatrix cost_matrix, const Eigen::Vector2d pp,
                        const CameraPose pose) {
    IntrinsicCalib calib = calibrate(points2D, points3D, cost_matrix, pp, pose);
    return calib;
}

IntrinsicCalib calibrate_multi_wrapper(const std::vector<std::vector<Eigen::Vector2d>> points2D, 
                        const std::vector<std::vector<Eigen::Vector3d>> points3D,
                        const CostMatrix cost_matrix, const Eigen::Vector2d pp,
                        const std::vector<CameraPose> pose) {
    IntrinsicCalib calib = calibrate_multi(points2D, points3D, cost_matrix, pp, pose);
    return calib;
}


IntrinsicCalib calibrate_fix_lambda_wrapper(const std::vector<Eigen::Vector2d> points2D, 
                        const std::vector<Eigen::Vector3d> points3D,
                        const CostMatrix cost_matrix, const Eigen::Vector2d pp,
                        const CameraPose pose, double lambda) {
    IntrinsicCalib calib = calibrate_fix_lambda(points2D, points3D, cost_matrix, pp, pose, lambda);
    return calib;
}

IntrinsicCalib calibrate_fix_lambda_multi_wrapper(const std::vector<std::vector<Eigen::Vector2d>> points2D, 
                        const std::vector<std::vector<Eigen::Vector3d>> points3D,
                        const CostMatrix cost_matrix, const Eigen::Vector2d pp,
                        const std::vector<CameraPose> poses, double lambda) {
    IntrinsicCalib calib = calibrate_fix_lambda_multi(points2D, points3D, cost_matrix, pp, poses, lambda);
    return calib;
}

std::vector<Eigen::Vector3d> undistort_wrapper(const std::vector<Eigen::Vector2d> points2D, const IntrinsicCalib calib) {
    std::vector<Eigen::Vector3d> output = undistort(points2D, calib);
    return output;
}
std::vector<Eigen::Vector2d> distort_wrapper(const std::vector<Eigen::Vector3d> points3D, const IntrinsicCalib calib) {
    std::vector<Eigen::Vector2d> output = distort(points3D, calib);
    return output;
}

CostMatrix build_cost_matrix_wrapper(const std::vector<Eigen::Vector2d> pts,
                        const CostMatrixOptions options,
                        const Eigen::Vector2d pp = Eigen::Vector2d(0.0,0.0)) {
    CostMatrix cm = build_cost_matrix(pts,options,pp);
    return cm;
}

CostMatrix build_cost_matrix_multi_wrapper(const std::vector<std::vector<Eigen::Vector2d>> pts,
                        const CostMatrixOptions options,
                        const Eigen::Vector2d pp = Eigen::Vector2d(0.0,0.0)) {
    CostMatrix cm = build_cost_matrix_multi(pts,options,pp);
    return cm;
}


PYBIND11_MODULE(pyimplicitdist, m)
{
    m.doc() = "Implicit Distortion";

    py::class_<CameraPose>(m, "CameraPose")
            .def(py::init([](const Eigen::Vector4d &qq, const Eigen::Vector3d &tt) { return new CameraPose(qq, tt); }))
            .def(py::init<>())
            .def_readwrite("q_vec", &CameraPose::q_vec)
            .def_readwrite("t", &CameraPose::t)
            .def("__repr__",
                [](const CameraPose &a) {
                    return "[q: " + to_string(a.q_vec.transpose()) + ", " +
                            "t: " + to_string(a.t.transpose()) + " ]\n";
                }
            );

    py::class_<CostMatrix>(m, "CostMatrix")
            .def(py::init<>())
            .def_readwrite("values", &CostMatrix::values)
            .def_readwrite("pt_index", &CostMatrix::pt_index)
            .def_readwrite("cam_index", &CostMatrix::cam_index)
            .def("__repr__",
                [](const CostMatrix &a) {
                    return "[values: " + vec_to_string(a.values) + "\n" +
                            "pt_index: " + vec_to_string(a.pt_index) +
                            "cam_index: " + vec_to_string(a.cam_index) + " ]\n";
                }
            );

    py::class_<CostMatrixOptions> cm_opt(m, "CostMatrixOptions");
            cm_opt.def(py::init<>())
            .def_readwrite("min_delta_r", &CostMatrixOptions::min_delta_r)
            .def_readwrite("max_delta_r", &CostMatrixOptions::max_delta_r)
            .def_readwrite("poly_knn", &CostMatrixOptions::poly_knn)
            .def_readwrite("use_subset", &CostMatrixOptions::use_subset)
            .def_readwrite("subset_interval", &CostMatrixOptions::subset_interval)          
            .def("clone", &CostMatrixOptions::clone)   
            .def("__repr__",
                [](const CostMatrixOptions &a) {
                    return "[min_delta_r: " + std::to_string(a.min_delta_r) + "\n" +
                            "max_delta_r: " + std::to_string(a.max_delta_r) +
                            "poly_knn: " + std::to_string(a.poly_knn) + "\n" +
                            "use_subset: " + std::to_string(a.use_subset) + " ]\n"+ 
                            "subset_interval: " + std::to_string(a.subset_interval) + " ]\n";
                }
            );


    py::class_<PoseRefinementOptions> pose_ref_opt(m, "PoseRefinementOptions");
    pose_ref_opt.def(py::init<>())
            .def_readwrite("loss_radial", &PoseRefinementOptions::loss_radial)
            .def_readwrite("loss_scale_radial", &PoseRefinementOptions::loss_scale_radial)
            .def_readwrite("loss_dist", &PoseRefinementOptions::loss_dist)       
            .def_readwrite("loss_scale_dist", &PoseRefinementOptions::loss_scale_dist)
            .def_readwrite("verbose", &PoseRefinementOptions::verbose)
            .def("clone", &PoseRefinementOptions::clone)   
            .def("__repr__",
                [](const PoseRefinementOptions &a) {
                    return "[loss_radial: " + std::to_string(a.loss_radial) + "\n" +
                            "loss_scale_radial: " + std::to_string(a.loss_scale_radial) + "\n" +
                            "loss_dist: " + std::to_string(a.loss_dist) + "\n" +
                            "loss_scale_dist: " + std::to_string(a.loss_scale_dist) + "\n" +
                            "verbose: " + std::to_string(a.verbose) + " ]\n";
                }
            );

    py::class_<BundleAdjustmentOptions, PoseRefinementOptions> ba_opt(m, "BundleAdjustmentOptions");
    ba_opt.def(py::init<>())
            .def_readwrite("max_ite_num", &BundleAdjustmentOptions::max_ite_num)
            .def_readwrite("min_curr_num", &BundleAdjustmentOptions::min_curr_num)
            .def_readwrite("stop_ratio", &BundleAdjustmentOptions::stop_ratio)
            .def_readwrite("filter_thres", &BundleAdjustmentOptions::filter_thres)
            .def_readwrite("upgrade_result", &BundleAdjustmentOptions::upgrade_result)
            .def_readwrite("filter_result", &BundleAdjustmentOptions::filter_result)
            .def("clone", &BundleAdjustmentOptions::clone)   
            .def("__repr__",
                [](const BundleAdjustmentOptions &a) {
                    return "[max_ite_num: " + std::to_string(a.max_ite_num) + "\n" + +
                            "min_curr_num: " + std::to_string(a.min_curr_num) + " ]\n"
                            "stop_ratio: " + std::to_string(a.stop_ratio) + "\n" +
                            "filter_thres: " + std::to_string(a.filter_thres) + "\n" +
                            "upgrade_result: " + std::to_string(a.upgrade_result) + "\n" +
                            "filter_result: " + std::to_string(a.filter_result) + "\n";
                }
            );

    py::enum_<PoseRefinementOptions::LossFunction>(pose_ref_opt, "LossFunction")
            .value("TRIVIAL", PoseRefinementOptions::LossFunction::TRIVIAL)
            .value("HUBER", PoseRefinementOptions::LossFunction::HUBER)
            .value("CAUCHY", PoseRefinementOptions::LossFunction::CAUCHY)
            .export_values();

    py::class_<PoseRefinement1DRadialOptions>(m, "PoseRefinement1DRadialOptions")
            .def(py::init<>())
            .def_readwrite("verbose", &PoseRefinement1DRadialOptions::verbose)    
            .def("__repr__",
                [](const PoseRefinement1DRadialOptions &a) {
                    return "[verbose: " + std::to_string(a.verbose) + " ]\n";
                }
            );

    py::class_<IntrinsicCalib>(m, "IntrinsicCalib")
            .def(py::init<>())
            .def_readwrite("r_f", &IntrinsicCalib::r_f)
            .def_readwrite("theta_r", &IntrinsicCalib::theta_r)
            .def_readwrite("pp", &IntrinsicCalib::pp)          
            .def("__repr__",
                [](const IntrinsicCalib &a) {
                    return "[r_f: (2x" + std::to_string(a.r_f.size()) + ")\n" +
                            "theta_r: (2x" + std::to_string(a.theta_r.size()) + ")\n" +                           
                            "pp: " + std::to_string(a.pp(0)) + "," + std::to_string(a.pp(1)) + " ]\n";
                }
            );

    m.def("pose_refinement_1D_radial", &pose_refinement_1D_radial_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("pp"), py::arg("refinement_opt"), "Refines 1D radial camera pose");  
    m.def("joint_pose_refinement_1D_radial", &joint_pose_refinement_1D_radial_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("initial_poses"), py::arg("pp"), py::arg("refinement_opt"), "Refines 1D radial camera pose");  
    
    m.def("pose_refinement", &pose_refinement_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("cost_matrix"), py::arg("pp"), py::arg("initial_pose"), py::arg("refinement_opt"), "Pose refinement using implicit distortion model", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    m.def("pose_refinement_multi", &pose_refinement_multi_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("cost_matrix"), py::arg("pp"), py::arg("initial_poses"), py::arg("refinement_opt"), "Pose refinement using implicit distortion model", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    
    m.def("filter_result_pose_refinement", &filter_result_pose_refinement_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("pose"), py::arg("pp"), py::arg("refinement_opt"), "Filtering for pose refinement result", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("filter_result_pose_refinement_multi", &filter_result_pose_refinement_multi_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("poses"), py::arg("pp"), py::arg("refinement_opt"), "Filtering for pose refinement result", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("bundle_adjustment", &bundle_adjustment_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("pointsInd"), py::arg("poses"), py::arg("cost_matrix"), py::arg("cm_opt"), py::arg("pp"), py::arg("ba_opt"), "Bundle adjustment with possible upgraded and filtering", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("build_cost_matrix", &build_cost_matrix_wrapper, py::arg("points2D"), py::arg("options"), py::arg("pp"), "Constructs the cost matrix for the regularizer.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    m.def("build_cost_matrix_multi", &build_cost_matrix_multi_wrapper,py::arg("points2D"), py::arg("options"), py::arg("pp"),  "Constructs the cost matrix for the regularizer.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  

    m.def("calibrate", &calibrate_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("cost_matrix"), py::arg("pp"), py::arg("poses"), "Recover intrinsic calibration.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    m.def("calibrate_multi", &calibrate_multi_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("cost_matrix"), py::arg("pp"), py::arg("poses"),  "Recover intrinsic calibration from multiple images.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    m.def("calibrate_fix_lambda", &calibrate_fix_lambda_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("cost_matrix"), py::arg("pp"), py::arg("poses"), py::arg("lambda"), "Recover intrinsic calibration.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    m.def("calibrate_fix_lambda_multi", &calibrate_fix_lambda_multi_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("cost_matrix"), py::arg("pp"), py::arg("poses"), py::arg("lambda"), "Recover intrinsic calibration from multiple images.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    
    m.def("undistort", &undistort_wrapper, py::arg("points2D"), py::arg("calib"), "Undistort points using calibration.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    m.def("distort", &distort_wrapper, py::arg("points3D"), py::arg("calib"), "Distort points using calibration.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  

    m.def("camposeco_iccv15_estimate_t3", &camposeco_iccv15_estimate_t3, py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("max_pairs") = 120, "Estimates the forward translation.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    m.def("camposeco_iccv15_optimize", &camposeco_iccv15_optimize, py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("max_pairs") = 120, "Estimates the forward translation.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  
    m.def("camposeco_iccv15_optimize_multi", &camposeco_iccv15_optimize_multi, py::arg("points2D"), py::arg("points3D"), py::arg("initial_poses"), py::arg("max_pairs") = 120, "Estimates the forward translation.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());  

    m.attr("__version__") = std::string("0.0.1");
  
}

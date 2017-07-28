#pragma once

#include <tuple>
#include <vector>
#include <limits>
#include <cmath>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "object_function.h"

using namespace std;
using namespace Eigen;

#define PI (atan(1) * 4)
#define DEPTH_MAX (1e+6)

struct FeaturePoint
{
    //     --> theta, x
    //  |
    //  v
    //  phi, y
    int camera_index;
    Vector2d theta_phi;
};

typedef pair<FeaturePoint, FeaturePoint> FeaturePair;

struct PolyCamera
{
    // y
    //
    // in --> x
    // |
    // v
    // z
    //             2
    //           /   \
    //         3       1
    //           \   /
    //             0
    //
    //   baseline(i, j) = i -> j

    Vector3d position[4];            // camera
    Matrix<Vector3d, 4, 4> baseline; // camera_i, camera_j, xyz
    PolyCamera(const Vector3d position0,
               const Vector3d position1,
               const Vector3d position2,
               const Vector3d position3)
    {
        position[0] = position0;
        position[1] = position1;
        position[2] = position2;
        position[3] = position3;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                baseline(i, j) = position[j] - position[i];
    }
};

//n_frame, n_row, n_col
Tensor<double, 3>
GenerateDeformableSphere(const vector<vector<FeaturePair>> &feature_pair_list,
                         const PolyCamera &ploy_camera,
                         const int n_row,
                         const int n_col,
                         const double depth_constrain_weight,
                         const double first_spatial_smooth_constraint_weight,
                         const double second_spatial_smooth_constraint_weight,
                         const double temporial_smooth_constraint_weight);

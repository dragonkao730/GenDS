#include <tuple>
#include <vector>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

struct Feature
{
    int camera_index;
    Vector2d theta_phi;
};

typedef pair<Feature, Feature> FeaturePair;

struct PolyCamera
{
    Vector3d position[4]; // 4 camera
};

//n_frame, n_row, n_col
Tensor<double, 3>
GenerateDeformableSphere(const vector<vector<FeaturePair>> &feature_pair_list,
                         const PolyCamera &ploy_camera,
                         const double depth_constrain_weight,
                         const double first_spatial_smooth_constraint_weight,
                         const double second_spatial_smooth_constraint_weight,
                         const double temporial_smooth_constraint_weight);

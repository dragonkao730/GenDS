#include "gends.h"

inline Vector3d
ThetaPhi2XYZ(const Vector2d &theta_phi)
{
	const double &theta = theta_phi(0);
	const double &phi = theta_phi(1);
	return Vector3d(sin(theta) * sin(phi),
					-cos(phi),
					cos(theta) * sin(phi));
}

inline Vector2d
XYZ2ThetaPhi(Vector3d xyz)
{
	xyz /= xyz.norm();
	double theta = atan(xyz(0) / xyz(2));
	double phi = acos(-xyz(1));
	if (xyz(2) < 0.0)
		theta += PI;
	else if (xyz(0) < 0.0)
		theta += 2 * PI;
	return Vector2d(theta, phi);
}

DepthPoint
GetDepthPoint(const FeaturePair &feature_pair,
			  const PolyCamera &ploy_camera,
			  const int frame_index)
{
	// input
	const Vector3d xyz_m = ThetaPhi2XYZ(feature_pair.first.theta_phi);  // norm
	const Vector3d xyz_n = ThetaPhi2XYZ(feature_pair.second.theta_phi); // norm
	const Vector3d &baseline = ploy_camera.baseline(feature_pair.first.camera_index,
													feature_pair.second.camera_index);
	// theta
	const double t = baseline.norm();
	const Vector3d e = baseline / t;
	const double theta_m = acos(e.dot(xyz_m));
	const double theta_n = acos(e.dot(xyz_n));
	// depth
	double d_m = numeric_limits<double>::infinity();
	double d_n = numeric_limits<double>::infinity();
	if (theta_m != theta_n)
	{
		d_m = t * sin(theta_n) / sin(theta_n - theta_m);
		d_n = t * sin(theta_m) / sin(theta_n - theta_m);
	}
	// p
	const Vector3d xyz_p = (xyz_m * d_m +
							ploy_camera.position[feature_pair.first.camera_index] +
							xyz_n * d_n +
							ploy_camera.position[feature_pair.second.camera_index]) *
						   0.5;
	const Vector2d theta_phi_p = XYZ2ThetaPhi(xyz_p);
	// return
	double depth = min(DEPTH_MAX, xyz_p.norm());
	if (d_m < 0 | d_n < 0)
		depth *= -1.0;
	return DepthPoint(theta_phi_p(0), theta_phi_p(1), depth, frame_index);
}

Tensor<double, 3>
GenerateDeformableSphere(const vector<vector<FeaturePair>> &feature_pair_list,
						 const PolyCamera &ploy_camera,
						 const int n_row,
						 const int n_col,
						 const double depth_constrain_weight,
						 const double first_spatial_smooth_constraint_weight,
						 const double second_spatial_smooth_constraint_weight,
						 const double temporial_smooth_constraint_weight)
{
	const int n_frame = feature_pair_list.size();
	GridInfo grid_info(n_frame, n_row, n_col);
	// new depth_point_list
	vector<DepthPoint> *depth_point_list = new vector<DepthPoint>();
	for (int frame_index = 0; frame_index < n_frame; frame_index++)
	{
		for (auto &feature_pair : feature_pair_list[frame_index])
		{
			DepthPoint depth_point = GetDepthPoint(feature_pair,
												   ploy_camera,
												   frame_index);
			if (depth_point.depth > 0)
				depth_point_list->push_back(depth_point);
		}
	}
	// new depth_constrain
{
	vector<Constrain> *depth_constrain;
	*depth_constrain = GetDepthConstraint(grid_info,
										  depth_point_list,
										  set<tuple<int, int, int>> &depth_constrain_flag);
}




	// depth < 0 不會push
	return Tensor<double, 3>(3, 3, 3);
}

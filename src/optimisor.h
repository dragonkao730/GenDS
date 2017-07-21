// This class implements optimization methods for alignment, including iterative nonlinear optimization and
// linear optimization
#ifndef OPTIMISOR_H
#define OPTIMISOR_H

#include <opencv2/opencv.hpp>
/*
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
*/


#include "alignment.h"
#include <Eigen/Eigen>

#define USE_FEATURE 1;


struct DistortionData{
	std::vector<int> idx_ver_r; //(4)
	std::vector<int> idx_ver_c; //(4)
	cv::Mat Cp; //(8, 8)

};

struct InterpolateInfo{
	InterpolateInfo(){
		vertex.resize(4);
		alpha.resize(4);
	}
	std::vector<int> vertex;
	std::vector<float> alpha;
	~InterpolateInfo(){
		std::vector<int>().swap(vertex);
		std::vector<float>().swap(alpha);
	}
	//	(0)-(1)
	//   |   |
	//  (2)-(3)
};

class Optimisor{
public:
	Optimisor(AlignData& align_data, std::vector<Camera>& cameras);
	Optimisor(AlignData& align_data);
	
	double linearSolve2();// used for another method
	
private:
	//----- Image warping utilities -----//
	void getWarpImage(cv::Mat& in_img, cv::Mat& out_img, cv::Mat& in_mask, cv::Mat& out_mask,  
					  ImageMesh& in_mesh, ImageMesh &out_mesh);
	double crossProduct(double u[2], double v[2]);
	//==== functions for linear system optimization ====//
	void getPairCorrespondence(std::vector<Correspondence>& corr);
	void getAnchorPoints(std::vector<Correspondence>& corr, std::vector<cv::Point2f>& anchor_points);
	void getDepthPoints(std::vector<Correspondence>& corr, std::vector<cv::Vec3d>& depth_points);
	void equi2Sphere(cv::Point2f& pe, cv::Vec3d& X, cv::Size equi_size);
	void sphere2Equi(cv::Vec3d& X, cv::Point2f& pe, cv::Size equi_size);
	void sphere2Rad(cv::Vec3d& X, cv::Point2f& pr);
	void Rad2sphere(cv::Point2f& pr ,cv::Vec3d& X);
	void buildCameraMatrix(Camera& cam, cv::Mat& cam_mat);
	void getWarpImage(cv::Mat& in_img, cv::Mat& out_img,  
					  ImageMesh& in_mesh, ImageMesh &out_mesh);
	void getGridVertices(std::vector<cv::Point2f> &vertex, ImageMesh&mesh, const int &idx_r, const int &idx_c);
	void corrConstraint(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int& row_count, std::vector<Correspondence>& corr);
	void anchorConstraint(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int& row_count, 
						  std::vector<Correspondence>& corr, std::vector<cv::Point2f>& anchor_points);
	void depthConstraint(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int& row_count, std::vector<cv::Vec3d>& depth_points, int num);
	void depthConstraint2(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int& row_count, std::vector<cv::Vec3d>& depth_points);
	void depthConstraint3(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int& row_count, std::vector<cv::Vec3d>& depth_points);
	void distConstraint(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int img, int& row_count);
	void smoothConstraint2(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int& row_count, int num);
	void smoothConstraint3(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int& row_count);
	void smoothConstraint(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int img, int& row_count);
	void temperalSmoothConstraint(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int& row_count, int max_vert);
	void lengthConstraint(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int img, int& row_count);
	void originConstraint(std::vector< std::vector<double> >& matrix_val, std::vector<double>& b, int img, int& row_count);
	// function for transform sphere into equi cordinated
	void transformSphere2equi(Eigen::VectorXd& equi, Eigen::VectorXd sphere);
	void StichBySphere(Eigen::VectorXd sphere);
	double FastIntersection(cv::Vec3d D, cv::Vec3d O, cv::Vec3d V0, cv::Vec3d V1, cv::Vec3d V2);
	// functions for segment-based linear optimization
	void initMesh(cv::Rect& bb, int num_vert_row, ImageMesh& in_mesh, ImageMesh& deform_mesh);
	void adaptiveMesh(cv::Rect& bb, ImageMesh& in_mesh, ImageMesh& deform_mesh);
	double equiCorre2Depth(cv::Point2f& fm, cv::Point2f& fn, cv::Vec3d& baseline, cv::Size& equi_size,
						   cv::Vec3d& psm, cv::Vec3d& psn, double& theta_m, double& theta_n);
	// visualization
	void drawComposition();

	// functions for computer bay-cordinated
	void Barycentric(cv::Point2f p, cv::Point2f a, cv::Point2f b, cv::Point2f c, float &u, float &v, float &w);
	
	void drawDispMaps(std::vector<cv::Mat>& disp_maps, std::string& filename);
	void drawMesh(ImageMesh& mesh, cv::Mat& img);
	AlignData &align_data;
	std::vector<Camera> cameras;

};

#endif
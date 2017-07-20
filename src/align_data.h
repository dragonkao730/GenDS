#ifndef ALIGNDATA_H
#define ALIGNDATA_H

#include <opencv2/opencv.hpp>

/*
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
*/
#include <utility>

using namespace std;
using namespace cv;

// store vertices in 2D array
typedef vector<vector<Point2f>> ImageMesh;

struct Correspondence{
	cv::Point2f fm; // point in image m
	cv::Point2f fn; // matched point in image n
	int m; 
	int n;
	// matched images: m <-> n
};


struct Feature{
	Feature(){
		idx_quad = -1;
		pos.x = -1;		pos.y = -1;
		vertex.resize(4);	alpha.resize(4);
	}
	cv::Point2f pos;
	int idx_quad;
	// ---- block information----
	//	(0)-(1)
	//   |   |
	//  (2)-(3)
	std::vector<int> vertex; // vertex number
	std::vector<double> alpha;

	//records which quad and vertices surround the feature
	//idx_quad: 1D idx of quad
	//vertex: 1D indices of the 4 surrounding vertices
	//alpha: corresponding coeff for interp.
};

struct FeatureGraph{
	// row: #images [idx_i]
	// col: #features [numFeat]
	std::vector< std::vector<Feature> > feat_info;
	std::vector<bool> isExist;
	// matFA: store alpha coefficients (size is mesh-dependent). row: feat idx; col: vertex idx 
	std::vector<cv::Mat> matFA;
	// matFb: store coordinate of matching feature (size is mesh independent). row: feat idx
	std::vector<cv::Mat> matFM;

	// featPic[idx_pic].featInfo[idx_i] is a vector of "Struct Feature" for image pair (idx_pic, idx_i)
};


struct MeshData{
	ImageMesh ori_mesh;
	std::vector<ImageMesh> deform_meshes;
};


// Each image is associated with a ImageData
struct ImageData{
	cv::Point2f pos; // image position on panorama ROI
	cv::Mat img; // original rgb image
	cv::Mat scale_img; // scaled image (for efficient warp computing)
	cv::Mat mask; // alpha mask (indicate pixels without final contribution to panorama)
	cv::Mat scale_mask;
	std::vector<cv::Mat> warp_imgs; // warped images to panorama relative to each other image
	std::vector<cv::Mat> warp_masks; // warped alpha mask to panorama relative to each other image
};
struct CorrData{
	std::vector<Correspondence> correspon;
};
struct AlignData{
	std::vector<ImageData> img_data;
	std::vector<std::pair<int, int> > match_pairs;
	std::vector<CorrData> corr_data;
	std::vector<MeshData> mesh_data;
	std::vector<FeatureGraph> feature_graph;
	double dist_weight;
	double feat_weight;
	double grad_weight;
	double shape_weight;
	double length_weight;
	double origin_weight;
	int scale;
	int frame_size;
};

#endif
//This class implement the two-stage alignment: pairwise match and overall aggregation
#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include <opencv2/opencv.hpp>
#include <string>
#include "align_data.h"

#define CACHE_DATA 1
#define NON_LOOP_GRID 0
#define LOOP_GRID 1

using namespace std;
using namespace cv;

struct Camera
{
	Vec3d pos;
	Vec3d ori;
	Vec3d up;
	double fovy;
	double ratio;
};

/*
struct PolyCamera
{
	vector<Camera> cameras;

	PolyCamera(const string& config_path)
   	{
		ifstream fs(config_path);
		assert(fs.good());
		int num_cam;
		fs >> num_cam;
		cameras.resize(num_cam);
		for(int c = 0; c < num_cam; ++c)
		{
			fs >> cameras[c].pos[0] >> cameras[c].pos[1] >> cameras[c].pos[2];
			fs >> cameras[c].ori[0] >> cameras[c].ori[1] >> cameras[c].ori[2];
			fs >> cameras[c].up[0] >> cameras[c].up[1] >> cameras[c].up[2];
			fs >> cameras[c].fovy >> cameras[c].ratio;
		}
		fs.close();
   	}
};
*/

struct Triangle{
	std::vector<cv::Vec3d> vertpos;
	std::vector<cv::Vec3d> vertnormal;
	std::vector<cv::Vec2d> vertuv;
	cv::Vec3i forecolor;
	cv::Vec3i backcolor;
};

struct Model{
	std::vector<cv::Vec3i> indices;
	std::vector<cv::Vec3d> vertices;
	std::vector<cv::Vec2d> uvs;
	std::vector<Triangle> triangles;
};

using namespace std;

class Alignment{
public:
	
	Alignment(	vector<string>& img_names,
				vector<string>& mask_names,
				vector<string>& feat_names,
				string cam_file);
	/*
	Alignment(	const vector<Camera>& input_cameras,
				const vector<Mat>& input_masks,
				const vector<Mat>& input_images);
	*/
	
	void aggregation();
private:
	void readImages(std::vector<std::string>& img_names, std::vector<std::string>& mask_names);
	void readFeature(std::string& feat_file);
	void readFeature(std::vector<std::string>& feat_names);
	void initMesh(int num_vert_row);
#if 1
	void exportAlignGridMesh(std::vector<cv::Mat>& remap_mat, float num_grid);
	void exportAlignDepthMaps(std::vector<cv::Mat>& remap_mat);
	void mapDeformMeshToSrcMesh(std::vector<cv::Mat>& remap_mat);
	void loadCameras(std::string filename);
	void equi2Sphere(cv::Point2f& pe, cv::Vec3d& X, cv::Size equi_size);
	void sphere2Equi(cv::Vec3d& X, cv::Point2f& pe, cv::Size equi_size);
	void buildCameraMatrix(Camera& cam, cv::Mat& cam_mat);
	void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color, int line_width);
	void triangulation(cv::Mat& image, std::vector<cv::Point2f>& keypoints, std::vector<cv::Vec3i>& triangles);
	void exportMeshes(std::vector<Model>& meshes, const char* filename);
	void assign3DTriangles(std::vector<Model>& meshes);
	bool corre2Depth(cv::Point2f& fm, cv::Point2f& fn, cv::Vec3d& cam_m, cv::Vec3d& cam_n, cv::Size equi_size, cv::Vec3d& ps);
	void constructPixelLists(int numSegs, cv::Mat &labelMap, 
							 std::vector<std::vector<cv::Point2i>> &segPixelLists);
	void constructBoundingBoxes(std::vector< std::vector<cv::Point2i> >& segments,
							    std::vector<cv::Rect>& segbbs);
#endif
	double crossProduct(double u[2], double v[2]);
	int num_imgs;
	AlignData align_data;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> equi_segmaps;
	std::vector< std::vector< std::vector<cv::Point2i> > > equi_segments;
	std::vector< std::vector< cv::Rect> > equi_segbbs;
	std::vector<cv::Vec2i> cam_pairs;
	std::vector<cv::Vec4f> keypoints;
};
#endif
//This class implement the two-stage alignment: pairwise match and overall aggregation
#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include <opencv2/opencv.hpp>

/*
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
*/

#include <string>
#include "align_data.h"

#define CACHE_DATA 1
#define NON_LOOP_GRID 0
#define LOOP_GRID 1

struct Camera
{
	cv::Vec3d pos;
	cv::Vec3d ori;
	cv::Vec3d up;
	double fovy;
	double ratio;
};

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

class Alignment{
public:
	//	這個沒用
	Alignment(std::vector<std::string>& img_names, std::vector<std::string>& mask_names, std::string feat_file,
			  std::string cam_file);
	
	Alignment(std::vector<std::string>& img_names, std::vector<std::string>& mask_names, std::vector<std::string>& feat_names,
			  std::string cam_file);
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
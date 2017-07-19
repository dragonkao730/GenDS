#include "alignment.h"
#include "optimisor.h"

/*
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
*/
#include <opencv2/opencv.hpp>

//#include <opencv2\contrib\contrib.hpp>
#include <iostream>
#include <fstream>
#include "dirent.h"

using namespace std;
using namespace cv;


/*
Alignment::Alignment(vector<string>& img_names, vector<string>& mask_names, 
					 string feat_file, string cam_file){
	align_data.scale = 1.0;
	loadCameras(cam_file);
	readImages(img_names, mask_names);
	readFeature(feat_file);
}
*/

Alignment::Alignment(vector<string>& img_names, vector<string>& mask_names, 
					 std::vector<std::string>& feat_names, string cam_file){
	align_data.scale = 1.0;
	align_data.frame_size = feat_names.size();
	loadCameras(cam_file);
	readImages(img_names, mask_names);
	readFeature(feat_names);
}

void Alignment::constructPixelLists(int numSegs, cv::Mat &labelMap, 
									vector<vector<Point2i>> &segPixelLists)
{
	segPixelLists.resize(numSegs);
	
	int numRows = labelMap.rows, numCols = labelMap.cols;
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = labelMap.at<int>(y, x);
			if(id >= 0)
			{
				segPixelLists[id].push_back(Point2i(x, y));
			}
		}
	}
}

void Alignment::constructBoundingBoxes(vector< vector<Point2i> >& segments,
									   vector<Rect>& segbbs)
{
	segbbs.resize(segments.size());
	for(size_t s = 0; s < segments.size(); ++s)
		{
			if(segments[s].size() == 0)
				continue;
			Point lt, rb;
			lt = rb = segments[s][0];
			for(size_t p = 1; p < segments[s].size(); ++p)
			{
				Point pt = segments[s][p];
				lt.x = (lt.x > pt.x) ? pt.x : lt.x;
				lt.y = (lt.y > pt.y) ? pt.y : lt.y;
				rb.x = (rb.x < pt.x) ? pt.x : rb.x;
				rb.y = (rb.y < pt.y) ? pt.y : rb.y;
			}
			segbbs[s].x = lt.x;
			segbbs[s].y = lt.y;
			segbbs[s].width = rb.x - lt.x;
			segbbs[s].height = rb.y - lt.y;
		}
}


void Alignment::readImages(vector<string>& img_names, vector<string>& mask_names)
{

	// Initialize align_data according to #images
	align_data.img_data.resize(img_names.size());
	align_data.feature_graph.resize(img_names.size());
	align_data.img_data.resize(img_names.size());
	align_data.mesh_data.resize(img_names.size());
	for(size_t i = 0; i < img_names.size(); ++i){
		align_data.mesh_data[i].deform_meshes.resize(img_names.size());
		align_data.img_data[i].warp_imgs.resize(img_names.size());
		align_data.img_data[i].warp_masks.resize(img_names.size());
	}
	// Initialize match pairs (for 4cam case)
	for(size_t i = 0; i < align_data.img_data.size(); ++i){
		if(i < align_data.img_data.size()-1)
			align_data.match_pairs.push_back(make_pair(i, i+1));
		else
			align_data.match_pairs.push_back(make_pair(i, 0));
	}

	// store rgb image, mask, image position into align_data.image_data (in opencv structure)
	// allocate warp image memory for each image
	for(size_t i = 0; i < img_names.size(); ++i){
		Mat &img = align_data.img_data[i].img;
		Mat &mask = align_data.img_data[i].mask;
		Mat &scale_img = align_data.img_data[i].scale_img;
		Mat &scale_mask = align_data.img_data[i].scale_mask;
		img = imread(img_names[i]);
		mask = imread(mask_names[i], CV_LOAD_IMAGE_GRAYSCALE);
		scale_img = Mat::zeros(img.rows/align_data.scale, img.cols/align_data.scale, CV_8UC3);
		resize(img, scale_img, scale_img.size());
		scale_mask = Mat::zeros(img.rows/align_data.scale, img.cols/align_data.scale, CV_8UC1);
		resize(mask, scale_mask, scale_mask.size());
	}
	// allocate memory for warp images in image data
	// For computation efficiency, we use downsampled images to compute warping parameter
	// Every warp image is allocated as scaled panorama size
	// Compute scaled panorama size
	int pano_width = align_data.img_data[0].img.cols;
	int pano_height = align_data.img_data[0].img.rows;
	int scale_pano_width = pano_width / align_data.scale;
	int scale_pano_height = pano_height /align_data.scale;
	for(size_t i = 0; i < img_names.size(); ++i){
		ImageData &img_data = align_data.img_data[i];
		Point2f &pos = img_data.pos;
		pos.x = pos.y = 0.0;
		for(size_t j = 0; j < align_data.img_data[i].warp_imgs.size(); ++j){
			img_data.warp_imgs[j] = Mat::zeros(scale_pano_height, scale_pano_width, CV_8UC3);
			img_data.warp_masks[j] = Mat::zeros(scale_pano_height, scale_pano_width, CV_8UC1);
		}
	}

	// allocate memory for feature graph
	align_data.feature_graph.resize(img_names.size());
	for(size_t i = 0; i < img_names.size(); ++i){
		align_data.feature_graph[i].feat_info.resize(img_names.size());
		align_data.feature_graph[i].matFA.resize(img_names.size());
		align_data.feature_graph[i].matFM.resize(img_names.size());
	}

	// draw initial scaled pano
	for(size_t i = 0; i < align_data.img_data.size(); ++i){
		Mat& warp = align_data.img_data[i].warp_imgs[i];	
		Mat& img = align_data.img_data[i].scale_img;
		Point2f pos = align_data.img_data[i].pos;
		for(int r = 0; r < img.rows; ++r)
			for(int c = 0; c < img.cols; ++c){
				int y2 = r + pos.y;
				int x2 = c + pos.x;
				warp.at<Vec3b>(r + (int)pos.y, c + (int)pos.x) = img.at<Vec3b>(r, c);
			}
	}
	Mat comp = 0.5 * align_data.img_data[0].warp_imgs[0];
	for(size_t i = 1; i < 4; ++i)
		comp += 0.5 * align_data.img_data[i].warp_imgs[i];
	imwrite("init_aggregation.png", comp);

}

bool Alignment::corre2Depth(Point2f& fm, Point2f& fn, Vec3d& cam_m, Vec3d& cam_n, Size equi_size, Vec3d& ps)
{
	Vec3d baseline = cam_n - cam_m;
	// currently size of input image and output image are the same
	Vec3d psm, psn;
	equi2Sphere(fm, psm, equi_size);
	equi2Sphere(fn, psn, equi_size);
	// back to pano coordinate. All equicam have same orientation with aligned sphere coordinate. Thus the vectors do not need change
	double t = cv::norm(baseline);
	Vec3d e = baseline / t;
	double theta_m = acos(e[0] * psm[0] + e[1] * psm[1] + e[2] * psm[2]);
	double theta_n = acos(e[0] * psn[0] + e[1] * psn[1] + e[2] * psn[2]);

	double Dm = t * sin(theta_n) / sin(theta_n - theta_m);

	if(Dm > 0.0)
	{
		// compute 3D point from camera n also, and average the position
		psm *= Dm;
		double Dn = -t * sin(theta_m) / sin(theta_m - theta_n);
		psn *= Dn;
		psm = psm + cam_m;;
		psn = psn + cam_n;
		ps = (psm + psn) * 0.5;
		return true;
	}	
	return false;
}

void Alignment::readFeature(vector<string>& feat_names){
	align_data.corr_data.resize(feat_names.size());
	for(int i=0;i<feat_names.size();i++)
	{
		fstream fs(feat_names[i].c_str(), fstream::in);
		char line_str[256];
		int img_i, img_j;
		Point2f pos_i, pos_j;
		while(fs.getline(line_str, 256)){
			string info(line_str);
			if(info.compare(0, 3, "c n") == 0){
				// read feature
				stringstream ss;
				ss << info;
				string s;
				while(ss >> s){
					if(s.compare(0, 1, "n") == 0){
						img_i = atoi(&s[1]);
					}
					else if(s.compare(0, 1, "N") == 0){
						img_j = atoi(&s[1]);
					}
					else if(s.compare(0, 1, "x") == 0){
						pos_i.x = atof(&s[1]);
					}
					else if(s.compare(0, 1, "y") == 0){
						pos_i.y = atof(&s[1]);
					}
					else if(s.compare(0, 1, "X") == 0){
						pos_j.x = atof(&s[1]);
					}
					else if(s.compare(0, 1, "Y") == 0){
						pos_j.y = atof(&s[1]);
					}
				}
				Feature feat_i, feat_j;
				pos_i.x /= align_data.scale;
				pos_i.y /= align_data.scale;
				pos_j.x /= align_data.scale;
				pos_j.y /= align_data.scale;
				pos_i -= align_data.img_data[img_i].pos;
				pos_j -= align_data.img_data[img_j].pos;
				feat_i.pos = pos_i;
				feat_j.pos = pos_j;
			
				/*FeatureGraph& feat_graph_i = align_data.feature_graph[img_i];
				FeatureGraph& feat_graph_j = align_data.feature_graph[img_j];

				feat_graph_i.feat_info[img_j].push_back(feat_i);
				feat_graph_j.feat_info[img_i].push_back(feat_j);*/

				Correspondence C;
				C.fm = pos_i;
				C.fn = pos_j;
				C.m = img_i;
				C.n = img_j;
				align_data.corr_data[i].correspon.push_back(C);
				/*align_data.feat_data[i].feature_i.push_back(feat_i);
				align_data.feat_data[i].feature_i.push_back(feat_j);*/
			}
		}
#if 0
	// project features to other cameras
	int num_cam = align_data.feature_graph.size();
	Size equi_size = align_data.img_data[0].scale_img.size();
	
	for(int i = 0; i < num_cam; ++i)
	{
		cout << cameras[i].pos<<endl;

		int j = (i == num_cam-1) ? 0 : i+1;
		int k = (i == 0) ? num_cam-1 : i-1;
		for(int f = 0; f < align_data.feature_graph[i].feat_info[j].size(); ++f)
		{
			Point2f fm = align_data.feature_graph[i].feat_info[j][f].pos;
			Point2f fn = align_data.feature_graph[j].feat_info[i][f].pos;
			Vec3d ps;

			if(corre2Depth(fm, fn, cameras[i].pos, cameras[j].pos, equi_size, ps))
			{
			ps = ps - cameras[k].pos;
			ps /= norm(ps);
			Point2f pe;
			sphere2Equi(ps, pe, equi_size);
			if(pe.x >= 0 && pe.x < equi_size.width && pe.y >= 0 && pe.y < equi_size.height)
				if(align_data.img_data[k].scale_mask.at<uchar>(pe) >= 128)
				{
					Feature feat_i, feat_k;
					feat_i.pos = fm;
					feat_k.pos = pe;
					align_data.feature_graph[i].feat_info[k].push_back(feat_i);
					align_data.feature_graph[k].feat_info[i].push_back(feat_k);
					
				}
			}
		}
		for(int f = 0; f < align_data.feature_graph[i].feat_info[k].size(); ++f)
		{
			Point2f fm = align_data.feature_graph[i].feat_info[k][f].pos;
			Point2f fn = align_data.feature_graph[k].feat_info[i][f].pos;
			Vec3d ps;
			if(corre2Depth(fm, fn, cameras[i].pos, cameras[k].pos, equi_size, ps))
			{
			ps = ps - cameras[j].pos;
			ps /= norm(ps);
			Point2f pe;
			sphere2Equi(ps, pe, equi_size);
			if(pe.x >= 0 && pe.x < equi_size.width && pe.y >= 0 && pe.y < equi_size.height)
				if(align_data.img_data[j].scale_mask.at<uchar>(pe) >= 128)
				{
					Feature feat_i, feat_j;
					feat_i.pos = fm;
					feat_j.pos = pe;
					if(i == 0 && k == 3 && align_data.feature_graph[i].feat_info[j].size() == 29297)
						cout << "pe:"<<pe<<", fm:"<<fm<<", fn:"<<fn<<endl;
					align_data.feature_graph[i].feat_info[j].push_back(feat_i);
					align_data.feature_graph[j].feat_info[i].push_back(feat_j);
				}
			}
		}

	}
#endif
	
#if 1
	/*// initialize matFM of each feature graph
	for(size_t i = 0; i < align_data.feature_graph.size(); ++i){
		FeatureGraph feat_graph_i = align_data.feature_graph[i];
		for(size_t j = 0; j < feat_graph_i.matFM.size(); ++j){
			size_t num_feat = feat_graph_i.feat_info[j].size();
			if(num_feat > 0){
				FeatureGraph feat_graph_j = align_data.feature_graph[j];
				feat_graph_i.matFM[j] = Mat::zeros(num_feat * 2, 1, CV_32FC1);
				for(size_t f = 0; f < num_feat; ++f){
					feat_graph_i.matFM[j].at<float>(f, 0) = feat_graph_j.feat_info[i][f].pos.x;
					feat_graph_i.matFM[j].at<float>(f+num_feat, 0) = feat_graph_j.feat_info[i][f].pos.y;
				}
			}
		}
	}*/
#endif
		fs.close();
	}
#if 0
	// draw feature points
	for(size_t i = 0; i < align_data.img_data.size()-1; ++i){
		int num_feat = align_data.feature_graph[i].feat_info[i+1].size();
		for(size_t j = 0; j < num_feat; ++j){
		circle(align_data.img_data[i].scale_img, align_data.feature_graph[i].feat_info[i+1][j].pos, 5, Scalar(0, 0, 255), -1);
		circle(align_data.img_data[i+1].scale_img, align_data.feature_graph[i+1].feat_info[i][j].pos, 5, Scalar(0, 255, 0), -1);
		}
	}
	char filename[256];
	for(size_t i = 0; i < align_data.img_data.size(); ++i){
		sprintf(filename, "ctrl-%d.png", i);
		imwrite(filename, align_data.img_data[i].scale_img);
	}
#endif
}
void Alignment::readFeature(string& feat_file){
		fstream fs(feat_file.c_str(), fstream::in);
		char line_str[256];
		int img_i, img_j;
		Point2f pos_i, pos_j;
		while(fs.getline(line_str, 256)){
			string info(line_str);
			if(info.compare(0, 3, "c n") == 0){
				// read feature
				stringstream ss;
				ss << info;
				string s;
				while(ss >> s){
					if(s.compare(0, 1, "n") == 0){
						img_i = atoi(&s[1]);
					}
					else if(s.compare(0, 1, "N") == 0){
						img_j = atoi(&s[1]);
					}
					else if(s.compare(0, 1, "x") == 0){
						pos_i.x = atof(&s[1]);
					}
					else if(s.compare(0, 1, "y") == 0){
						pos_i.y = atof(&s[1]);
					}
					else if(s.compare(0, 1, "X") == 0){
						pos_j.x = atof(&s[1]);
					}
					else if(s.compare(0, 1, "Y") == 0){
						pos_j.y = atof(&s[1]);
					}
				}
				Feature feat_i, feat_j;
				pos_i.x /= align_data.scale;
				pos_i.y /= align_data.scale;
				pos_j.x /= align_data.scale;
				pos_j.y /= align_data.scale;
				pos_i -= align_data.img_data[img_i].pos;
				pos_j -= align_data.img_data[img_j].pos;
				feat_i.pos = pos_i;
				feat_j.pos = pos_j;
			
				FeatureGraph& feat_graph_i = align_data.feature_graph[img_i];
				FeatureGraph& feat_graph_j = align_data.feature_graph[img_j];

				feat_graph_i.feat_info[img_j].push_back(feat_i);
				feat_graph_j.feat_info[img_i].push_back(feat_j);
			}
		}
#if 0
	// project features to other cameras
	int num_cam = align_data.feature_graph.size();
	Size equi_size = align_data.img_data[0].scale_img.size();
	
	for(int i = 0; i < num_cam; ++i)
	{
		cout << cameras[i].pos<<endl;

		int j = (i == num_cam-1) ? 0 : i+1;
		int k = (i == 0) ? num_cam-1 : i-1;
		for(int f = 0; f < align_data.feature_graph[i].feat_info[j].size(); ++f)
		{
			Point2f fm = align_data.feature_graph[i].feat_info[j][f].pos;
			Point2f fn = align_data.feature_graph[j].feat_info[i][f].pos;
			Vec3d ps;

			if(corre2Depth(fm, fn, cameras[i].pos, cameras[j].pos, equi_size, ps))
			{
			ps = ps - cameras[k].pos;
			ps /= norm(ps);
			Point2f pe;
			sphere2Equi(ps, pe, equi_size);
			if(pe.x >= 0 && pe.x < equi_size.width && pe.y >= 0 && pe.y < equi_size.height)
				if(align_data.img_data[k].scale_mask.at<uchar>(pe) >= 128)
				{
					Feature feat_i, feat_k;
					feat_i.pos = fm;
					feat_k.pos = pe;
					align_data.feature_graph[i].feat_info[k].push_back(feat_i);
					align_data.feature_graph[k].feat_info[i].push_back(feat_k);
					
				}
			}
		}
		for(int f = 0; f < align_data.feature_graph[i].feat_info[k].size(); ++f)
		{
			Point2f fm = align_data.feature_graph[i].feat_info[k][f].pos;
			Point2f fn = align_data.feature_graph[k].feat_info[i][f].pos;
			Vec3d ps;
			if(corre2Depth(fm, fn, cameras[i].pos, cameras[k].pos, equi_size, ps))
			{
			ps = ps - cameras[j].pos;
			ps /= norm(ps);
			Point2f pe;
			sphere2Equi(ps, pe, equi_size);
			if(pe.x >= 0 && pe.x < equi_size.width && pe.y >= 0 && pe.y < equi_size.height)
				if(align_data.img_data[j].scale_mask.at<uchar>(pe) >= 128)
				{
					Feature feat_i, feat_j;
					feat_i.pos = fm;
					feat_j.pos = pe;
					if(i == 0 && k == 3 && align_data.feature_graph[i].feat_info[j].size() == 29297)
						cout << "pe:"<<pe<<", fm:"<<fm<<", fn:"<<fn<<endl;
					align_data.feature_graph[i].feat_info[j].push_back(feat_i);
					align_data.feature_graph[j].feat_info[i].push_back(feat_j);
				}
			}
		}
	}
#endif
	
#if 1
	// initialize matFM of each feature graph
	for(size_t i = 0; i < align_data.feature_graph.size(); ++i){
		FeatureGraph feat_graph_i = align_data.feature_graph[i];
		for(size_t j = 0; j < feat_graph_i.matFM.size(); ++j){
			size_t num_feat = feat_graph_i.feat_info[j].size();
			if(num_feat > 0){
				FeatureGraph feat_graph_j = align_data.feature_graph[j];
				feat_graph_i.matFM[j] = Mat::zeros(num_feat * 2, 1, CV_32FC1);
				for(size_t f = 0; f < num_feat; ++f){
					feat_graph_i.matFM[j].at<float>(f, 0) = feat_graph_j.feat_info[i][f].pos.x;
					feat_graph_i.matFM[j].at<float>(f+num_feat, 0) = feat_graph_j.feat_info[i][f].pos.y;
				}
			}
		}
	}
#endif
	fs.close();
#if 0
	// draw feature points
	for(size_t i = 0; i < align_data.img_data.size()-1; ++i){
		int num_feat = align_data.feature_graph[i].feat_info[i+1].size();
		for(size_t j = 0; j < num_feat; ++j){
		circle(align_data.img_data[i].scale_img, align_data.feature_graph[i].feat_info[i+1][j].pos, 5, Scalar(0, 0, 255), -1);
		circle(align_data.img_data[i+1].scale_img, align_data.feature_graph[i+1].feat_info[i][j].pos, 5, Scalar(0, 255, 0), -1);
		}
	}
	char filename[256];
	for(size_t i = 0; i < align_data.img_data.size(); ++i){
		sprintf(filename, "ctrl-%d.png", i);
		imwrite(filename, align_data.img_data[i].scale_img);
	}
#endif
}




void Alignment::aggregation(){
	initMesh(21);
#if 0
	// parameter setting considering #constraints
	// initial shape weight
	align_data.shape_weight = 500;
	align_data.dist_weight = 100;
	align_data.feat_weight = 5000;
	align_data.length_weight = 50;
	align_data.origin_weight = 1e6;
#endif
#if 1
	// parameter setting for each constraint
	align_data.shape_weight = 1.0;
	align_data.dist_weight = 4.0;
	align_data.feat_weight = 1.0;
	align_data.length_weight = 1.0;
	align_data.origin_weight = 1e4;
#endif
	bool finish = false;
	double curr_energy, prev_energy;
	double emp = 1e-5;
	Optimisor aggrOpt(align_data, cameras);
	//change here
	prev_energy = aggrOpt.linearSolve3();

	//prev_energy = aggrOpt.linearSolve();
	/*
	vector<Mat> remap_mat;
	mapDeformMeshToSrcMesh(remap_mat);
	exportAlignGridMesh(remap_mat, 500);
	exportAlignDepthMaps(remap_mat);
	*/


}

void Alignment::initMesh(int num_vert_row){
	for(size_t i = 0; i < align_data.img_data.size(); ++i){
		for(size_t j = 0; j < align_data.mesh_data[i].deform_meshes.size(); ++j)
		{
			ImageMesh& in_mesh = align_data.mesh_data[i].ori_mesh;
			ImageMesh& deform_mesh = align_data.mesh_data[i].deform_meshes[j];
			

			in_mesh.resize(num_vert_row);
			Size grid_size;
			grid_size.width = align_data.img_data[i].scale_img.cols-1;
			grid_size.height = align_data.img_data[i].scale_img.rows-1;
			grid_size.width /= (float) (num_vert_row - 1);
			grid_size.height /= (float) (num_vert_row - 1);
			
			for(size_t r = 0; r < in_mesh.size(); ++r){
#if NON_LOOP_GRID
				in_mesh[r].resize(num_vert_row);
#endif
#if LOOP_GRID
				in_mesh[r].resize(num_vert_row-1);
#endif
				
				for(size_t c = 0; c < in_mesh[r].size(); ++c){
					in_mesh[r][c].x = c * grid_size.width;
					in_mesh[r][c].y = r * grid_size.height;
				}
			}	
			deform_mesh.resize(num_vert_row);
			for(size_t r = 0; r < deform_mesh.size(); ++r){
#if NON_LOOP_GRID
				deform_mesh[r].resize(num_vert_row);
#endif
#if LOOP_GRID
				deform_mesh[r].resize(num_vert_row-1);
#endif
			}
		}
	}
}

double Alignment::crossProduct(double u[2], double v[2]){ 
	// note: this function is NOT symmetric!
	return u[0]*v[1]-u[1]*v[0];
}

void Alignment::loadCameras(string filename)
{
	ifstream fs(filename);
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
}

void Alignment::mapDeformMeshToSrcMesh(vector<Mat>& remap_mat)
{
	remap_mat.resize(align_data.img_data.size());
	for(int c = 0; c < align_data.img_data.size(); ++c)
	{
		ImageMesh& in_mesh = align_data.mesh_data[c].ori_mesh;
		ImageMesh& out_mesh = align_data.mesh_data[c].deform_meshes[c];
		Mat& in_img = align_data.img_data[c].scale_img;
		Mat& out_img = align_data.img_data[c].warp_imgs[c];
		Mat& in_mask = align_data.img_data[c].scale_mask;
		remap_mat[c] = Mat::ones(out_img.size(), CV_64FC2);
		remap_mat[c] *= -1.0;
		double max_limit = 1e+9;
		double emp = 1e-9;

		int numRow = in_mesh.size();
#if NON_LOOP_GRID
	int numCol = in_mesh[0].size();
#endif
#if LOOP_GRID
	int numCol = in_mesh[0].size()+1;
#endif
		Size mesh_s = Size(numCol, numRow);
		Size in_s = Size(in_img.cols, in_img.rows);
		Size out_s = Size(out_img.cols, out_img.rows);
		
		for(int my = 0; my<mesh_s.height-1; my++){
			for(int mx = 0; mx<mesh_s.width-1; mx++){ // for every quad
#if NON_LOOP_GRID
			Point2f in_quad[2][2] = { {in_mesh[my][mx], in_mesh[my][mx+1]}, {in_mesh[my+1][mx], in_mesh[my+1][mx+1]}}; // current quad
			Point2f out_quad[2][2] = { {out_mesh[my][mx], out_mesh[my][mx+1]}, {out_mesh[my+1][mx], out_mesh[my+1][mx+1]}}; // current quad
#endif
#if LOOP_GRID
			Point2f in_quad[2][2], out_quad[2][2];
			if(mx < mesh_s.width-2)
			{
				in_quad[0][0] = in_mesh[my][mx];
				in_quad[0][1] = in_mesh[my][mx+1];
				in_quad[1][0] = in_mesh[my+1][mx];
				in_quad[1][1] = in_mesh[my+1][mx+1];

				out_quad[0][0] = out_mesh[my][mx];
				out_quad[0][1] = out_mesh[my][mx+1];
				out_quad[1][0] = out_mesh[my+1][mx];
				out_quad[1][1] = out_mesh[my+1][mx+1];
			}
			else
			{
				in_quad[0][0] = in_mesh[my][mx];
				in_quad[0][1] = in_mesh[my][0];
				in_quad[1][0] = in_mesh[my+1][mx];
				in_quad[1][1] = in_mesh[my+1][0];
				in_quad[0][1].x += in_img.cols - 1;
				in_quad[1][1].x += in_img.cols - 1;

				out_quad[0][0] = out_mesh[my][mx];
				out_quad[0][1] = out_mesh[my][0];
				out_quad[1][0] = out_mesh[my+1][mx];
				out_quad[1][1] = out_mesh[my+1][0];
				out_quad[0][1].x += out_img.cols - 1;
				out_quad[1][1].x += out_img.cols - 1;
			}
#endif
				//bounding box
				Point2f minmin = Point2f( max_limit,  max_limit);
				Point2f maxmax = Point2f(-max_limit, -max_limit);
				for(int i = 0; i<2; i++){
					for(int j = 0; j<2; j++){
						if(out_quad[i][j].x<minmin.x) minmin.x = out_quad[i][j].x;
						if(out_quad[i][j].y<minmin.y) minmin.y = out_quad[i][j].y;
						if(out_quad[i][j].x>maxmax.x) maxmax.x = out_quad[i][j].x;
						if(out_quad[i][j].y>maxmax.y) maxmax.y = out_quad[i][j].y;
					}
				}
#if NON_LOOP_GRID
			if(	minmin.x>=out_s.width || minmin.y>=out_s.height || maxmax.x<0 || maxmax.y<0) continue; // bounding box is out of image
			if(minmin.x<0) minmin.x = 0; //calculate the intersection between bounding box and image border to obtain the valid bounding box
			if(maxmax.x>=out_s.width) maxmax.x = (float)out_s.width - 1;
			
#endif
			if(minmin.y<0) minmin.y = 0;
			if(maxmax.y>=out_s.height) maxmax.y = (float)out_s.height - 1;
				minmin.x = ceil(minmin.x); // convert to integer value
				minmin.y = ceil(minmin.y);
				maxmax.x = floor(maxmax.x);
				maxmax.y = floor(maxmax.y);
				//for every pixel in the bounding box, find its bilinear coefficients of quad and then inverse warp to find the inverse position
				double p0_p2[2] = { out_quad[0][0].x-out_quad[1][0].x, out_quad[0][0].y-out_quad[1][0].y };//for calculate s and t
				double p1_p3[2] = { out_quad[0][1].x-out_quad[1][1].x, out_quad[0][1].y-out_quad[1][1].y };//for calculate s and t
				double denom = 0;
				int count = 0;
				for(int y = (int)minmin.y; y<=(int)maxmax.y; y++){
					for(int x = (int)minmin.x; x<=(int)maxmax.x; x++){
						//calculate s
						double s1 = 0, s2 = 0; // 2 possible solution of s
						double p0_p[2] = { out_quad[0][0].x-x, out_quad[0][0].y-y };
						double p1_p[2] = { out_quad[0][1].x-x, out_quad[0][1].y-y };
						double A = crossProduct(p0_p, p0_p2);
						double B = (double)(crossProduct(p0_p, p1_p3)+crossProduct(p1_p, p0_p2))/2;
						double C = crossProduct(p1_p, p1_p3);
						denom = A - 2*B + C;
						if(fabs(denom) < emp){
							denom = A - C;
							if(fabs(denom) < emp){
								if(fabs(A) < emp){ //all values for s contain p
									s1 = 0;
								}
								else
									continue;
							}else{
								s1 = A / denom;
							}
							s2 = s1;
						}else{
							double tmp = B*B-A*C;
							if(tmp<-(emp)){
								continue; // out of quad
							}else if(tmp<0){ // negative near zero (treat it as zero)
								tmp = 0;
							}
							s1 = (double)( (A-B) + sqrt(tmp) ) / denom;
							s2 = (double)( (A-B) - sqrt(tmp) ) / denom;
						}

						//calculate t
						double t1 = 0, t2 = 0; // 2 possible solution of t
						//t1
						denom = (1-s1)*p0_p2[0] + s1*p1_p3[0];
						if(fabs(denom) < emp){
							denom = (1-s1)*p0_p2[1] + s1*p1_p3[1];
							if(fabs(denom) < emp) denom += emp; // this happens when p0_p2 is parallel to p1_p3. In this case, s1 in not in [0, 1], s2 is the solution
							t1 = (double)((1-s1)*p0_p[1] + s1*p1_p[1])/denom;
						}else{
							t1 = (double)((1-s1)*p0_p[0] + s1*p1_p[0])/denom;
						}
						//t2
						denom = (1-s2)*p0_p2[0] + s2*p1_p3[0];
						if(fabs(denom) < emp){
							denom = (1-s2)*p0_p2[1] + s2*p1_p3[1];
							if(fabs(denom) < emp) denom += emp; // this happens when p0_p2 is parallel to p1_p3. In this case, s2 in not in [0, 1], s1 is the solution
							t2 = (double)((1-s2)*p0_p[1] + s2*p1_p[1])/denom;
						}else{
							t2 = (double)((1-s2)*p0_p[0] + s2*p1_p[0])/denom;
						}

						double s = 0, t = 0;

						if(s1>=0 && s1<=1 && t1>=0 && t1<=1){
							s = s1;
							t = t1;
						}else if(s2>=0 && s2<=1 && t2>=0 && t2<=1){
							s = s2;
							t = t2;
						}else{
							continue; // out of quad
						}
						//calculate bilinear interpolation of 4 neighbor pixels (anti-aliasing)
						double inv_pos[2] = { (1-s) * ((1-t)*in_quad[0][0].x+t*in_quad[1][0].x) + s * ((1-t)*in_quad[0][1].x+t*in_quad[1][1].x), 
							(1-s) * ((1-t)*in_quad[0][0].y+t*in_quad[1][0].y) + s * ((1-t)*in_quad[0][1].y+t*in_quad[1][1].y) };
						int fip[2] = { (int)floor(inv_pos[0]), (int)floor(inv_pos[1]) }; // floored inverse position
						if(fip[0]<0 || fip[1]<0 || fip[0] > in_s.width-1 || fip[1] > in_s.height-1) continue; // pixel out of image
						if(in_mask.at<uchar>(inv_pos[1], inv_pos[0]) >= 128)
						{
							if( x >= 0 && x < out_img.cols)
							{
								remap_mat[c].at<Vec2d>(y, x) = Vec2d(inv_pos[0], inv_pos[1]); 
							}
#if LOOP_GRID
							else
							{
								if(x < 0)
								{
									remap_mat[c].at<Vec2d>(y, x + out_img.cols) = Vec2d(inv_pos[0], inv_pos[1]); 

								}
								else
								{
									remap_mat[c].at<Vec2d>(y, x - out_img.cols) = Vec2d(inv_pos[0], inv_pos[1]); 
								}
							}
#endif
						}
					}
				}
			}
		}
	}
}

void Alignment::equi2Sphere(Point2f& pe, Vec3d& X, Size equi_size)
{
	double theta = pe.y * CV_PI / (double) equi_size.height;
	double phi = pe.x * CV_PI * 2 / (double) equi_size.width;
	X[0] = sin(phi) * sin(theta);
	X[1] = -cos(theta);
	X[2] = cos(phi) * sin(theta);
}

void Alignment::buildCameraMatrix(Camera& cam, Mat& cam_mat)
{
	Mat up = Mat::zeros(3, 1, CV_64FC1);
	Mat forward = Mat::zeros(3, 1, CV_64FC1);
	forward.at<double>(0) = cam.ori[0];
	forward.at<double>(1) = cam.ori[1];
	forward.at<double>(2) = cam.ori[2];
	up.at<double>(0) = cam.up[0];
	up.at<double>(1) = cam.up[1];
	up.at<double>(2) = cam.up[2];
	Mat side = forward.cross(up);
	up = side.cross(forward);
	cam_mat.at<double>(0, 0) = side.at<double>(0);
	cam_mat.at<double>(0, 1) = side.at<double>(1);
	cam_mat.at<double>(0, 2) = side.at<double>(2);
	cam_mat.at<double>(1, 0) = up.at<double>(0);
	cam_mat.at<double>(1, 1) = up.at<double>(1);
	cam_mat.at<double>(1, 2) = up.at<double>(2);
	cam_mat.at<double>(2, 0) = -forward.at<double>(0);
	cam_mat.at<double>(2, 1) = -forward.at<double>(1);
	cam_mat.at<double>(2, 2) = -forward.at<double>(2);
	// Camera Matrix: R(X+T), not RX+T!
	Mat Rc = Mat::zeros(3, 3, CV_64FC1);
	for(int r = 0; r < 3; ++r)
		for(int c = 0; c < 3; ++c)
			Rc.at<double>(r, c) = cam_mat.at<double>(r, c);
	Mat cam_pos = Mat::zeros(3, 1, CV_64FC1);
	cam_pos.at<double>(0) = -cam.pos[0];
	cam_pos.at<double>(1) = -cam.pos[1];
	cam_pos.at<double>(2) = -cam.pos[2];
	Mat rot_cam_pos = Rc * cam_pos;
	cam_mat.at<double>(0, 3) = rot_cam_pos.at<double>(0);
	cam_mat.at<double>(1, 3) = rot_cam_pos.at<double>(1);
	cam_mat.at<double>(2, 3) = rot_cam_pos.at<double>(2);
	cam_mat.at<double>(3, 3) = 1.0;
}

void Alignment::sphere2Equi(Vec3d& X, Point2f& pe, Size equi_size)
{
	// opencv coordinates
	double xs = X[0];
	double ys = X[1];
	double zs = X[2];

#if 0
	// current coordinates
	double theta = acos(zs);
	double phi = atan(ys/xs);
	if(xs < 0.0)
		phi += CV_PI;
	else if(ys < 0.0)
		phi += 2 * CV_PI;
#endif
#if 1
	double theta = acos(-ys);
	double phi = atan(xs / zs);
	if(zs < 0.0)
		phi += CV_PI;
	else if(xs < 0.0)
		phi += 2 * CV_PI;
#endif
	pe.y = theta * equi_size.height / CV_PI;
	pe.x = phi * equi_size.width * 0.5 / CV_PI;
}

void Alignment::draw_delaunay( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color, int line_width )
{

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);
	Size size = img.size();
	Rect rect(0,0, size.width, size.height);

	for( size_t i = 0; i < triangleList.size(); i++ )
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

		// Draw rectangles completely inside the image.
		if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			line(img, pt[0], pt[1], delaunay_color, line_width, CV_AA, 0);
			line(img, pt[1], pt[2], delaunay_color, line_width, CV_AA, 0);
			line(img, pt[2], pt[0], delaunay_color, line_width, CV_AA, 0);
		}
	}
}


void Alignment::triangulation(Mat& image, vector<Point2f>& keypoints,
							  vector<Vec3i>& triangles)
{
	// draw keypoints
	// Delaunay Triangulation
	// Rectangle to be used with Subdiv2D
	Size size = image.size();
	Rect rect(0, 0, size.width, size.height);

	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);

	// Create a vector of points.
	vector<Point2f> points = keypoints;

	// Insert points into subdiv
	for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
	{
		subdiv.insert(*it);
	}
	

	// get vertex indices for each triangles
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point2f> pt(3);
	
	/*map<int, int> corr;
	for(int i = 0; i < keypoints.size(); ++i)
	{
		int ed, vt;
		subdiv.locate(keypoints[i], ed, vt);
		corr[vt] = i;
	}

	for( size_t i = 0; i < triangleList.size(); i++ )
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5]);
		if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
		Vec3i ed, vt;
		subdiv.locate(pt[0], ed[0], vt[0]);
		subdiv.locate(pt[1], ed[1], vt[1]);
		subdiv.locate(pt[2], ed[2], vt[2]);
		vt[0] = corr[vt[0]];
		vt[1] = corr[vt[1]];
		vt[2] = corr[vt[2]];
		triangles.push_back(vt);
		}
	}

	// Draw delaunay triangles
	draw_delaunay( image, subdiv, Scalar(0, 255, 0),1);*/

	
	// Allocate space for Voronoi Diagram
	//Mat img_voronoi = Mat::zeros(image.rows, image.cols, CV_8UC3);

	// Draw Voronoi diagram
	//draw_voronoi( img_voronoi, subdiv );
	//vorimg << "voronoi-" << i << ".jpg";
	//imwrite(filename, image);
	//imwrite(vorimg.str().c_str(), img_voronoi);
}

void Alignment::assign3DTriangles(vector<Model>& meshes)
{
	for(int i = 0; i < meshes.size(); ++i)
	{
		for(int j = 0; j < meshes[i].indices.size(); ++j)
		{
			Triangle t;
			t.vertnormal.resize(3);
			t.vertpos.resize(3);
			t.vertuv.resize(3);
			Vec3i ind = meshes[i].indices[j];
			t.backcolor = t.forecolor = Vec3i(0, 0, 0);
			for(int v = 0; v < 3; ++v)
			{
				t.vertnormal[v] = Vec3d(0.0, 0.0, 0.0);
				t.vertpos[v] = meshes[i].vertices[ind[v]];
			}
			meshes[i].triangles.push_back(t);
		}
	}
}

void Alignment::exportMeshes(vector<Model>& meshes, const char* filename)
{
	for(int m = 0; m < meshes.size(); ++m)
	{
		// export .tri
		stringstream sstr;
		sstr << filename << "-" << m << ".tri";
		ofstream fs(sstr.str());
		for(int t = 0; t < meshes[m].triangles.size(); ++t)
		{
			fs << "Triangle" <<endl;
			Triangle tr = meshes[m].triangles[t];
			fs << tr.forecolor[0]<<" "<<tr.forecolor[1]<<" "<<tr.forecolor[2]<<" ";
			fs << tr.backcolor[0]<<" "<<tr.backcolor[1]<<" "<<tr.backcolor[2]<<endl;
			for(int v = 0; v < 3; ++v)
			{
				fs << tr.vertpos[v][0] << " "<<tr.vertpos[v][1]<<" "<<tr.vertpos[v][2]<<" ";
				fs << tr.vertnormal[v][0]<<" "<<tr.vertnormal[v][1]<<" "<<tr.vertnormal[v][2]<<endl;
			}
		}
		fs.close();
	}
}

void Alignment::exportAlignDepthMaps(vector<Mat>& remap_mat)
{
	vector<Mat> camera_mat(cameras.size());
	for(int i = 0; i < cameras.size(); ++i)
	{
		camera_mat[i] = Mat::zeros(4, 4, CV_64FC1);
		buildCameraMatrix(cameras[i], camera_mat[i]);
	}
	vector<Mat> Rc(cameras.size());
	vector<Mat> Tc(cameras.size());
	for(int i = 0; i < cameras.size(); ++i)
	{
		Rc[i] = Mat::zeros(3, 3, CV_64FC1);
		Tc[i] = Mat::zeros(3, 1, CV_64FC1);
		for(int r = 0; r < 3; ++r)
			for(int c = 0; c < 3; ++c)
				Rc[i].at<double>(r, c) = camera_mat[i].at<double>(r, c);
		for(int r = 0; r < 3; ++r)
			Tc[i].at<double>(r) = camera_mat[i].at<double>(r, 3);
		// compute origin offset of each camera
		Tc[i] = Rc[i].t() * Tc[i];
		Tc[i] = Tc[i].reshape(3, 1);
	}
	
	Mat& out_img = align_data.img_data[0].warp_imgs[0];
	
	// consider baseline vectors of all camera pairs 
	vector< vector<Vec3d> > baseline(cameras.size());
	for(int i = 0; i < cameras.size(); ++i)
	{
		baseline[i].resize(cameras.size());
		for(int j = 0; j < cameras.size(); ++j)
			baseline[i][j] = (i == j) ? Vec3d(0.0, 0.0, 0.0) : cameras[j].pos - cameras[i].pos;
	}
	
	vector<Mat> equi_dmaps(align_data.img_data.size());
	Mat align_equi_dmap = Mat::zeros(out_img.size(), CV_64FC1);
	for(int i = 0; i < equi_dmaps.size(); ++i)
		equi_dmaps[i]  = Mat::zeros(out_img.size(), CV_64FC1);
	
	double dmax = 0.0, dmin = 1e6;

	//
	for(int r = 0; r < out_img.rows; r ++)
	{
		for(int c = 0; c < out_img.cols; c++)
		{
			Point2f grid(c, r);
			
			// check remapping mat for each image
			vector<Vec2d> src_pts;
			vector<int> hit_imgs;
			vector<Point3f> corre_3dpts;	
			
			for(int i = 0; i < align_data.img_data.size(); ++i)
			{
				Vec2d src_pt = remap_mat[i].at<Vec2d>(grid);
				if(src_pt[0] >= 0.0)
				{
					hit_imgs.push_back(i);
					src_pts.push_back(src_pt);
				}
			}
			vector<Correspondence> corres;

			Correspondence corr;
			// consider all pairs
			
			int num_corr = src_pts.size();
			vector<int> count(num_corr);
			for(int i = 0; i < num_corr; ++i)
			{
				int count = 0;
				Vec3d _3dpt = Vec3d(0.0, 0.0, 0.0);
				corr.fm = Point2f(src_pts[i][0], src_pts[i][1]);
				corr.m = hit_imgs[i];
				int m = corr.m;
				Point2f fm = corr.fm;
				for(int j = 0; j < num_corr; ++j)
				{
					if(i == j)
						continue;
					corr.fn = Point2f(src_pts[j][0], src_pts[j][1]);		
					corr.n = hit_imgs[j];
					corres.push_back(corr);

					// compute 3D points for each corres
					
					int n = corr.n;
					
					Point2f fn = corr.fn;
					// currently size of input image and output image are the same
					Vec3d psm, psn;
					equi2Sphere(fm, psm, out_img.size());
					equi2Sphere(fn, psn, out_img.size());
					// back to pano coordinate. All equicam have same orientation with aligned sphere coordinate. Thus the vectors do not need change
					double t = cv::norm(baseline[m][n]);
					Vec3d e = baseline[m][n] / t;
					double theta_m = acos(e[0] * psm[0] + e[1] * psm[1] + e[2] * psm[2]);
					double theta_n = acos(e[0] * psn[0] + e[1] * psn[1] + e[2] * psn[2]);

					double Dm = t * sin(theta_n) / sin(theta_n - theta_m);
					if(Dm > 0.0)
					{
						// compute 3D point from camera n also, and average the position
						psm *= Dm;
						double Dn = -t * sin(theta_m) / sin(theta_m - theta_n);
						psn *= Dn;

						psm = psm - Tc[m].at<Vec3d>(0);
						psn = psn - Tc[n].at<Vec3d>(0);
						Vec3d ps = (psm + psn) * 0.5;
						double d = norm(ps);
						// convert depth to disparity for visualization
						d = 1.0/d;
						dmax = (dmax < d) ? d : dmax;
						dmin = (dmin > d) ? d : dmin;
						
						equi_dmaps[m].at<double>(Point(fm)) += d;
						_3dpt += ps;	
						count++;
					}

				}
				// select one 3D points with minimum depth, give to aligned mesh
				corre_3dpts.push_back(Point3f(_3dpt/(double)count));
				equi_dmaps[m].at<double>(Point(fm)) /= (double)count;
			}
			
			if(corre_3dpts.size() > 0)
			{
				double mind = norm(corre_3dpts[0]);
				mind = 1.0/mind;
				
				for(int i = 1; i < corre_3dpts.size(); ++i)
				{
					double d = cv::norm(corre_3dpts[i]);
					d = 1.0/d;
					if(mind < d)
					{
						mind = d;
					}
				}
				align_equi_dmap.at<double>(r, c) = mind;
			}
			
		}
	}
	cout << "apply color map"<<endl;
	for(int i = 0; i < equi_dmaps.size(); ++i)
	{
		for(int r = 0; r < out_img.rows; r ++)
			for(int c = 0; c < out_img.cols; c++)
				equi_dmaps[i].at<double>(r, c) = 255.0 * (equi_dmaps[i].at<double>(r, c) - dmin)/(dmax - dmin);
			
		equi_dmaps[i].convertTo(equi_dmaps[i], CV_8UC1);
		applyColorMap(equi_dmaps[i], equi_dmaps[i], COLORMAP_JET);
		for(int r = 0; r < out_img.rows; r ++)
			for(int c = 0; c < out_img.cols; c++)
				if(remap_mat[i].at<Vec2d>(r, c)[0] < 0.0)
					equi_dmaps[i].at<Vec3b>(r, c) = Vec3b(0, 0, 0);
		stringstream sstr;
		sstr << "equi-dmaps-"<<i<<".jpg";
		imwrite(sstr.str(), equi_dmaps[i]);
	}
	cout << "apply align color map"<<endl;
	for(int r = 0; r < out_img.rows; r ++)
		for(int c = 0; c < out_img.cols; c++)
			align_equi_dmap.at<double>(r, c) = 255.0 * (align_equi_dmap.at<double>(r, c) - dmin)/(dmax - dmin);
	align_equi_dmap.convertTo(align_equi_dmap, CV_8UC1);
	applyColorMap(align_equi_dmap, align_equi_dmap, COLORMAP_JET);
	stringstream sstr;	
	imwrite("align-equi-dmap.jpg", align_equi_dmap);
}
void Alignment::exportAlignGridMesh(vector<Mat>& remap_mat, float num_grid)
{
	vector<Mat> camera_mat(cameras.size());
	for(int i = 0; i < cameras.size(); ++i)
	{
		camera_mat[i] = Mat::zeros(4, 4, CV_64FC1);
		buildCameraMatrix(cameras[i], camera_mat[i]);
	}
	vector<Mat> Rc(cameras.size());
	vector<Mat> Tc(cameras.size());
	for(int i = 0; i < cameras.size(); ++i)
	{
		Rc[i] = Mat::zeros(3, 3, CV_64FC1);
		Tc[i] = Mat::zeros(3, 1, CV_64FC1);
		for(int r = 0; r < 3; ++r)
			for(int c = 0; c < 3; ++c)
				Rc[i].at<double>(r, c) = camera_mat[i].at<double>(r, c);
		for(int r = 0; r < 3; ++r)
			Tc[i].at<double>(r) = camera_mat[i].at<double>(r, 3);
		// compute origin offset of each camera
		Tc[i] = Rc[i].t() * Tc[i];
		Tc[i] = Tc[i].reshape(3, 1);
	}
	
	Mat& out_img = align_data.img_data[0].warp_imgs[0];
	float grid_height = out_img.rows / num_grid;
	float grid_width = out_img.cols / num_grid;
	vector<Correspondence> all_corres;
	vector<Point3f> all_corre_3dpts;
	vector<Point2f> all_corre_2dpts;
	vector<Point3f> align_3dpts;
	vector<Point2f> align_2dpts;
	// consider baseline vectors of all camera pairs 
	vector< vector<Vec3d> > baseline(cameras.size());
	for(int i = 0; i < cameras.size(); ++i)
	{
		baseline[i].resize(cameras.size());
		for(int j = 0; j < cameras.size(); ++j)
			baseline[i][j] = (i == j) ? Vec3d(0.0, 0.0, 0.0) : cameras[j].pos - cameras[i].pos;
	}
	

	for(int r = 0; r <= num_grid; r ++)
	{
		for(int c = 0; c <= num_grid; c++)
		{
			Point2f grid(c * grid_width, r * grid_height);
			grid.x = (grid.x >= out_img.cols) ? out_img.cols-1 : grid.x;
			grid.y = (grid.y >= out_img.rows) ? out_img.rows-1 : grid.y;
			// check remapping mat for each image
			vector<Vec2d> src_pts;
			vector<int> hit_imgs;
			for(int i = 0; i < align_data.img_data.size(); ++i)
			{
				Vec2d src_pt = remap_mat[i].at<Vec2d>(grid);
				if(src_pt[0] >= 0.0)
				{
					hit_imgs.push_back(i);
					src_pts.push_back(src_pt);
				}
			}
			vector<Correspondence> corres;
			vector<Point3f> corre_3dpts;
			vector<Point2f> corre_2dpts;
			Point3f align_3dpt;
			Point2f align_2dpt;
			Correspondence corr;
			// consider all pairs
			
			int num_corr = src_pts.size();
			
			for(int i = 0; i < num_corr; ++i)
			{
				//cout <<"i:"<<i<<endl;
				corr.fm = Point2f(src_pts[i][0], src_pts[i][1]);
				corr.m = hit_imgs[i];
				int m = corr.m;
				Point2f fm = corr.fm;
				int count = 0;
				Vec3d _3dpt = Vec3d(0.0, 0.0, 0.0);
				for(int j = 0; j < num_corr; ++j)
				{
					if(i == j)
						continue;
					
					corr.fn = Point2f(src_pts[j][0], src_pts[j][1]);
					
					corr.n = hit_imgs[j];
					corres.push_back(corr);

					// compute 3D points for each corres
					
					int n = corr.n;
					
					Point2f fn = corr.fn;
					// currently size of input image and output image are the same
					Vec3d psm, psn;
					equi2Sphere(fm, psm, out_img.size());
					equi2Sphere(fn, psn, out_img.size());
					// back to pano coordinate. All equicam have same orientation with aligned sphere coordinate. Thus the vectors do not need change
					double t = cv::norm(baseline[m][n]);
					Vec3d e = baseline[m][n] / t;
					double theta_m = acos(e[0] * psm[0] + e[1] * psm[1] + e[2] * psm[2]);
					double theta_n = acos(e[0] * psn[0] + e[1] * psn[1] + e[2] * psn[2]);

					double Dm = t * sin(theta_n) / sin(theta_n - theta_m);
					if(Dm > 0.0)
					{
						// compute 3D point from camera n also, and average the position
						all_corres.push_back(corr);
						psm *= Dm;
						double Dn = -t * sin(theta_m) / sin(theta_m - theta_n);
						psn *= Dn;

						psm = psm - Tc[m].at<Vec3d>(0);
						psn = psn - Tc[n].at<Vec3d>(0);
						Vec3d ps = (psm + psn) * 0.5;

						
						// compute projection on aligned panorama coordinate
						ps /= cv::norm(ps);
						Point2f pe;
						sphere2Equi(ps, pe, out_img.size());
						corre_2dpts.push_back(pe); 
						all_corre_2dpts.push_back(pe);
						_3dpt += ps;	
						count++;
					}
				}
				// ensure #corr = #3dpts
				for(int j = 0; j < corre_2dpts.size(); ++j)
				{
					corre_3dpts.push_back(Point3f(_3dpt/(double)count));
					all_corre_3dpts.push_back(Point3f(_3dpt/(double)count));
				}
			}
			
			if(corre_3dpts.size() > 0)
			{
				double mind = norm(corre_3dpts[0]);
				align_3dpt = corre_3dpts[0];
				for(int i = 1; i < corre_3dpts.size(); ++i)
				{
					double d = cv::norm(corre_3dpts[i]);
					if(mind > d)
					{
						align_3dpt = corre_3dpts[i];
						mind = d;
					}
				}
				align_2dpt = grid;
				align_3dpts.push_back(align_3dpt);
				align_2dpts.push_back(align_2dpt);
			}
		}
	}
	cout << "corre_3dpts.size():"<<all_corre_3dpts.size()<<", "<<align_2dpts.size()<<endl;
	cout << "build 3D mesh"<<endl;
	// build 3D mesh for all spheres
	vector<Mat> equi_feat_img(cameras.size());
	vector< vector<Point2f> > equi_feat(cameras.size());
	Mat align_equi_feat_img = Mat::zeros(out_img.size(), CV_8UC3);
	vector<Model> equi_meshes(cameras.size());
	Model align_equi_mesh;
	for(int c = 0; c < all_corres.size(); ++c)
	{
		int m = all_corres[c].m;
		int n = all_corres[c].n;
		Point2f fm = all_corres[c].fm;
		Point2f fn = all_corres[c].fn;
		equi_feat[m].push_back(fm);
		equi_feat[n].push_back(fn);
		equi_meshes[m].vertices.push_back(Vec3d(all_corre_3dpts[c].x, all_corre_3dpts[c].y, all_corre_3dpts[c].z));
		equi_meshes[n].vertices.push_back(Vec3d(all_corre_3dpts[c].x, all_corre_3dpts[c].y, all_corre_3dpts[c].z));
	}
	//
	int offset = out_img.cols / cameras.size() / 2;
	int step = out_img.cols / cameras.size();
	for(int i = 0; i < cameras.size(); ++i)
	{
		int j = (i == cameras.size() - 1) ? 0 : i+1;
		equi_feat_img[j] = align_data.img_data[j].scale_img.clone();
		Mat warp_equi_img = align_data.img_data[j].warp_imgs[j].clone();
		int rx = offset+i*step;

		if(j > 0)
			warp_equi_img.colRange(rx, rx+step).copyTo(align_equi_feat_img.colRange(rx, rx+step));
		else
		{
			warp_equi_img.colRange(rx, out_img.cols).copyTo(align_equi_feat_img.colRange(rx, out_img.cols));
			warp_equi_img.colRange(0, offset).copyTo(align_equi_feat_img.colRange(0, offset));
		}
	}
	cout << "triangulation"<<endl;
	for(int i = 0; i < cameras.size(); ++i)
	{
		//triangulation(equi_feat_img[i], equi_feat[i], equi_meshes[i].indices);
	}

	cout << "generate equi feat img"<<endl;
	for(int i = 0; i < align_3dpts.size(); ++i)
		align_equi_mesh.vertices.push_back(Vec3d(align_3dpts[i].x, align_3dpts[i].y, align_3dpts[i].z));

	cout << "align_3dpts:"<<align_3dpts.size()<<", "<<align_2dpts.size()<<endl;
	//triangulation(align_equi_feat_img, align_2dpts, align_equi_mesh.indices);
	equi_meshes.push_back(align_equi_mesh);
	equi_feat_img.push_back(align_equi_feat_img);
	assign3DTriangles(equi_meshes);
	exportMeshes(equi_meshes, "auxiliary/equi-meshes");

#if 0
	//reproject 3D mesh into 2D mesh
	for(int i = 0; i < equi_meshes.size(); ++i)
	{
		for(int t = 0; t < equi_meshes[i].triangles.size(); ++t)
		{
			Triangle tr = equi_meshes[i].triangles[t];
			vector<Vec3d> vert_pos = tr.vertpos;
			vector<Point2f> pe(3);
			for(int v = 0; v < vert_pos.size(); ++v)
			{
				Vec3d ps = vert_pos[v];
				if(i < 4)
					ps += Tc[i].at<Vec3d>(0);
				ps /= cv::norm(ps);
				sphere2Equi(ps, pe[v], out_img.size());
			}
			line(equi_feat_img[i], pe[0], pe[1], Scalar(0, 0, 255), 1);
			line(equi_feat_img[i], pe[0], pe[2], Scalar(0, 0, 255), 1);
			line(equi_feat_img[i], pe[1], pe[2], Scalar(0, 0, 255), 1);
		}
	}
#endif

	//draw images
	for(int i = 0; i < equi_meshes.size(); ++i)
	{
		stringstream sstr;
		sstr << "equi-mesh-"<<i<<".jpg";
		imwrite(sstr.str(), equi_feat_img[i]);
	}
}



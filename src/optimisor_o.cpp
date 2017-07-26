//#include "taucs_interface.h"
#include "optimisor.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <time.h>

#include "object_function.h"
#include "linear_solve.h"

using namespace std;
using namespace cv;

int frame = 0;

Optimisor::Optimisor(AlignData &align, vector<Camera> &cam) : align_data(align), cameras(cam)
{
}

Optimisor::Optimisor(AlignData &align) : align_data(align)
{
}

void Optimisor::getWarpImage(Mat &in_img, Mat &out_img, Mat &in_mask, Mat &out_mask,
							 ImageMesh &in_mesh, ImageMesh &out_mesh)
{
#define RECTANGLE 0
	// Warp both rgb image and alpha mask
	double max_limit = 1e+9;
	double emp = 1e-9;

	int numRow = in_mesh.size();
#if NON_LOOP_GRID
	int numCol = in_mesh[0].size();
#endif
#if LOOP_GRID
	int numCol = in_mesh[0].size() + 1;
#endif
	Size mesh_s = Size(numCol, numRow);
	Size in_s = Size(in_img.cols, in_img.rows);
	Size out_s = Size(out_img.cols, out_img.rows);
	out_img.setTo(Scalar(0, 0, 0));
	out_mask.setTo(0);
	for (int my = 0; my < mesh_s.height - 1; my++)
	{
		for (int mx = 0; mx < mesh_s.width - 1; mx++)
		{ // for every quad
#if NON_LOOP_GRID
			Point2f in_quad[2][2] = {{in_mesh[my][mx], in_mesh[my][mx + 1]}, {in_mesh[my + 1][mx], in_mesh[my + 1][mx + 1]}};	  // current quad
			Point2f out_quad[2][2] = {{out_mesh[my][mx], out_mesh[my][mx + 1]}, {out_mesh[my + 1][mx], out_mesh[my + 1][mx + 1]}}; // current quad
#endif
#if LOOP_GRID
			Point2f in_quad[2][2], out_quad[2][2];
			if (mx < mesh_s.width - 2)
			{
				in_quad[0][0] = in_mesh[my][mx];
				in_quad[0][1] = in_mesh[my][mx + 1];
				in_quad[1][0] = in_mesh[my + 1][mx];
				in_quad[1][1] = in_mesh[my + 1][mx + 1];

				out_quad[0][0] = out_mesh[my][mx];
				out_quad[0][1] = out_mesh[my][mx + 1];
				out_quad[1][0] = out_mesh[my + 1][mx];
				out_quad[1][1] = out_mesh[my + 1][mx + 1];
			}
			else
			{
				in_quad[0][0] = in_mesh[my][mx];
				in_quad[0][1] = in_mesh[my][0];
				in_quad[1][0] = in_mesh[my + 1][mx];
				in_quad[1][1] = in_mesh[my + 1][0];
				in_quad[0][1].x += in_img.cols - 1;
				in_quad[1][1].x += in_img.cols - 1;

				out_quad[0][0] = out_mesh[my][mx];
				out_quad[0][1] = out_mesh[my][0];
				out_quad[1][0] = out_mesh[my + 1][mx];
				out_quad[1][1] = out_mesh[my + 1][0];
				out_quad[0][1].x += out_img.cols - 1;
				out_quad[1][1].x += out_img.cols - 1;
			}
#endif
			//bounding box
			Point2f minmin = Point2f(max_limit, max_limit);
			Point2f maxmax = Point2f(-max_limit, -max_limit);
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					if (out_quad[i][j].x < minmin.x)
						minmin.x = out_quad[i][j].x;
					if (out_quad[i][j].y < minmin.y)
						minmin.y = out_quad[i][j].y;
					if (out_quad[i][j].x > maxmax.x)
						maxmax.x = out_quad[i][j].x;
					if (out_quad[i][j].y > maxmax.y)
						maxmax.y = out_quad[i][j].y;
				}
			}
#if RECTANGLE
			if (minmin.x >= out_s.width || minmin.y >= out_s.height || maxmax.x < 0 || maxmax.y < 0)
				continue; // bounding box is out of image
			if (minmin.x < 0)
				minmin.x = 0; //calculate the intersection between bounding box and image border to obtain the valid bounding box
			if (maxmax.x >= out_s.width)
				maxmax.x = (float)out_s.width - 1;

#endif
			if (minmin.y < 0)
				minmin.y = 0;
			if (maxmax.y >= out_s.height)
				maxmax.y = (float)out_s.height - 1;
			minmin.x = ceil(minmin.x); // convert to integer value
			minmin.y = ceil(minmin.y);
			maxmax.x = floor(maxmax.x);
			maxmax.y = floor(maxmax.y);
			//for every pixel in the bounding box, find its bilinear coefficients of quad and then inverse warp to find the inverse position
			double p0_p2[2] = {out_quad[0][0].x - out_quad[1][0].x, out_quad[0][0].y - out_quad[1][0].y}; //for calculate s and t
			double p1_p3[2] = {out_quad[0][1].x - out_quad[1][1].x, out_quad[0][1].y - out_quad[1][1].y}; //for calculate s and t
			double denom = 0;
			int count = 0;
			for (int y = (int)minmin.y; y <= (int)maxmax.y; y++)
			{
				for (int x = (int)minmin.x; x <= (int)maxmax.x; x++)
				{
					//calculate s
					double s1 = 0, s2 = 0; // 2 possible solution of s
					double p0_p[2] = {out_quad[0][0].x - x, out_quad[0][0].y - y};
					double p1_p[2] = {out_quad[0][1].x - x, out_quad[0][1].y - y};
					double A = crossProduct(p0_p, p0_p2);
					double B = (double)(crossProduct(p0_p, p1_p3) + crossProduct(p1_p, p0_p2)) / 2;
					double C = crossProduct(p1_p, p1_p3);
					denom = A - 2 * B + C;
					if (fabs(denom) < emp)
					{
						denom = A - C;
						if (fabs(denom) < emp)
						{
							if (fabs(A) < emp)
							{ //all values for s contain p
								s1 = 0;
							}
							else
								continue;
						}
						else
						{
							s1 = A / denom;
						}
						s2 = s1;
					}
					else
					{
						double tmp = B * B - A * C;
						if (tmp < -(emp))
						{
							continue; // out of quad
						}
						else if (tmp < 0)
						{ // negative near zero (treat it as zero)
							tmp = 0;
						}
						s1 = (double)((A - B) + sqrt(tmp)) / denom;
						s2 = (double)((A - B) - sqrt(tmp)) / denom;
					}

					//calculate t
					double t1 = 0, t2 = 0; // 2 possible solution of t
					//t1
					denom = (1 - s1) * p0_p2[0] + s1 * p1_p3[0];
					if (fabs(denom) < emp)
					{
						denom = (1 - s1) * p0_p2[1] + s1 * p1_p3[1];
						if (fabs(denom) < emp)
							denom += emp; // this happens when p0_p2 is parallel to p1_p3. In this case, s1 in not in [0, 1], s2 is the solution
						t1 = (double)((1 - s1) * p0_p[1] + s1 * p1_p[1]) / denom;
					}
					else
					{
						t1 = (double)((1 - s1) * p0_p[0] + s1 * p1_p[0]) / denom;
					}
					//t2
					denom = (1 - s2) * p0_p2[0] + s2 * p1_p3[0];
					if (fabs(denom) < emp)
					{
						denom = (1 - s2) * p0_p2[1] + s2 * p1_p3[1];
						if (fabs(denom) < emp)
							denom += emp; // this happens when p0_p2 is parallel to p1_p3. In this case, s2 in not in [0, 1], s1 is the solution
						t2 = (double)((1 - s2) * p0_p[1] + s2 * p1_p[1]) / denom;
					}
					else
					{
						t2 = (double)((1 - s2) * p0_p[0] + s2 * p1_p[0]) / denom;
					}

					double s = 0, t = 0;

					if (s1 >= 0 && s1 <= 1 && t1 >= 0 && t1 <= 1)
					{
						s = s1;
						t = t1;
					}
					else if (s2 >= 0 && s2 <= 1 && t2 >= 0 && t2 <= 1)
					{
						s = s2;
						t = t2;
					}
					else
					{
						continue; // out of quad
					}
					//calculate bilinear interpolation of 4 neighbor pixels (anti-aliasing)
					double inv_pos[2] = {(1 - s) * ((1 - t) * in_quad[0][0].x + t * in_quad[1][0].x) + s * ((1 - t) * in_quad[0][1].x + t * in_quad[1][1].x),
										 (1 - s) * ((1 - t) * in_quad[0][0].y + t * in_quad[1][0].y) + s * ((1 - t) * in_quad[0][1].y + t * in_quad[1][1].y)};
					int fip[2] = {(int)floor(inv_pos[0]), (int)floor(inv_pos[1])}; // floored inverse position
					if (fip[0] < 0 || fip[1] < 0 || fip[0] > in_s.width - 1 || fip[1] > in_s.height - 1)
						continue; // pixel out of image
					double s_2 = inv_pos[0] - fip[0];
					double t_2 = inv_pos[1] - fip[1];
					Vec3b nb[2][2] = {{in_img.at<Vec3b>(fip[1], fip[0]), in_img.at<Vec3b>(fip[1], fip[0] + 1)},
									  {in_img.at<Vec3b>(fip[1] + 1, fip[0]), in_img.at<Vec3b>(fip[1] + 1, fip[0] + 1)}}; // 4 neighbor pixels
					uchar na[2][2] = {{in_mask.at<uchar>(fip[1], fip[0]), in_mask.at<uchar>(fip[1], fip[0] + 1)},
									  {in_mask.at<uchar>(fip[1] + 1, fip[0]), in_mask.at<uchar>(fip[1] + 1, fip[0] + 1)}};					 // 4 neighbor pixels
					Vec3b final_color = (1 - s_2) * ((1 - t_2) * nb[0][0] + t_2 * nb[1][0]) + s_2 * ((1 - t_2) * nb[0][1] + t_2 * nb[1][1]); // interpolation
					uchar final_alpha = (1 - s_2) * ((1 - t_2) * na[0][0] + t_2 * na[1][0]) + s_2 * ((1 - t_2) * na[0][1] + t_2 * na[1][1]);

					if (x >= 0 && x < out_img.cols)
					{
						out_img.at<Vec3b>(y, x) = final_color;
						out_mask.at<uchar>(y, x) = final_alpha;
					}
					else
					{
						if (x < 0)
						{
							out_img.at<Vec3b>(y, x + out_img.cols) = final_color;
							out_mask.at<uchar>(y, x + out_img.cols) = final_alpha;
						}
						else
						{
							out_img.at<Vec3b>(y, x - out_img.cols) = final_color;
							out_mask.at<uchar>(y, x - out_img.cols) = final_alpha;
						}
					}
				}
			}
		}
	}
}

void Optimisor::getWarpImage(Mat &in_img, Mat &out_img,
							 ImageMesh &in_mesh, ImageMesh &out_mesh)
{
	// Warp both rgb image and alpha mask
	double max_limit = 1e+9;
	double emp = 1e-9;

	int numRow = in_mesh.size();
	int numCol = in_mesh[0].size();
	Size mesh_s = Size(numCol, numRow);
	Size in_s = Size(in_img.cols, in_img.rows);
	Size out_s = Size(out_img.cols, out_img.rows);
	out_img.setTo(Scalar(0, 0, 0));
	for (int my = 0; my < mesh_s.height - 1; my++)
	{
		for (int mx = 0; mx < mesh_s.width - 1; mx++)
		{																														   // for every quad
			Point2f in_quad[2][2] = {{in_mesh[my][mx], in_mesh[my][mx + 1]}, {in_mesh[my + 1][mx], in_mesh[my + 1][mx + 1]}};	  // current quad
			Point2f out_quad[2][2] = {{out_mesh[my][mx], out_mesh[my][mx + 1]}, {out_mesh[my + 1][mx], out_mesh[my + 1][mx + 1]}}; // current quad
			//bounding box
			Point2f minmin = Point2f(max_limit, max_limit);
			Point2f maxmax = Point2f(-max_limit, -max_limit);
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					if (out_quad[i][j].x < minmin.x)
						minmin.x = out_quad[i][j].x;
					if (out_quad[i][j].y < minmin.y)
						minmin.y = out_quad[i][j].y;
					if (out_quad[i][j].x > maxmax.x)
						maxmax.x = out_quad[i][j].x;
					if (out_quad[i][j].y > maxmax.y)
						maxmax.y = out_quad[i][j].y;
				}
			}
			if (minmin.x >= out_s.width || minmin.y >= out_s.height || maxmax.x < 0 || maxmax.y < 0)
				continue; // bounding box is out of image
			if (minmin.x < 0)
				minmin.x = 0; //calculate the intersection between bounding box and image border to obtain the valid bounding box
			if (minmin.y < 0)
				minmin.y = 0;
			if (maxmax.x >= out_s.width)
				maxmax.x = (float)out_s.width - 1;
			if (maxmax.y >= out_s.height)
				maxmax.y = (float)out_s.height - 1;
			minmin.x = ceil(minmin.x); // convert to integer value
			minmin.y = ceil(minmin.y);
			maxmax.x = floor(maxmax.x);
			maxmax.y = floor(maxmax.y);
			//for every pixel in the bounding box, find its bilinear coefficients of quad and then inverse warp to find the inverse position
			double p0_p2[2] = {out_quad[0][0].x - out_quad[1][0].x, out_quad[0][0].y - out_quad[1][0].y}; //for calculate s and t
			double p1_p3[2] = {out_quad[0][1].x - out_quad[1][1].x, out_quad[0][1].y - out_quad[1][1].y}; //for calculate s and t
			double denom = 0;
			int count = 0;
			for (int y = (int)minmin.y; y <= (int)maxmax.y; y++)
			{
				for (int x = (int)minmin.x; x <= (int)maxmax.x; x++)
				{
					//calculate s
					double s1 = 0, s2 = 0; // 2 possible solution of s
					double p0_p[2] = {out_quad[0][0].x - x, out_quad[0][0].y - y};
					double p1_p[2] = {out_quad[0][1].x - x, out_quad[0][1].y - y};
					double A = crossProduct(p0_p, p0_p2);
					double B = (double)(crossProduct(p0_p, p1_p3) + crossProduct(p1_p, p0_p2)) / 2;
					double C = crossProduct(p1_p, p1_p3);
					denom = A - 2 * B + C;
					if (fabs(denom) < emp)
					{
						denom = A - C;
						if (fabs(denom) < emp)
						{
							if (fabs(A) < emp)
							{ //all values for s contain p
								s1 = 0;
							}
							else
								continue;
						}
						else
						{
							s1 = A / denom;
						}
						s2 = s1;
					}
					else
					{
						double tmp = B * B - A * C;
						if (tmp < -(emp))
						{
							continue; // out of quad
						}
						else if (tmp < 0)
						{ // negative near zero (treat it as zero)
							tmp = 0;
						}
						s1 = (double)((A - B) + sqrt(tmp)) / denom;
						s2 = (double)((A - B) - sqrt(tmp)) / denom;
					}

					//calculate t
					double t1 = 0, t2 = 0; // 2 possible solution of t
					//t1
					denom = (1 - s1) * p0_p2[0] + s1 * p1_p3[0];
					if (fabs(denom) < emp)
					{
						denom = (1 - s1) * p0_p2[1] + s1 * p1_p3[1];
						if (fabs(denom) < emp)
							denom += emp; // this happens when p0_p2 is parallel to p1_p3. In this case, s1 in not in [0, 1], s2 is the solution
						t1 = (double)((1 - s1) * p0_p[1] + s1 * p1_p[1]) / denom;
					}
					else
					{
						t1 = (double)((1 - s1) * p0_p[0] + s1 * p1_p[0]) / denom;
					}
					//t2
					denom = (1 - s2) * p0_p2[0] + s2 * p1_p3[0];
					if (fabs(denom) < emp)
					{
						denom = (1 - s2) * p0_p2[1] + s2 * p1_p3[1];
						if (fabs(denom) < emp)
							denom += emp; // this happens when p0_p2 is parallel to p1_p3. In this case, s2 in not in [0, 1], s1 is the solution
						t2 = (double)((1 - s2) * p0_p[1] + s2 * p1_p[1]) / denom;
					}
					else
					{
						t2 = (double)((1 - s2) * p0_p[0] + s2 * p1_p[0]) / denom;
					}

					double s = 0, t = 0;

					if (s1 >= 0 && s1 <= 1 && t1 >= 0 && t1 <= 1)
					{
						s = s1;
						t = t1;
					}
					else if (s2 >= 0 && s2 <= 1 && t2 >= 0 && t2 <= 1)
					{
						s = s2;
						t = t2;
					}
					else
					{
						continue; // out of quad
					}
					//calculate bilinear interpolation of 4 neighbor pixels (anti-aliasing)
					double inv_pos[2] = {(1 - s) * ((1 - t) * in_quad[0][0].x + t * in_quad[1][0].x) + s * ((1 - t) * in_quad[0][1].x + t * in_quad[1][1].x),
										 (1 - s) * ((1 - t) * in_quad[0][0].y + t * in_quad[1][0].y) + s * ((1 - t) * in_quad[0][1].y + t * in_quad[1][1].y)};
					int fip[2] = {(int)floor(inv_pos[0]), (int)floor(inv_pos[1])}; // floored inverse position
					if (fip[0] < 0 || fip[1] < 0 || fip[0] >= in_s.width - 2 || fip[1] >= in_s.height - 2)
						continue; // pixel out of image
					double s_2 = inv_pos[0] - fip[0];
					double t_2 = inv_pos[1] - fip[1];
					Vec3b nb[2][2] = {{in_img.at<Vec3b>(fip[1], fip[0]), in_img.at<Vec3b>(fip[1], fip[0] + 1)},
									  {in_img.at<Vec3b>(fip[1] + 1, fip[0]), in_img.at<Vec3b>(fip[1] + 1, fip[0] + 1)}};					 // 4 neighbor pixels
					Vec3b final_color = (1 - s_2) * ((1 - t_2) * nb[0][0] + t_2 * nb[1][0]) + s_2 * ((1 - t_2) * nb[0][1] + t_2 * nb[1][1]); // interpolation
					out_img.at<Vec3b>(y, x) = final_color;
				}
			}
		}
	}
}

double Optimisor::crossProduct(double u[2], double v[2])
{
	// note: this function is NOT symmetric!
	return u[0] * v[1] - u[1] * v[0];
}

void Optimisor::equi2Sphere(Point2f &pe, Vec3d &X, Size equi_size)
{
	double theta = pe.y * CV_PI / (double)equi_size.height;
	double phi = pe.x * CV_PI * 2 / (double)equi_size.width;
	X[0] = sin(phi) * sin(theta);
	X[1] = -cos(theta);
	X[2] = cos(phi) * sin(theta);
}

void Optimisor::sphere2Equi(Vec3d &X, Point2f &pe, Size equi_size)
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
	//cout<<endl<<theta<<"  "<<phi<<endl;
	if (zs < 0.0)
		phi += CV_PI;
	else if (xs < 0.0)
		phi += 2 * CV_PI;
#endif
	pe.y = theta * equi_size.height / CV_PI;
	pe.x = phi * equi_size.width * 0.5 / CV_PI;
}

void Optimisor::sphere2Rad(Vec3d &X, Point2f &pr)
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
	//cout<<endl<<theta<<"  "<<phi<<endl;
	if (zs < 0.0)
		phi += CV_PI;
	else if (xs < 0.0)
		phi += 2 * CV_PI;
#endif
	pr.y = theta;
	pr.x = phi;
}

void Optimisor::buildCameraMatrix(Camera &cam, Mat &cam_mat)
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
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
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

double Optimisor::equiCorre2Depth(Point2f &fm, Point2f &fn, Vec3d &baseline, Size &equi_size,
								  Vec3d &psm, Vec3d &psn, double &theta_m, double &theta_n)
{
	//if(fm)
	equi2Sphere(fm, psm, equi_size);
	equi2Sphere(fn, psn, equi_size);
	// back to pano coordinate. All equicam have same orientation with aligned sphere coordinate. Thus the vectors do not need change
	double t = cv::norm(baseline);
	Vec3d e = baseline / t;
	theta_m = acos(e[0] * psm[0] + e[1] * psm[1] + e[2] * psm[2]);
	theta_n = acos(e[0] * psn[0] + e[1] * psn[1] + e[2] * psn[2]);
	double Dm = t * sin(theta_n) / sin(theta_n - theta_m);
	return Dm;
}

void Optimisor::getAnchorPoints(vector<Correspondence> &corres, vector<Point2f> &anchor_points)
{
	Size equi_size = align_data.img_data[0].warp_imgs[0].size();
	vector<Mat> camera_mat(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		camera_mat[i] = Mat::zeros(4, 4, CV_64FC1);
		buildCameraMatrix(cameras[i], camera_mat[i]);
	}
	vector<Mat> Rc(cameras.size());
	vector<Mat> Tc(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		Rc[i] = Mat::zeros(3, 3, CV_64FC1);
		Tc[i] = Mat::zeros(3, 1, CV_64FC1);
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				Rc[i].at<double>(r, c) = camera_mat[i].at<double>(r, c);
		for (int r = 0; r < 3; ++r)
			Tc[i].at<double>(r) = camera_mat[i].at<double>(r, 3);
		// compute origin offset of each camera
		Tc[i] = Rc[i].t() * Tc[i];
		Tc[i] = Tc[i].reshape(3, 1);
	}
	vector<vector<Vec3d>> baseline(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		baseline[i].resize(cameras.size());
		for (int j = 0; j < cameras.size(); ++j)
			baseline[i][j] = (i == j) ? Vec3d(0.0, 0.0, 0.0) : cameras[j].pos - cameras[i].pos;
	}

	for (int i = 0; i < corres.size(); ++i)
	{
		Correspondence corr = corres[i];
		// compute 3D points for each corres
		int m = corr.m;
		int n = corr.n;
		Point2f fm = corr.fm;
		Point2f fn = corr.fn;
		if (fm == fn)
		{
			anchor_points.push_back(fn);
			continue;
		}
		// currently size of input image and output image are the same
		Vec3d psm, psn;
		double theta_m, theta_n;
		double Dm = equiCorre2Depth(fm, fn, baseline[m][n], equi_size, psm, psn, theta_m, theta_n);
		if (Dm > 0.0)
		{
			double t = norm(baseline[m][n]);
			// compute 3D point from camera n also, and average the position
			psm *= Dm;
			double Dn = -t * sin(theta_m) / sin(theta_m - theta_n);
			psn *= Dn;
			psm = psm - Tc[m].at<Vec3d>(0);
			psn = psn - Tc[n].at<Vec3d>(0);
			Vec3d ps = (psm + psn) * 0.5;
			// compute projection on aligned panorama coordinate
			ps /= cv::norm(ps);
			Point2f pe;
			sphere2Equi(ps, pe, equi_size);
			anchor_points.push_back(pe);
		}
		/*else
		{
			cout<<"Dm:"<<Dm<<endl;
			cout<<"fm:"<<fm.x<<","<<fm.y<<endl;
			cout<<"fn:"<<fn.x<<","<<fn.y<<endl;
			cout<<"pm:"<<psm[0]<<","<<psm[1]<<","<<psm[2]<<endl;
			cout<<"pn:"<<psn[0]<<","<<psm[1]<<","<<psm[2]<<endl;
		}*/
	}
}

void Optimisor::getDepthPoints(vector<Correspondence> &corres, vector<Vec3d> &depth_points)
{
	Size equi_size = align_data.img_data[0].warp_imgs[0].size();
	vector<Mat> camera_mat(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		camera_mat[i] = Mat::zeros(4, 4, CV_64FC1);
		buildCameraMatrix(cameras[i], camera_mat[i]);
	}
	vector<Mat> Rc(cameras.size());
	vector<Mat> Tc(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		Rc[i] = Mat::zeros(3, 3, CV_64FC1);
		Tc[i] = Mat::zeros(3, 1, CV_64FC1);
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				Rc[i].at<double>(r, c) = camera_mat[i].at<double>(r, c);
		for (int r = 0; r < 3; ++r)
			Tc[i].at<double>(r) = camera_mat[i].at<double>(r, 3);
		// compute origin offset of each camera
		Tc[i] = Rc[i].t() * Tc[i];
		Tc[i] = Tc[i].reshape(3, 1);
	}
	vector<vector<Vec3d>> baseline(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		baseline[i].resize(cameras.size());
		for (int j = 0; j < cameras.size(); ++j)
			baseline[i][j] = (i == j) ? Vec3d(0.0, 0.0, 0.0) : cameras[j].pos - cameras[i].pos;
	}
	int count = 0, count14 = 0, count16 = 0, count18 = 0;
	for (int i = 0; i < corres.size(); ++i)
	{
		Correspondence corr = corres[i];

		// compute 3D points for each corres
		int m = corr.m;
		int n = corr.n;
		Point2f fm = corr.fm;
		Point2f fn = corr.fn;
		/*if((abs(fm.x-fn.x)>200)&&(abs(3000+(fm.x-fn.x)>200)))
			continue;*/
		if (fm == fn) // mean depth is infinite set to 10000
		{
			Vec3d ps;
			equi2Sphere(fm, ps, equi_size);

			ps /= cv::norm(ps);

			double theta = acos(-ps[1]);
			double phi = atan(ps[0] / ps[2]);
			if (ps[2] < 0.0)
				phi += CV_PI;
			else if (ps[0] < 0.0)
				phi += 2 * CV_PI;

			ps[0] = 1000000; //r
			ps[1] = phi;	 //theta
			ps[2] = theta;   //phi

			depth_points.push_back(ps);
			continue;
		}
		// currently size of input image and output image are the same

		Vec3d psm, psn;
		double theta_m, theta_n;
		double Dm = equiCorre2Depth(fm, fn, baseline[m][n], equi_size, psm, psn, theta_m, theta_n);
		if (Dm > 0.0)
		{
			double t = norm(baseline[m][n]);
			// compute 3D point from camera n also, and average the position
			psm *= Dm;
			double Dn = -t * sin(theta_m) / sin(theta_m - theta_n);
			psn *= Dn;

			psm = psm - Tc[m].at<Vec3d>(0);
			psn = psn - Tc[n].at<Vec3d>(0);

			Vec3d ps = (psm + psn) * 0.5;

			double depth = cv::norm(ps);
			ps /= depth;

			Point2f pe;
			//here to select mode
			//sphere2Equi(ps, pe, equi_size);
			sphere2Rad(ps, pe);
			//pe.x:phi, pe.y:theata
			/*double theta = acos(-ps[1]);
			double phi = atan(ps[0] / ps[2]);
			if(ps[2] < 0.0)
				phi += CV_PI;
			else if(ps[0] < 0.0)
				phi += 2 * CV_PI;*/

			if (depth > 1e+6)
				depth = 1e+6;

			ps[0] = depth; //r
			ps[1] = pe.x;  //phi
			ps[2] = pe.y;  //theata

			depth_points.push_back(ps);
			if (depth > 1e8)
				count18++;
			else if (depth > 1e6)
				count16++;
			else if (depth > 1e4)
				count14++;
			count++;
			/*if(depth<100)
				cout<<i<<endl;*/
		}
	}
	//cout<<equi_size.width<<"\t"<<equi_size.height<<endl;
	//cout<<"ALL:"<<count<<",14:"<<count14<<",16:"<<count16<<",18:"<<count18<<endl;
}

void Optimisor::drawComposition()
{
#if 1
	// draw composition image
	Mat comp = 0.5 * align_data.img_data[0].warp_imgs[0];

	for (size_t i = 1; i < 4; ++i)
		comp += 0.5 * align_data.img_data[i].warp_imgs[i];
	imwrite("aggregation.png", comp);

	for (int i = 0; i < align_data.img_data.size(); ++i)
	{
		Mat mesh_img = align_data.img_data[i].warp_imgs[i].clone();
		Mat src_mesh_img = align_data.img_data[i].scale_img.clone();
		ImageMesh &mesh = align_data.mesh_data[i].deform_meshes[i];
		ImageMesh src_mesh = align_data.mesh_data[i].ori_mesh;
		vector<ImageMesh> equi_meshes(3);
		equi_meshes[0].resize(mesh.size());
		equi_meshes[1].resize(mesh.size());
		equi_meshes[2].resize(mesh.size());
		for (size_t r = 0; r < mesh.size(); ++r)
		{
			Scalar color(rand() % 255, rand() % 255, rand() % 255);
#if LOOP_GRID
			equi_meshes[2][r].push_back(mesh[r][mesh[r].size() - 1]);
#endif
			for (size_t c = 0; c < mesh[r].size(); ++c)
			{
				equi_meshes[0][r].push_back(Point2f(mesh[r][c].x - mesh_img.cols, mesh[r][c].y));
				equi_meshes[1][r].push_back(mesh[r][c]);
				equi_meshes[2][r].push_back(Point2f(mesh[r][c].x + mesh_img.cols, mesh[r][c].y));
			}
#if LOOP_GRID

			equi_meshes[0][r].push_back(mesh[r][0]);
			equi_meshes[1][r].push_back(Point2f(mesh[r][0].x + mesh_img.cols, mesh[r][0].y));
			src_mesh[r].push_back(Point2f(src_mesh[r][0].x + mesh_img.cols, src_mesh[r][0].y));
#endif
		}
		// draw lines of deform mesh
		for (int k = 0; k < 3; ++k)
			drawMesh(equi_meshes[k], mesh_img);
		drawMesh(src_mesh, src_mesh_img);
		// draw grid points of deform mesh
		vector<Scalar> row_colors(src_mesh.size());
		for (size_t r = 0; r < row_colors.size(); ++r)
			row_colors[r] = Scalar(rand() % 255, rand() % 255, rand() % 255);
		for (int k = 0; k < 3; ++k)
		{
			for (size_t r = 0; r < equi_meshes[k].size(); ++r)
			{

				for (size_t c = 0; c < equi_meshes[k][r].size(); ++c)
					circle(mesh_img, equi_meshes[k][r][c], 10, row_colors[r], -1);
				for (size_t c = 0; c < src_mesh[r].size(); ++c)
					circle(src_mesh_img, src_mesh[r][c], 10, row_colors[r], -1);
			}
		}

		stringstream sstr;
		sstr << "deform-mesh-" << i << ".jpg";
		imwrite(sstr.str(), mesh_img);
		sstr.str("");
		sstr << "source-mesh-" << i << ".jpg";
		imwrite(sstr.str(), src_mesh_img);
	}
#endif
}

double Optimisor::linearSolve()
{
	cout << "Aggregation..." << endl;
	double e = 0.0;
	vector<Correspondence> corr;
	vector<Point2f> anchor_points;
	getPairCorrespondence(corr);
	getAnchorPoints(corr, anchor_points);
	cout << "#corr:" << corr.size() << ", #anchor:" << anchor_points.size() << endl;

	// set linear system
	int row_count = 0;
	vector<vector<double>> matrix_val;
	vector<double> b;
	// === add constraint of correspondence term ===== //
	//corrConstraint(matrix_val, b, row_count, corr);
	// === add anchor points ============//
	anchorConstraint(matrix_val, b, row_count, corr, anchor_points);
	// === add constraint of distortion term ===== //

	cout << "	distortion constraint" << endl;
	for (size_t i = 0; i < align_data.img_data.size(); ++i)
		distConstraint(matrix_val, b, i, row_count);

	cout << "	length constraint" << endl;
	for (size_t i = 0; i < align_data.img_data.size(); ++i)
		lengthConstraint(matrix_val, b, i, row_count);

	// === add constraint of smoothness term ===== //

	cout << "	smooth constraint" << endl;
	for (size_t i = 0; i < align_data.img_data.size(); ++i)
		smoothConstraint(matrix_val, b, i, row_count);

	// === add constraint of origin term (boundary condition) =====//

	cout << "	origin constraint" << endl;
	for (size_t i = 0; i < align_data.img_data.size(); ++i)
		originConstraint(matrix_val, b, i, row_count);

	// Transfer the linear system into Eigen interface
	unsigned long startTime = clock();
	int num_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size();
	int num_img = align_data.img_data.size();
	Eigen::SparseMatrix<double> A(row_count, 2 * num_vert * num_img);
	A.reserve(matrix_val.size());
	for (size_t i = 0; i < matrix_val.size(); ++i)
	{
		A.insert((int)matrix_val[i][0], (int)matrix_val[i][1]) = matrix_val[i][2];
	}

	Eigen::SparseMatrix<double> AT = A.transpose();

	Eigen::VectorXd B(row_count);
	for (size_t i = 0; i < b.size(); ++i)
		B[i] = b[i];

	B = AT * B; //b = A'*b
	B = (-1) * B;

	A = AT * A; //A = A'*A
	A = (-1) * A;
	A.makeCompressed();

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> linearSolver;
	linearSolver.compute(A);

	/*int num_row = align_data.mesh_data[0].ori_mesh.size()-1;
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	vector<Size> grid_size(align_data.img_data.size());
	for(size_t i = 0; i < grid_size.size(); ++i){
		grid_size[i].width = (float)(align_data.img_data[i].scale_img.cols-1)/(float)num_col;
		grid_size[i].height = (float)(align_data.img_data[i].scale_img.rows-1)/(float)num_row;
	}


	Eigen::VectorXd X0(2*num_vert*num_img) ;
	for(size_t i = 0; i < align_data.img_data.size(); ++i){
		int index = 0;
		for(size_t r = 0; r < num_row+1; ++r){
			for(size_t c = 0; c < num_col; ++c){
				X0[2*num_vert*i+index] = c*grid_size[i].width;
				X0[2*num_vert*i+index+num_vert] = r*grid_size[i].height;
				index++;
			}
		}
	}*/
	//const Eigen::VectorXd x0=X0 ;
	unsigned long constructMatrixTime = clock(); //construct matrix time

	/*for(int i=0;i<2*num_vert*num_img;i++)
	{
		cout <<int(X0[i])<<" ";
		if((i+1)%(align_data.mesh_data[0].ori_mesh[0].size())==0)
			cout<<endl;
		//X3[i]=10000.0f;
	}*/

	Eigen::VectorXd X = linearSolver.solve(B);
	//Eigen::VectorXd X = linearSolver.solveWithGuess(B,X0);
	//X = A * X -B;
	//cout<<A;
	unsigned long solveTime = clock();
	for (int i = 0; i < 2 * num_vert * num_img; i++)
	{
		cout << B[i] << " ";
		if ((i + 1) % (align_data.mesh_data[0].ori_mesh[0].size()) == 0)
			cout << endl;
		//X3[i]=10000.0f;
	}

	//Taucs solver
	// Transfer the linear system into Taucs interface
	/*unsigned long startTime = clock();

	int num_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size();
	int num_img = align_data.img_data.size();
	InitTaucsInterface();
	int A_ID = CreateMatrix(row_count, 2 * num_vert * num_img);
	for(size_t i = 0; i < matrix_val.size(); ++i){
		SetMatrixEntry(A_ID, (int)matrix_val[i][0], (int)matrix_val[i][1], matrix_val[i][2]);
	}
	taucsType* B = new taucsType[row_count];
	for(size_t i = 0; i < b.size(); ++i)
		B[i] = b[i];
	taucsType* X = new taucsType[2 * num_vert * num_img];
	
	unsigned long constructMatrixTime = clock(); //construct matrix time

	bool solve = SolveATA(A_ID, B, X, 1);

	unsigned long solveTime = clock(); //construct matrix time*/

	cout << "construct time: " << (constructMatrixTime - startTime) / 1000.0 << endl;
	cout << "solve time: " << (solveTime - constructMatrixTime) / 1000.0 << endl;
	// update mesh of each image (store the deform mesh at same id)
	vector<Mat> disp_maps(align_data.img_data.size());
	for (size_t i = 0; i < align_data.img_data.size(); ++i)
	{
		disp_maps[i] = Mat::zeros(align_data.img_data[i].scale_img.size(), CV_32FC1);
		ImageMesh &in_mesh = align_data.mesh_data[i].ori_mesh;
		ImageMesh &deform_mesh = align_data.mesh_data[i].deform_meshes[i];
		deform_mesh.resize(in_mesh.size());
		int idx = 0;
		for (size_t r = 0; r < deform_mesh.size(); ++r)
		{
			deform_mesh[r].resize(in_mesh[r].size());
			for (size_t c = 0; c < deform_mesh[r].size(); ++c)
			{
				deform_mesh[r][c].x = X[2 * num_vert * i + idx];
				deform_mesh[r][c].y = X[2 * num_vert * i + idx + num_vert];
				cout << deform_mesh[r][c].x << "," << deform_mesh[r][c].y << endl;
				idx++;
			}
		}
		Mat &in_mask = align_data.img_data[i].scale_mask;
		Mat &out_mask = align_data.img_data[i].warp_masks[i];
		Mat &in_img = align_data.img_data[i].scale_img;
		Mat &out_img = align_data.img_data[i].warp_imgs[i];
		getWarpImage(in_img, out_img, in_mask, out_mask, in_mesh, deform_mesh);
	}
	// leave taucs interface
	//ReleaseMatrix(A_ID);
	//DeinitTaucsInterface();
	//
	drawComposition();

	return e;
}

void TempTransform(const vector<Constrain> &constrain_list,
                 vector<vector<double>> &matrix_val,
                 vector<double> &b,
                 int &row_count,
                 double w)
{
	//cout << constrain_list.size() << endl;
	for (auto &constrain : constrain_list)
	{
		for (auto &coefficient : constrain.coefficients)
		{
			vector<double> val(3);
			val[0] = row_count;
			val[1] = int(coefficient.first);
			val[2] = coefficient.second * w;

			// debug
			/*
			cout << "============" << endl;
            cout << "r=" << val[0] << endl;
            cout << "c=" << val[1] << endl;
            cout << "v=" << val[2] << endl;
			*/
			
            matrix_val.push_back(val);        
		}
		
		//debug
		/*
		static int dragonkaodebug = 10;
		assert(dragonkaodebug);
		dragonkaodebug --;
		*/
		
        b.push_back(constrain.b * w);

		row_count++;
	}
}

double Optimisor::linearSolve2()
{
	cout << "Aggregation..." << endl;
	double e = 0.0;
	vector<Correspondence> corr;
	vector<vector<Vec3d>> depth_points;
	depth_points.resize(align_data.frame_size);
	//getPairCorrespondence(corr);
	for (int i = 0; i < align_data.frame_size; i++)
	{
		getDepthPoints(align_data.corr_data[i].correspon, depth_points[i]);
		cout << "#corr:" << corr.size() << ", #anchor:" << depth_points[i].size() << endl;
	}
	int frame_num = align_data.frame_size;
	int vert_num = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size();
	// set linear system

	bool left = false;
	vector<vector<double>> left_val;
	int frame_limit = 20;
	int set_num = frame_num / frame_limit;
	for (int s = 0; s < set_num; s++)
	{
		int row_count = 0;
		vector<vector<double>> matrix_val;
		vector<double> b;

		// === add anchor points ============//
		
		//for (int i = 0; i < frame_limit; i++)
		//	depthConstraint(matrix_val, b, row_count, depth_points[frame_limit * s + i], i);
		
		const GridInfo grid_info(20, 20, 20);
		for (int iii = 0; iii < 20; iii++)
		{
			vector<DepthPoint> my_dp_list;
			for (auto &dp : depth_points[iii])
			{
				DepthPoint my_dp;
				my_dp.theta = dp[1];
				my_dp.phi = dp[2];
				my_dp.depth = dp[0];
				my_dp.frame_index = iii;
				//if(my_dp.theta>6.28)
				//	cout << "theta=" << my_dp.theta << endl;
				//if(my_dp.phi>3.14)
				//	cout << "phi=" << my_dp.phi << endl;
				my_dp_list.push_back(my_dp);
			}
			TempTransform(GetDepthConstraint(grid_info, my_dp_list),
                     matrix_val,
                     b,
                     row_count,
                     align_data.feat_weight);
		}

		// === add constraint of smoothness term ===== //
		cout << "smooth constraint" << endl;
		for (int i = 0; i < frame_limit; i++)
			smoothConstraint2(matrix_val, b, row_count, i);

		/// === add constraint of time smoothness term ===== //
		int max_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size() * frame_limit;
		cout << "temporal constraint" << endl;
		if (frame_num > 1)
			temperalSmoothConstraint(matrix_val, b, row_count, max_vert);
		// === add constraint of origin term (boundary condition) =====//
		/*cout << "	origin constraint"<<endl;
		for(size_t i = 0; i < align_data.img_data.size(); ++i)
			originConstraint(matrix_val, b, i, row_count);*/

		// Transfer the linear system into Eigen interface

		unsigned long startTime = clock();
		int num_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size();
		int num_img = align_data.img_data.size();
		int num_col = align_data.mesh_data[0].ori_mesh[0].size(); //10
		int num_row = align_data.mesh_data[0].ori_mesh.size();	//11
		Eigen::SparseMatrix<double> A(row_count, num_vert * frame_num);
		A.reserve(matrix_val.size());
		for (size_t i = 0; i < matrix_val.size(); ++i)
		{
			//cout << i << endl;
			A.insert((int)matrix_val[i][0], (int)matrix_val[i][1]) = matrix_val[i][2];
		}
		cout << "constrcut done" << endl;
		Eigen::SparseMatrix<double> AT = A.transpose();

		Eigen::VectorXd B(row_count);
		for (size_t i = 0; i < b.size(); ++i)
			B[i] = b[i];

		B = AT * B; //b = A'*b
		B = (-1) * B;

		A = AT * A; //A = A'*A
		A = (-1) * A;

		A.makeCompressed();

		Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> linearSolver;
		linearSolver.compute(A);

		unsigned long constructMatrixTime = clock(); //construct matrix time

		Eigen::VectorXd X3 = linearSolver.solve(B);
		cout << "over otimizes" << endl;

		for (int i = 0; i < num_vert * frame_limit; i++)
		{
			int count = 0;
			while ((X3[i] < 1000))
			{
				//cout<<i<<" ";
				int left = -1, right = 1;
				if (i % num_col == 0)
					left = num_col - 1;
				if (i % num_col == (num_col - 1))
					right = -num_col + 1;
				if ((i % num_vert) < num_col)
					X3[i] = (X3[i + left] + X3[i + right] + X3[i + num_col]) / 3;
				else if ((i % num_vert) >= (num_vert - num_col))
					X3[i] = (X3[i + left] + X3[i + right] + X3[i - num_col]) / 3;
				else
					X3[i] = (X3[i + left] + X3[i + right] + X3[i - num_col] + X3[i + num_col]) / 4;

				if (count >= 1)
				{
					X3[i] = 1000;
				}
				count++;
			}
			/*cout <<int(X3[i])<<"\t";
			if((i+1)%(align_data.mesh_data[0].ori_mesh[0].size())==0)
				cout<<endl;
			if((i+1)%(vert_num)==0)
				cout<<endl<<endl;*/
		}
		//X3[i] = 10000;
		for (int i = 0; i < num_vert * frame_limit; i++)
		{
			int left = -1, right = 1;
			if (i % num_col == 0)
				left = num_col - 1;
			if (i % num_col == (num_col - 1))
				right = -num_col + 1;
			if (i < num_col)
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i + num_col]) / 4;
			else if ((i + num_col) >= num_vert)
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i - num_col]) / 4;
			else
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i - num_col] + X3[i + num_col]) / 5;
			/*cout <<int(X3[i])<<"\t";
			if((i+1)%(align_data.mesh_data[0].ori_mesh[0].size())==0)
				cout<<endl;*/
			//X3[i]=10000.0f;
		}
		cout << endl;
		Size equi_size = align_data.img_data[0].warp_imgs[0].size();
		float gw = equi_size.width / num_col;
		float gh = equi_size.height / num_row;
		char output[30];
		for (int k = 0; k < frame_limit; k++)
		{
			sprintf(output, "deform/deform_%d.txt", s * frame_limit + k);
			ofstream fs(output);
			for (int i = 0; i < num_row; i++)
			{
				for (int j = 0; j <= num_col; j++)
				{
					Point2f pe(j * gw, i * gh);
					Vec3d ps;
					equi2Sphere(pe, ps, equi_size);
					if (j != num_col)
						ps *= X3[k * vert_num + i * num_col + j];
					else
						ps *= X3[k * vert_num + i * num_col];
					//if(ps[1]>0)
					//fs<<int(ps[0])<<"\t"<<int(ps[1])<<"\t"<<int(ps[2])<<endl;
					fs << i * (num_col + 1) + j << "\t" << ps[0] << "\t" << ps[1] << "\t" << ps[2] << endl;
				}
			}
			fs.close();
		}

		unsigned long solveTime = clock();

		cout << "construct time: " << (constructMatrixTime - startTime) / 1000.0 << endl;
		cout << "solve time: " << (solveTime - constructMatrixTime) / (1000.0 * frame_limit) << endl;
	}
	int over_frame = set_num * frame_limit;
	int left_frame = frame_num - over_frame;

	if (left_frame > 0)
	{
		int row_count = 0;
		vector<vector<double>> matrix_val;
		vector<double> b;
		// === add constraint of correspondence term ===== //
		//corrConstraint(matrix_val, b, row_count, corr);
		// === add anchor points ============//
		//change here
		for (int i = 0; i < left_frame; i++)
			depthConstraint(matrix_val, b, row_count, depth_points[over_frame + i], i);
		//anchorConstraint(matrix_val, b, row_count, corr, anchor_points);

		// === add constraint of smoothness term ===== //
		cout << "smooth constraint" << endl;
		for (int i = 0; i < left_frame; i++)
			smoothConstraint2(matrix_val, b, row_count, i);

		/// === add constraint of time smoothness term ===== //
		int max_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size() * left_frame;
		cout << "temporal constraint" << endl;
		if (left_frame > 1)
			temperalSmoothConstraint(matrix_val, b, row_count, max_vert);
		// === add constraint of origin term (boundary condition) =====//
		/*cout << "	origin constraint"<<endl;
	for(size_t i = 0; i < align_data.img_data.size(); ++i)
		originConstraint(matrix_val, b, i, row_count);*/

		// Transfer the linear system into Eigen interface

		unsigned long startTime = clock();
		int num_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size();
		int num_img = align_data.img_data.size();
		int num_col = align_data.mesh_data[0].ori_mesh[0].size(); //10
		int num_row = align_data.mesh_data[0].ori_mesh.size();	//11
		Eigen::SparseMatrix<double> A(row_count, num_vert * frame_num);
		A.reserve(matrix_val.size());
		for (size_t i = 0; i < matrix_val.size(); ++i)
		{
			A.insert((int)matrix_val[i][0], (int)matrix_val[i][1]) = matrix_val[i][2];
		}
		cout << "constrcut done" << endl;
		Eigen::SparseMatrix<double> AT = A.transpose();

		Eigen::VectorXd B(row_count);
		for (size_t i = 0; i < b.size(); ++i)
			B[i] = b[i];

		B = AT * B; //b = A'*b
		B = (-1) * B;

		A = AT * A; //A = A'*A
		A = (-1) * A;

		A.makeCompressed();

		Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> linearSolver;
		linearSolver.compute(A);

		unsigned long constructMatrixTime = clock(); //construct matrix time

		Eigen::VectorXd X3 = linearSolver.solve(B);
		cout << "over otimizes" << endl;

		//Taucs solver
		// Transfer the linear system into Taucs interface
		/*	unsigned long startTime = clock();

	int num_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size();
	int num_img = align_data.img_data.size();
	InitTaucsInterface();
	int A_ID = CreateMatrix(row_count, num_vert * num_img);
	for(size_t i = 0; i < matrix_val.size(); ++i){
		SetMatrixEntry(A_ID, (int)matrix_val[i][0], (int)matrix_val[i][1], matrix_val[i][2]);
	}
	taucsType* B = new taucsType[row_count];
	for(size_t i = 0; i < b.size(); ++i)
		B[i] = b[i];
	taucsType* X = new taucsType[num_vert * num_img];
	
	unsigned long constructMatrixTime = clock(); //construct matrix time

	bool solve = SolveATA(A_ID, B, X, 1);
	*/

		for (int i = 0; i < num_vert * left_frame; i++)
		{
			int count = 0;
			while ((X3[i] < 1000))
			{
				//cout<<i<<" ";
				int left = -1, right = 1;
				if (i % num_col == 0)
					left = num_col - 1;
				if (i % num_col == (num_col - 1))
					right = -num_col + 1;
				if ((i % num_vert) < num_col)
					X3[i] = (X3[i + left] + X3[i + right] + X3[i + num_col]) / 3;
				else if ((i % num_vert) >= (num_vert - num_col))
					X3[i] = (X3[i + left] + X3[i + right] + X3[i - num_col]) / 3;
				else
					X3[i] = (X3[i + left] + X3[i + right] + X3[i - num_col] + X3[i + num_col]) / 4;

				if (count >= 1)
				{
					X3[i] = 1000;
				}
				count++;
			}
			/*cout <<int(X3[i])<<"\t";
			if((i+1)%(align_data.mesh_data[0].ori_mesh[0].size())==0)
				cout<<endl;
			if((i+1)%(vert_num)==0)
				cout<<endl<<endl;*/
		}
		//X3[i] = 10000;
		for (int i = 0; i < num_vert * left_frame; i++)
		{
			int left = -1, right = 1;
			if (i % num_col == 0)
				left = num_col - 1;
			if (i % num_col == (num_col - 1))
				right = -num_col + 1;
			if (i < num_col)
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i + num_col]) / 4;
			else if ((i + num_col) >= num_vert)
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i - num_col]) / 4;
			else
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i - num_col] + X3[i + num_col]) / 5;
			/*cout <<int(X3[i])<<"\t";
		if((i+1)%(align_data.mesh_data[0].ori_mesh[0].size())==0)
			cout<<endl;*/
			//X3[i]=10000.0f;
		}
		cout << endl;
		Size equi_size = align_data.img_data[0].warp_imgs[0].size();
		float gw = equi_size.width / num_col;
		float gh = equi_size.height / num_row;
		char output[30];

		for (int k = 0; k < left_frame; k++)
		{
			sprintf(output, "deform/deform_%d.txt", over_frame + k);
			ofstream fs(output);
			for (int i = 0; i < num_row; i++)
			{
				for (int j = 0; j <= num_col; j++)
				{
					Point2f pe(j * gw, i * gh);
					Vec3d ps;
					equi2Sphere(pe, ps, equi_size);
					if (j != num_col)
						ps *= X3[k * vert_num + i * num_col + j];
					else
						ps *= X3[k * vert_num + i * num_col];
					//if(ps[1]>0)
					//fs<<int(ps[0])<<"\t"<<int(ps[1])<<"\t"<<int(ps[2])<<endl;
					fs << i * (num_col + 1) + j << "\t" << ps[0] << "\t" << ps[1] << "\t" << ps[2] << endl;
				}
			}
			fs.close();
		}

		unsigned long solveTime = clock();

		cout << "construct time: " << (constructMatrixTime - startTime) / 1000.0 << endl;
		cout << "solve time: " << (solveTime - constructMatrixTime) / (1000.0 * frame_limit) << endl;
	}

	//transform sphere into equi
	/*Eigen::VectorXd X(2*num_vert*4);
	transformSphere2equi( X, X3);
	StichBySphere(X3);
	// update mesh of each image (store the deform mesh at same id)
	vector<Mat> disp_maps(align_data.img_data.size());
	for(size_t i = 0; i < align_data.img_data.size(); ++i){
		disp_maps[i] = Mat::zeros(align_data.img_data[i].scale_img.size(), CV_32FC1);
		ImageMesh& in_mesh = align_data.mesh_data[i].ori_mesh;
		ImageMesh& deform_mesh = align_data.mesh_data[i].deform_meshes[i];
		deform_mesh.resize(in_mesh.size());
		int idx = 0;
		for(size_t r = 0; r < deform_mesh.size(); ++r){
			deform_mesh[r].resize(in_mesh[r].size());
			for(size_t c = 0; c < deform_mesh[r].size(); ++c){
				deform_mesh[r][c].x = X[2*num_vert*i+idx];
				deform_mesh[r][c].y = X[2*num_vert*i+idx+num_vert];
				idx++;
			}
		}
		Mat& in_mask = align_data.img_data[i].scale_mask;
		Mat& out_mask = align_data.img_data[i].warp_masks[i];
		Mat& in_img = align_data.img_data[i].scale_img;
		Mat& out_img = align_data.img_data[i].warp_imgs[i];
		getWarpImage(in_img, out_img, in_mask, out_mask, in_mesh, deform_mesh);
	}
	// leave taucs interface
	//ReleaseMatrix(A_ID);
	//DeinitTaucsInterface();
	//
	drawComposition();*/

	return e;
}

double Optimisor::linearSolve3()
{
	cout << "Aggregation..." << endl;
	double e = 0.0;
	vector<Correspondence> corr;
	vector<vector<Vec3d>> depth_points;
	depth_points.resize(align_data.frame_size);
	//getPairCorrespondence(corr);
	for (int i = 0; i < align_data.frame_size; i++)
	{
		getDepthPoints(align_data.corr_data[i].correspon, depth_points[i]);
		cout << "#corr:" << corr.size() << ", #anchor:" << depth_points[i].size() << endl;
	}
	int frame_num = align_data.frame_size;
	// set linear system
	for (int f = 0; f < frame_num; f++)
	{
		int row_count = 0;
		vector<vector<double>> matrix_val;
		vector<double> b;
		// === add constraint of correspondence term ===== //
		//corrConstraint(matrix_val, b, row_count, corr);
		// === add anchor points ============//
		//change here
		depthConstraint(matrix_val, b, row_count, depth_points[f], 0);
		//anchorConstraint(matrix_val, b, row_count, corr, anchor_points);

		// === add constraint of smoothness term ===== //
		cout << "smooth constraint" << endl;
		smoothConstraint2(matrix_val, b, row_count, 0);

		// === add constraint of origin term (boundary condition) =====//
		/*cout << "	origin constraint"<<endl;
	for(size_t i = 0; i < align_data.img_data.size(); ++i)
		originConstraint(matrix_val, b, i, row_count);*/

		// Transfer the linear system into Eigen interface

		unsigned long startTime = clock();
		int num_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size();
		int num_img = align_data.img_data.size();
		int num_col = align_data.mesh_data[0].ori_mesh[0].size(); //10
		int num_row = align_data.mesh_data[0].ori_mesh.size();	//11
		Eigen::SparseMatrix<double> A(row_count, num_vert * frame_num);
		A.reserve(matrix_val.size());
		for (size_t i = 0; i < matrix_val.size(); ++i)
		{
			A.insert((int)matrix_val[i][0], (int)matrix_val[i][1]) = matrix_val[i][2];
		}
		cout << "constrcut done" << endl;
		Eigen::SparseMatrix<double> AT = A.transpose();

		Eigen::VectorXd B(row_count);
		for (size_t i = 0; i < b.size(); ++i)
			B[i] = b[i];

		B = AT * B; //b = A'*b
		B = (-1) * B;

		A = AT * A; //A = A'*A
		A = (-1) * A;

		A.makeCompressed();

		Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> linearSolver;
		linearSolver.compute(A);

		unsigned long constructMatrixTime = clock(); //construct matrix time

		Eigen::VectorXd X3 = linearSolver.solve(B);
		cout << "over " << f << "-th otimizes" << endl;

		//Taucs solver
		// Transfer the linear system into Taucs interface
		/*	unsigned long startTime = clock();

	int num_vert = align_data.mesh_data[0].ori_mesh.size() * align_data.mesh_data[0].ori_mesh[0].size();
	int num_img = align_data.img_data.size();
	InitTaucsInterface();
	int A_ID = CreateMatrix(row_count, num_vert * num_img);
	for(size_t i = 0; i < matrix_val.size(); ++i){
		SetMatrixEntry(A_ID, (int)matrix_val[i][0], (int)matrix_val[i][1], matrix_val[i][2]);
	}
	taucsType* B = new taucsType[row_count];
	for(size_t i = 0; i < b.size(); ++i)
		B[i] = b[i];
	taucsType* X = new taucsType[num_vert * num_img];
	
	unsigned long constructMatrixTime = clock(); //construct matrix time

	bool solve = SolveATA(A_ID, B, X, 1);
	*/

		for (int i = 0; i < num_vert; i++)
		{
			int count = 0;
			while ((X3[i] < 1000))
			{
				//cout<<i<<" ";
				int left = -1, right = 1;
				if (i % num_col == 0)
					left = num_col - 1;
				if (i % num_col == (num_col - 1))
					right = -num_col + 1;
				if ((i % num_vert) < num_col)
					X3[i] = (X3[i + left] + X3[i + right] + X3[i + num_col]) / 3;
				else if ((i % num_vert) >= (num_vert - num_col))
					X3[i] = (X3[i + left] + X3[i + right] + X3[i - num_col]) / 3;
				else
					X3[i] = (X3[i + left] + X3[i + right] + X3[i - num_col] + X3[i + num_col]) / 4;

				if (count >= 1)
				{
					X3[i] = 1000;
				}
				count++;
			}
		}
		//X3[i] = 10000;
		for (int i = 0; i < num_vert; i++)
		{
			int left = -1, right = 1;
			if (i % num_col == 0)
				left = num_col - 1;
			if (i % num_col == (num_col - 1))
				right = -num_col + 1;
			if (i < num_col)
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i + num_col]) / 4;
			else if ((i + num_col) >= num_vert)
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i - num_col]) / 4;
			else
				X3[i] = (X3[i] + X3[i + left] + X3[i + right] + X3[i - num_col] + X3[i + num_col]) / 5;
			/*cout <<int(X3[i])<<"\t";
		if((i+1)%(align_data.mesh_data[0].ori_mesh[0].size())==0)
			cout<<endl;*/
			//X3[i]=10000.0f;
		}
		cout << endl;
		Size equi_size = align_data.img_data[0].warp_imgs[0].size();
		float gw = equi_size.width / num_col;
		float gh = equi_size.height / num_row;
		char output[30];

		/*sprintf(output, "deform/deform_%d.txt", f);
	ofstream fs(output);
	for(int i=0;i<num_row;i++)
	{
		for(int j=0;j<=num_col;j++)
		{
			Point2f pe(j*gw,i*gh);
			Vec3d ps;
			equi2Sphere( pe, ps, equi_size);
			if(j!=num_col)
					ps *= X3[i*num_col+j];
				else
					ps *= X3[i*num_col];
			//if(ps[1]>0)
				//fs<<int(ps[0])<<"\t"<<int(ps[1])<<"\t"<<int(ps[2])<<endl;
			fs<<i*(num_col+1)+j<<"\t"<<ps[0]<<"\t"<<ps[1]<<"\t"<<ps[2]<<endl;
		}
	}
	fs.close();*/
		sprintf(output, "point/point_%d.txt", f);
		ofstream fs(output);
		for (int i = 0; i < num_row; i++)
		{
			for (int j = 0; j <= num_col; j++)
			{
				Point2f pe(j * gw, i * gh);
				float r;

				if (j != num_col)
					r = X3[i * num_col + j];
				else
					r = X3[i * num_col];
				//if(ps[1]>0)
				//fs<<int(ps[0])<<"\t"<<int(ps[1])<<"\t"<<int(ps[2])<<endl;
				if ((i == 0) || ((i + 1) == num_row))
					fs << 1 << endl;
				else
					fs << int(r / 1000) << endl;
			}
		}
		fs.close();

		unsigned long solveTime = clock();

		cout << "construct time: " << (constructMatrixTime - startTime) / 1000.0 << endl;
		cout << "solve time: " << (solveTime - constructMatrixTime) / 1000.0 << endl;
	}
	//transform sphere into equi
	/*Eigen::VectorXd X(2*num_vert*4);
	transformSphere2equi( X, X3);
	StichBySphere(X3);
	// update mesh of each image (store the deform mesh at same id)
	vector<Mat> disp_maps(align_data.img_data.size());
	for(size_t i = 0; i < align_data.img_data.size(); ++i){
		disp_maps[i] = Mat::zeros(align_data.img_data[i].scale_img.size(), CV_32FC1);
		ImageMesh& in_mesh = align_data.mesh_data[i].ori_mesh;
		ImageMesh& deform_mesh = align_data.mesh_data[i].deform_meshes[i];
		deform_mesh.resize(in_mesh.size());
		int idx = 0;
		for(size_t r = 0; r < deform_mesh.size(); ++r){
			deform_mesh[r].resize(in_mesh[r].size());
			for(size_t c = 0; c < deform_mesh[r].size(); ++c){
				deform_mesh[r][c].x = X[2*num_vert*i+idx];
				deform_mesh[r][c].y = X[2*num_vert*i+idx+num_vert];
				idx++;
			}
		}
		Mat& in_mask = align_data.img_data[i].scale_mask;
		Mat& out_mask = align_data.img_data[i].warp_masks[i];
		Mat& in_img = align_data.img_data[i].scale_img;
		Mat& out_img = align_data.img_data[i].warp_imgs[i];
		getWarpImage(in_img, out_img, in_mask, out_mask, in_mesh, deform_mesh);
	}
	// leave taucs interface
	//ReleaseMatrix(A_ID);
	//DeinitTaucsInterface();
	//
	drawComposition();*/

	return e;
}

void Optimisor::initMesh(Rect &bb, int num_vert_row, ImageMesh &in_mesh, ImageMesh &deform_mesh)
{
	in_mesh.resize(num_vert_row);
	Size grid_size;
	grid_size.width = bb.width;
	grid_size.height = bb.height;
	grid_size.width /= (float)(num_vert_row - 1);
	grid_size.height /= (float)(num_vert_row - 1);
	bool is_loop;
	if (bb.width == align_data.img_data[0].scale_img.cols - 1)
		is_loop = true;
	else
		is_loop = false;

	for (size_t r = 0; r < in_mesh.size(); ++r)
	{
#if NON_LOOP_GRID
		in_mesh[r].resize(num_vert_row);
#endif
#if LOOP_GRID
		// check: whether the bounding box reacches image boundary

		if (is_loop)
			in_mesh[r].resize(num_vert_row - 1);
		else
			in_mesh[r].resize(num_vert_row);
#endif

		for (size_t c = 0; c < in_mesh[r].size(); ++c)
		{
			in_mesh[r][c].x = c * grid_size.width + bb.x;
			in_mesh[r][c].y = r * grid_size.height + bb.y;
		}
	}
	deform_mesh.resize(num_vert_row);
	for (size_t r = 0; r < deform_mesh.size(); ++r)
	{
#if NON_LOOP_GRID
		deform_mesh[r].resize(num_vert_row);
#endif
#if LOOP_GRID
		if (is_loop)
			deform_mesh[r].resize(num_vert_row - 1);
		else
			deform_mesh[r].resize(num_vert_row);
#endif
	}
}
void Optimisor::adaptiveMesh(Rect &bb, ImageMesh &in_mesh, ImageMesh &deform_mesh)
{
	int num_vert_row, num_vert_col;
	// determine num_vert_row and num_vert_col according to bounding box
	float step = 20;
	float min_gw = (float)align_data.img_data[0].scale_img.cols / step;
	float min_gh = (float)align_data.img_data[0].scale_img.cols / step;
	int min_verts = 4;
	num_vert_row = (float)bb.height / min_gh + 2;
	num_vert_col = (float)bb.width / min_gw + 2;
	num_vert_row = (num_vert_row > min_verts) ? min_verts : num_vert_row;
	num_vert_col = (num_vert_col > min_verts) ? min_verts : num_vert_col;
	in_mesh.resize(num_vert_row);
	Size grid_size;
	grid_size.width = bb.width;
	grid_size.height = bb.height;
	grid_size.width /= (float)(num_vert_col - 1);
	grid_size.height /= (float)(num_vert_row - 1);
	bool is_loop;
	if (bb.width == align_data.img_data[0].scale_img.cols - 1)
		is_loop = true;
	else
		is_loop = false;

	for (size_t r = 0; r < in_mesh.size(); ++r)
	{
#if NON_LOOP_GRID
		in_mesh[r].resize(num_vert_col);
#endif
#if LOOP_GRID
		// check: whether the bounding box reacches image boundary

		if (is_loop)
			in_mesh[r].resize(num_vert_col - 1);
		else
			in_mesh[r].resize(num_vert_col);
#endif

		for (size_t c = 0; c < in_mesh[r].size(); ++c)
		{
			in_mesh[r][c].x = c * grid_size.width + bb.x;
			in_mesh[r][c].y = r * grid_size.height + bb.y;
		}
	}
	deform_mesh.resize(num_vert_row);
	for (size_t r = 0; r < deform_mesh.size(); ++r)
	{
#if NON_LOOP_GRID
		deform_mesh[r].resize(num_vert_col);
#endif
#if LOOP_GRID
		if (is_loop)
			deform_mesh[r].resize(num_vert_col - 1);
		else
			deform_mesh[r].resize(num_vert_col);
#endif
	}
}

void Optimisor::getPairCorrespondence(vector<Correspondence> &corr)
{
	// add original feature points in align_data
	for (size_t i = 0; i < align_data.feature_graph.size(); ++i)
	{
		for (size_t j = 0; j < align_data.feature_graph[i].feat_info.size(); ++j)
		{
			for (size_t f = 0; f < align_data.feature_graph[i].feat_info[j].size(); ++f)
			{
				Correspondence C;
				C.fm = align_data.feature_graph[i].feat_info[j][f].pos;
				C.fn = align_data.feature_graph[j].feat_info[i][f].pos;
				C.m = i;
				C.n = j;
				corr.push_back(C);
			}
		}
	}

	// draw correspondence between 0 and 3
	Scalar color[4];
	for (int i = 0; i < 4; ++i)
	{
		int b = rand() % 255;
		int g = rand() % 255;
		int r = rand() % 255;
		color[i] = Scalar(b, g, r);
	}
#if 0
	int num_cam = align_data.img_data.size();
	for(int i = 0; i < 4; ++i)
	{
		int j = (i == num_cam-1) ? 0 : i+1;
		Mat& img_i = align_data.img_data[i].scale_img;
		Mat& img_j = align_data.img_data[j].scale_img;
		
		if(i == 0 && j == 1)	
		for(int m = 0; m < align_data.feature_graph[i].feat_info[j].size(); ++m)
		{
			Point2f pt = align_data.feature_graph[i].feat_info[j][m].pos;
			Point2f ptm = align_data.feature_graph[j].feat_info[i][m].pos;	
			circle(img_i, pt, 2, color[i], -1);
			circle(img_j, ptm, 2, color[i], -1);
		}
		
		
	}
	for(int i = 0; i < align_data.img_data.size(); ++i)
	{
		stringstream sstr;
		sstr << "feat-"<<i<<".jpg";
		imwrite(sstr.str(), align_data.img_data[i].scale_img);
	}
	system("pause");
#endif
#if 0
	// extract correspondence of uniform grids according to warping result of pairwise alignment
	int num_row = align_data.mesh_data[0].ori_mesh.size()-1;
	int num_col = align_data.mesh_data[0].ori_mesh[0].size()-1;
	for(size_t m = 0; m < align_data.match_pairs.size(); ++m){
		int idx_i = align_data.match_pairs[m].first;
		int idx_j = align_data.match_pairs[m].second;
		int idx[2][2] = {{idx_i, idx_j}, {idx_j, idx_i}};
		for(int i = 0; i < 2; ++i){
			int idx_m = idx[i][0];
			int idx_n = idx[i][1];
			// create idx_img: each pixel records its grid index. (r,g,b):(10 * r, 10 * c, 1)
			Mat idx_img = Mat::zeros(align_data.img_data[idx_n].scale_img.size(), CV_8UC3);
			Mat warp_idx_img = Mat::zeros(align_data.img_data[idx_n].warp_imgs[idx_m].size(), CV_8UC3);
			ImageMesh& in_mesh_n = align_data.mesh_data[idx_n].ori_mesh;
			ImageMesh& deform_mesh_n = align_data.mesh_data[idx_n].deform_meshes[idx_m];
			for(size_t r = 0; r < num_row; ++r){
				for(size_t c = 0; c < num_col; ++c){
					int c_start = (int)in_mesh_n[r][c].x;		int c_end = (int)in_mesh_n[r][c+1].x;
					int r_start = (int)in_mesh_n[r][c].y;		int r_end = (int)in_mesh_n[r+1][c].y;
					int grid_w = c_end - c_start + 1;		int grid_h = r_end - r_start + 1;
					idx_img(Rect(c_start, r_start, grid_w, grid_h)).setTo(Scalar(10 * r, 10 * c, 1));
				}
			}
			imwrite("grid_idx_img.png", idx_img);
			getWarpImage(idx_img, warp_idx_img, in_mesh_n, deform_mesh_n);
			// Find out which grid the vertices of the deform mesh of image m to n are located on
			ImageMesh& in_mesh_m = align_data.mesh_data[idx_m].ori_mesh;
			ImageMesh& deform_mesh_m = align_data.mesh_data[idx_m].deform_meshes[idx_n];
			Mat &mask_m = align_data.img_data[idx_m].scale_mask;
			Mat &mask_n = align_data.img_data[idx_n].scale_mask;
			for(size_t r = 0; r < num_row; ++r){
				for(size_t c = 0; c < num_col; ++c){
					Point2f in_vert, out_vert;
					in_vert.x = in_mesh_m[r][c].x;	in_vert.y = in_mesh_m[r][c].y;
					out_vert.x = deform_mesh_m[r][c].x; out_vert.y = deform_mesh_m[r][c].y;
					if((out_vert.x < warp_idx_img.cols) && (out_vert.x >= 0) && (out_vert.y < warp_idx_img.rows) && (out_vert.y >=0))	{
						Vec3b s = warp_idx_img.at<Vec3b>(out_vert.y, out_vert.x);
						if(mask_m.at<uchar>(out_vert.y, out_vert.x) == 0)
							continue;
						if(!(((s.val[0] == 0) && (s.val[1] == 0) && (s.val[2] == 0)))) {
							int idx_r = (int)(warp_idx_img.at<Vec3b>(out_vert.y, out_vert.x)[0])/10;
							int idx_c = (int)(warp_idx_img.at<Vec3b>(out_vert.y, out_vert.x)[1])/10;

							vector<Point2f> src_vertex;		src_vertex.resize(4);
							vector<Point2f> des_vertex;		des_vertex.resize(4);
							getGridVertices(src_vertex, deform_mesh_n, idx_r, idx_c);
							getGridVertices(des_vertex, in_mesh_n, idx_r, idx_c);

							Mat H = findHomography(Mat(src_vertex), Mat(des_vertex), 0);
							// find corresponding location in image n for out_vert by Homography transformation 
							// ---- Mat(rows, cols, type=6)
							Mat P = Mat(3, 1, 6);	Mat in_P = Mat(3, 1, 6);
							P.at<double>(0,0)= out_vert.x;
							P.at<double>(1,0)= out_vert.y;
							P.at<double>(2,0)= 1.0;

							in_P = H*P;
							for(int i = 0; i < 3; ++i) {
								in_P.at<double>(i, 0) = in_P.at<double>(i, 0)/in_P.at<double>(2, 0);
							}
							Point2f pos;
							pos.x = in_P.at<double>(0, 0);
							pos.y = in_P.at<double>(1, 0);
							if(mask_n.at<uchar>(pos.y, pos.x) == 0)
								continue;

							if((pos.x >= 0) && (pos.y >= 0)	&& (pos.x < idx_img.cols) && (pos.y < idx_img.rows)) {
								Correspondence C;
								C.m = idx_m;
								C.n = idx_n;
								C.fm = in_vert;
								C.fn = pos;
								corr.push_back(C);
							}
						}
					}
				}
			}
		}
	}
#endif
}

void Optimisor::getGridVertices(vector<Point2f> &vertex, ImageMesh &mesh, const int &idx_r, const int &idx_c)
{
	// 0 -- 1
	// |    |
	// 2 -- 3	vertex index
	vertex[0].x = mesh[idx_r][idx_c].x;
	vertex[0].y = mesh[idx_r][idx_c].y;
	vertex[1].x = mesh[idx_r][idx_c + 1].x;
	vertex[1].y = mesh[idx_r][idx_c + 1].y;
	vertex[2].x = mesh[idx_r + 1][idx_c].x;
	vertex[2].y = mesh[idx_r + 1][idx_c].y;
	vertex[3].x = mesh[idx_r + 1][idx_c + 1].x;
	vertex[3].y = mesh[idx_r + 1][idx_c + 1].y;
}

void Optimisor::corrConstraint(vector<vector<double>> &matrix_val, vector<double> &b, int &row_count, vector<Correspondence> &corr)
{
	cout << "	correspondence constraint" << endl;
	// Compute alpha values and fill them to matrix
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int nv_col = num_col + 1;
	int num_vert = (num_row + 1) * (num_col + 1);
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int nv_col = num_col;
	int num_vert = (num_row + 1) * num_col;
#endif

	vector<Size> grid_size(align_data.img_data.size());
	for (size_t i = 0; i < grid_size.size(); ++i)
	{
		grid_size[i].width = (float)(align_data.img_data[i].scale_img.cols - 1) / (float)num_col;
		grid_size[i].height = (float)(align_data.img_data[i].scale_img.rows - 1) / (float)num_row;
	}

	// ---- Compute 'alpha value' & 'vertex index' of the features ----
	vector<double> constraint(3);
	vector<double> col_idx(8), val(8);
	//double w = align_data.feat_weight / (double)corr.size();
	double w = align_data.feat_weight;
	int temp = 0;
	for (size_t p = 0; p < corr.size(); ++p)
	{
		int m = corr[p].m;
		int n = corr[p].n;
		Point2f &fm = corr[p].fm;
		Point2f &fn = corr[p].fn;
		int idx[2] = {m, n};
		Point2f pos[2] = {fm, fn};
		Feature feat[2];
		double offset[2] = {0.0, 0.0};
		for (int i = 0; i < 2; ++i)
		{
			float gw = grid_size[idx[i]].width;
			float gh = grid_size[idx[i]].height;

			/*  alpha1-- left, alpha2--right
			*  alpha3-- up  , alpha4--down	 */
			double alpha1, alpha2, alpha3, alpha4;
			int c, r;
			c = (int)(pos[i].x / gw);
			if (c == num_col)
			{
				c--;
			}
			alpha2 = pos[i].x - c * gw;
			alpha2 = alpha2 / gw;
			alpha1 = 1 - alpha2;

			r = (int)(pos[i].y / gh);
			if (r == num_row)
			{
				r--;
			}
			alpha4 = pos[i].y - r * gh;
			alpha4 = alpha4 / gh;
			alpha3 = 1 - alpha4;

			feat[i].vertex[0] = nv_col * r + c;
			feat[i].vertex[1] = nv_col * r + c + 1;
			feat[i].vertex[2] = nv_col * (r + 1) + c;
			feat[i].vertex[3] = nv_col * (r + 1) + c + 1;
#if LOOP_GRID
			if (c == num_col - 1)
			{
				feat[i].vertex[1] = nv_col * r;
				feat[i].vertex[3] = nv_col * (r + 1);
				offset[i] = align_data.img_data[0].scale_img.cols - 1;
			}
#endif

			feat[i].alpha[0] = alpha1 * alpha3;
			feat[i].alpha[1] = alpha2 * alpha3;
			feat[i].alpha[2] = alpha1 * alpha4;
			feat[i].alpha[3] = alpha2 * alpha4;
			/*cout<<idx[i]<<":"<<pos[i].x<<","<<pos[i].y<<endl;
			/*cout<<r<<","<<c<<endl;
			cout<<feat[i].alpha[0]<<","<<feat[i].alpha[1]<<","<<feat[i].alpha[2]<<","<<feat[i].alpha[3]<<endl;*/
		}
		for (int i = 0; i < 2; ++i)
		{
			for (size_t v = 0; v < feat[i].vertex.size(); ++v)
			{
				col_idx[4 * i + v] = 2 * num_vert * idx[i] + feat[i].vertex[v];
				val[4 * i + v] = (i == 0) ? feat[i].alpha[v] * w : -feat[i].alpha[v] * w;
			}
		}
		// X coordinate
		for (size_t c_c = 0; c_c < col_idx.size(); ++c_c)
		{
			constraint[0] = row_count;
			constraint[1] = col_idx[c_c];
			constraint[2] = val[c_c];
			matrix_val.push_back(constraint);
		}
		row_count++;
		// Y coordinate
		for (size_t c_c = 0; c_c < col_idx.size(); ++c_c)
		{
			constraint[0] = row_count;
			constraint[1] = col_idx[c_c] + num_vert;
			constraint[2] = val[c_c];
			matrix_val.push_back(constraint);
		}
		row_count++;
#if NON_LOOP_GRID
		b.push_back(0);
		b.push_back(0);
#endif
#if LOOP_GRID
		//cout<<offset[1]<<","<<offset[0]<<endl;
		if (offset[1] == offset[0])
			b.push_back((val[5] + val[7]) * offset[1] - (val[1] + val[3]) * offset[0]);
		else
		{
			temp++;
			b.push_back((val[5] + val[7]) * offset[1] - (val[1] + val[3]) * offset[0] + (align_data.img_data[0].scale_img.cols - 1) * w);
		}
		//b.push_back(0);
		b.push_back(0);
#endif
	}
	cout << temp << endl;
}

void Optimisor::anchorConstraint(vector<vector<double>> &matrix_val, vector<double> &b, int &row_count,
								 vector<Correspondence> &corr, vector<Point2f> &anchor_points)
{
	cout << "	anchor point constraint" << endl;
	// Compute alpha values and fill them to matrix
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int nv_col = num_col + 1;
	int num_vert = (num_row + 1) * (num_col + 1);
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int nv_col = num_col;
	int num_vert = (num_row + 1) * num_col;
#endif

	vector<float> grid_width(align_data.img_data.size());
	vector<float> grid_height(align_data.img_data.size());
	for (size_t i = 0; i < grid_width.size(); ++i)
	{
		grid_width[i] = (float)(align_data.img_data[i].scale_img.cols - 1) / (float)num_col;
		grid_height[i] = (float)(align_data.img_data[i].scale_img.rows - 1) / (float)num_row;
	}

	// ---- Compute 'alpha value' & 'vertex index' of the features ----
	vector<double> constraint(3);

	//double w = align_data.feat_weight / (double)corr.size();

	double w = align_data.feat_weight;
	for (size_t p = 0; p < corr.size(); ++p)
	{
		int m = corr[p].m;
		int n = corr[p].n;
		double width[2];
		width[0] = align_data.img_data[m].scale_img.cols;
		width[1] = align_data.img_data[n].scale_img.cols;
		Point2f &fm = corr[p].fm;
		Point2f &fn = corr[p].fn;
		int idx[2] = {m, n};
		Point2f pos[2] = {fm, fn};
		Feature feat[2];
		double offset[2] = {0.0, 0.0};
		for (int i = 0; i < 2; ++i)
		{
			float gw = grid_width[idx[i]];
			float gh = grid_height[idx[i]];

			/*  alpha1-- left, alpha2--right
			*  alpha3-- up  , alpha4--down	 */
			double alpha1, alpha2, alpha3, alpha4;
			int c, r;
			c = (int)(pos[i].x / gw);
			if (c == num_col)
			{
				c--;
			}
			alpha2 = pos[i].x - c * gw;
			alpha2 = alpha2 / gw;
			alpha1 = 1 - alpha2;

			r = (int)(pos[i].y / gh);
			if (r == num_row)
			{
				r--;
			}
			alpha4 = pos[i].y - r * gh;
			alpha4 = alpha4 / gh;
			alpha3 = 1.0 - alpha4;

			feat[i].vertex[0] = nv_col * r + c;
			feat[i].vertex[1] = nv_col * r + c + 1;
			feat[i].vertex[2] = nv_col * (r + 1) + c;
			feat[i].vertex[3] = nv_col * (r + 1) + c + 1;

#if LOOP_GRID
			if (c == num_col - 1)
			{
				feat[i].vertex[1] = nv_col * r;
				feat[i].vertex[3] = nv_col * (r + 1);
				offset[i] = align_data.img_data[0].scale_img.cols - 1;
			}
#endif
			feat[i].alpha[0] = alpha1 * alpha3;
			feat[i].alpha[1] = alpha2 * alpha3;
			feat[i].alpha[2] = alpha1 * alpha4;
			feat[i].alpha[3] = alpha2 * alpha4;
		}
		for (int i = 0; i < 2; ++i)
		{
			vector<double> col_idx(4), val(4);
			for (size_t v = 0; v < feat[i].vertex.size(); ++v)
			{
				col_idx[v] = 2 * num_vert * idx[i] + feat[i].vertex[v];
				val[v] = feat[i].alpha[v] * w;
			}
			// X coordinate
			for (size_t c_c = 0; c_c < col_idx.size(); ++c_c)
			{
				constraint[0] = row_count;
				constraint[1] = col_idx[c_c];
				constraint[2] = val[c_c];
				matrix_val.push_back(constraint);
			}
			row_count++;
			// Y coordinate
			for (size_t c_c = 0; c_c < col_idx.size(); ++c_c)
			{
				constraint[0] = row_count;
				constraint[1] = col_idx[c_c] + num_vert;
				constraint[2] = val[c_c];
				matrix_val.push_back(constraint);
			}

			row_count++;
			float anchor_x[3];

			anchor_x[0] = anchor_points[p].x;
			anchor_x[1] = anchor_points[p].x + width[i];
			anchor_x[2] = anchor_points[p].x - width[i];
			float mind = abs(anchor_x[0] - pos[i].x);
			int closest = 0;
			for (int k = 1; k < 3; ++k)
			{
				float d = abs(anchor_x[k] - pos[i].x);
				if (d < mind)
				{
					mind = d;
					closest = k;
				}
			}

#if NON_LOOP_GRID
			if (dval < dval2)
				b.push_back(anchor_points[p].x * w);
			else
				b.push_back((anchor_points[p].x + width[i]) * w);
#endif
#if LOOP_GRID
			b.push_back(anchor_x[closest] * w - (val[1] + val[3]) * offset[i]);
#endif

			b.push_back(anchor_points[p].y * w);
		}
	}
}

void Optimisor::depthConstraint(vector<vector<double>> &matrix_val, vector<double> &b, int &row_count, vector<Vec3d> &depth_points, int num)
{
	cout << "	depth point constraint" << endl;
	// Compute alpha values and fill them to matrix
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = (num_row + 1) * num_col;

	float gw = 2 * CV_PI / (float)num_col;
	float gh = CV_PI / (float)num_row;
	cout << "num_row:" << num_row << endl;
	// ---- Compute 'alpha value' & 'vertex index' of the features ----
	vector<double> constraint(3);
	vector<int> verticecount(num_vert);
	//double w = align_data.feat_weight / (double)corr.size();
	vector<Point2f> vertpos((num_row + 1) * (num_col + 1));
	for (size_t i = 0; i <= num_row; ++i)
	{
		for (size_t j = 0; j <= num_col; j++)
		{
			vertpos[i * (num_col + 1) + j].y = i * gh;
			vertpos[i * (num_col + 1) + j].x = j * gw;
		}
	}

	double w = align_data.feat_weight;
	for (size_t p = 0; p < depth_points.size(); ++p)
	{
		Feature feat;
		double offset = {0.0};
		Vec3d ps = depth_points[p];
		double depth = ps[0];
		double phi = ps[1];   //width
		double theta = ps[2]; //height
		Point2f pos;
		pos.x = ps[1];
		pos.y = ps[2];
		//  alpha1-- left, alpha2--right
		//  alpha3-- up  , alpha4--down
		double alpha2, alpha4;
		int c, r;
		c = (int)(phi / gw);
		if (c == num_col)
		{
			c--;
		}
		alpha2 = phi - c * gw;
		alpha2 = alpha2 / gw;

		r = (int)(theta / gh);
		if (r == num_row)
		{
			r--;
		}
		alpha4 = theta - r * gh;
		alpha4 = alpha4 / gh;

		// gv[0] - gv[1] < upper
		// |     \     |
		// gv[2] - gv[3]
		// ^
		// lower
		feat.vertex[0] = num_col * r + c;
		feat.vertex[1] = num_col * r + c + 1;
		feat.vertex[2] = num_col * (r + 1) + c;
		feat.vertex[3] = num_col * (r + 1) + c + 1;

		float b1, b2, b3;
		int idx = (num_col + 1) * r + c;
		//if(alpha2+alpha4<1)//upertri
		if (alpha2 > alpha4) //upertri
			Barycentric(pos, vertpos[idx], vertpos[idx + 1], vertpos[idx + (num_col + 1) + 1], b1, b2, b3);
		else //downtri
			Barycentric(pos, vertpos[idx], vertpos[idx + (num_col + 1)], vertpos[idx + (num_col + 1) + 1], b1, b2, b3);
		if (c == num_col - 1)
		{
			feat.vertex[1] = num_col * r;
			feat.vertex[3] = num_col * (r + 1);
			offset = align_data.img_data[0].scale_img.cols - 1;
		}

		vector<double> col_idx(3), val(3);
		if (alpha2 > alpha4)
		{ //upertri
			col_idx[0] = feat.vertex[0];
			col_idx[1] = feat.vertex[1];
			col_idx[2] = feat.vertex[3];
		}
		else
		{ //downtri
			col_idx[0] = feat.vertex[0];
			col_idx[1] = feat.vertex[2];
			col_idx[2] = feat.vertex[3];
		}

		val[0] = b1;
		val[1] = b2;
		val[2] = b3;
		for (size_t c_c = 0; c_c < col_idx.size(); ++c_c)
		{
			constraint[0] = row_count;
			constraint[1] = num_vert * num + col_idx[c_c];
			constraint[2] = val[c_c];
			//dragon debug
			cout << "===================" << endl;
			cout << "r=" << constraint[0] << endl;
			cout << "c=" << constraint[1] << endl;
			cout << "v=" << constraint[2] << endl;
			//dragon
			matrix_val.push_back(constraint);
			verticecount[col_idx[c_c]] += 1;
		}
		row_count++;

		//debug
		static int dragonkaodebug = 10;
		assert(dragonkaodebug);
		dragonkaodebug --;
        
		b.push_back(ps[0]);
	}

	vector<double> col_idx_2(2), val_2(2);
	w = 1;
	/*
	for(int i=0;i<num_vert;i++)
	{
		//Ex,Ey
		//cout<<verticecount[i]<<"\t";
		//if((i+1)%num_col==0)
		//	cout<<endl;
		if(verticecount[i]==0){
		//if(1){
				bool isEdge, isVer, isHor;

			if(i >= num_col * num_row)
			{//----Down edge (Horizotal)----
				isEdge = true;	isVer = false;	isHor = true;
			}
			else if(i < num_col)
			{//---- Up edge (Horizotal)----
				isEdge = false;	isVer = false; isHor = true;
			}
			else {
				isEdge = false;	isVer = true;	isHor = false;
			}

			if(isHor) {
				val_2[0] = w/2;
				val_2[1] = -w/2;
				
				//x
				col_idx_2[0] =  i+1 ;
				col_idx_2[1] =  i-1;
				
				if(i % num_col == 0)
					col_idx_2[1] = i + num_col-1;
				
				if(i % num_col == num_col - 1)
					col_idx_2[0] = i - num_col+1;

				for(int s = 0; s < 2; ++s) {
					constraint[0] = row_count;
					constraint[1] = num_vert*num + col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);
				
			}
		

			if(isVer) {
				val_2[0] = w/2;
				val_2[1] = -w/2;
				//x
	
				col_idx_2[0] =  i+1;
				col_idx_2[1] =  i-1;

				if(i % num_col == 0)
					col_idx_2[1] = i + num_col-1;
				
				if(i % num_col == num_col - 1)
					col_idx_2[0] = i - num_col+1;

	

				for(int s = 0; s < 2; ++s) {
					constraint[0] = row_count;
					constraint[1] = num_vert*num + col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);

				//y
				col_idx_2[0] =  i+num_col ;
				col_idx_2[1] =  i-num_col;

				for(int s = 0; s < 2; ++s) {
					constraint[0] = row_count;
					constraint[1] = num_vert*num + col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);
			}

		}
	}
	*/
}
void Optimisor::depthConstraint2(vector<vector<double>> &matrix_val, vector<double> &b, int &row_count, vector<Vec3d> &depth_points)
{
	cout << "	depth point constraint" << endl;
	// Compute alpha values and fill them to matrix
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int nv_col = num_col + 1;
	int num_vert = (num_row + 1) * (num_col + 1);
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int nv_col = num_col;
	int num_vert = (num_row + 1) * num_col;
#endif
	Size equi_size = align_data.img_data[0].warp_imgs[0].size();
	vector<float> grid_width(align_data.img_data.size());
	vector<float> grid_height(align_data.img_data.size());
	for (size_t i = 0; i < grid_width.size(); ++i)
	{
		grid_width[i] = (float)(align_data.img_data[i].scale_img.cols - 1) / (float)num_col;
		grid_height[i] = (float)(align_data.img_data[i].scale_img.rows - 1) / (float)num_row;
	}

	// ---- Compute 'alpha value' & 'vertex index' of the features ----
	vector<int> verticecount(num_vert);
	vector<double> constraint(3);

	double w = align_data.feat_weight;

	float gw = (equi_size.width - 1) / (float)num_col;
	float gh = (equi_size.height - 1) / (float)num_row;
	cout << "ew:" << equi_size.width << "\teh:" << equi_size.height << endl;
	cout << "gw:" << gw << "\tgh:" << gh << endl;

	for (size_t p = 0; p < depth_points.size(); ++p)
	{
		Feature feat;
		double offset = {0.0};
		Vec3d ps = depth_points[p];
		Point2f pos;
		pos.x = ps[1];
		pos.y = ps[2];
		/*  alpha1-- left, alpha2--right
		*  alpha3-- up  , alpha4--down	 */
		double alpha1, alpha2, alpha3, alpha4;
		int c, r;
		c = (int)(pos.x / gw);
		if (c == num_col)
		{
			c--;
		}
		alpha2 = pos.x - c * gw;
		alpha2 = alpha2 / gw;
		alpha1 = 1 - alpha2;

		r = (int)(pos.y / gh);
		if (r == num_row)
		{
			r--;
		}
		alpha4 = pos.y - r * gh;
		alpha4 = alpha4 / gh;
		alpha3 = 1.0 - alpha4;

		feat.vertex[0] = nv_col * r + c;
		feat.vertex[1] = nv_col * r + c + 1;
		feat.vertex[2] = nv_col * (r + 1) + c;
		feat.vertex[3] = nv_col * (r + 1) + c + 1;

#if LOOP_GRID
		if (c == num_col - 1)
		{
			feat.vertex[1] = nv_col * r;
			feat.vertex[3] = nv_col * (r + 1);
			offset = align_data.img_data[0].scale_img.cols - 1;
		}
#endif
		feat.alpha[0] = alpha1 * alpha3;
		feat.alpha[1] = alpha2 * alpha3;
		feat.alpha[2] = alpha1 * alpha4;
		feat.alpha[3] = alpha2 * alpha4;

		vector<double> col_idx(4), val(4);
		for (size_t v = 0; v < feat.vertex.size(); ++v)
		{
			col_idx[v] = feat.vertex[v];
			verticecount[feat.vertex[v]] += 1;
			val[v] = feat.alpha[v] * w;
		}

		for (size_t c_c = 0; c_c < col_idx.size(); ++c_c)
		{
			constraint[0] = row_count;
			constraint[1] = col_idx[c_c];
			constraint[2] = val[c_c];
			matrix_val.push_back(constraint);
		}

		row_count++;

#if NON_LOOP_GRID
		if (dval < dval2)
			b.push_back(anchor_points[p].x * w);
		else
			b.push_back((anchor_points[p].x + width[i]) * w);
#endif
#if LOOP_GRID
//b.push_back(anchor_x[closest] * w - (val[1]+val[3]) * offset[i]);
#endif
		b.push_back(ps[0]);
		//p+=100;
	}
	w = 1.0;
	vector<double> col_idx_2(2), val_2(2);
	for (int i = 0; i < num_vert; i++)
	{
		//Ex,Ey
		cout << verticecount[i] << "\t";
		if ((i + 1) % num_col == 0)
			cout << endl;
		bool flag = true;
		if ((verticecount[i] == 0) && (flag))
		{
			bool isEdge, isVer, isHor;

			if (i >= num_col * num_row)
			{ //----Down edge (Horizotal)----
				isEdge = true;
				isVer = false;
				isHor = true;
			}
			else if (i < num_col)
			{ //---- Up edge (Horizotal)----
				isEdge = false;
				isVer = false;
				isHor = true;
			}
			else
			{
				isEdge = false;
				isVer = true;
				isHor = false;
			}

			if (isHor)
			{
				val_2[0] = w / 2;
				val_2[1] = -w / 2;

				//x
				col_idx_2[0] = i + 1;
				col_idx_2[1] = i - 1;

				if (i % num_col == 0)
					col_idx_2[1] = i + num_col - 1;

				if (i % num_col == num_col - 1)
					col_idx_2[0] = i - num_col + 1;

				for (int s = 0; s < 2; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);
			}

			if (isVer)
			{
				val_2[0] = w / 2;
				val_2[1] = -w / 2;
//x
#if LOOP_GRID
				col_idx_2[0] = i + 1;
				col_idx_2[1] = i - 1;

				if (i % num_col == 0)
					col_idx_2[1] = i + num_col - 1;

				if (i % num_col == num_col - 1)
					col_idx_2[0] = i - num_col + 1;

#endif

				for (int s = 0; s < 2; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);

				//y
				col_idx_2[0] = i + num_col;
				col_idx_2[1] = i - num_col;

				for (int s = 0; s < 2; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);
			}
		}
	}
}
void Optimisor::depthConstraint3(vector<vector<double>> &matrix_val, vector<double> &b, int &row_count, vector<Vec3d> &depth_points)
{
	cout << "	depth point constraint" << endl;
	// Compute alpha values and fill them to matrix
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int nv_col = num_col + 1;
	int num_vert = (num_row + 1) * (num_col + 1);
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int nv_col = num_col;
	int num_vert = (num_row + 1) * num_col;
#endif
	Size equi_size = align_data.img_data[0].warp_imgs[0].size();
	float gw = equi_size.width / (float)num_col;
	float gh = equi_size.height / (float)num_row;
	vector<Point2f> vertpos((num_row + 1) * (num_col + 1));
	cout << num_row << "," << num_col << endl;
	for (size_t i = 0; i <= num_row; ++i)
	{
		for (size_t j = 0; j <= num_col; j++)
		{
			vertpos[i * (num_col + 1) + j].x = j * gw;
			vertpos[i * (num_col + 1) + j].y = i * gh;
		}
	}

	// ---- Compute 'alpha value' & 'vertex index' of the features ----
	vector<int> verticecount(num_vert);
	vector<double> constraint(3);

	double w = align_data.feat_weight;

	cout << "ew:" << equi_size.width << "\teh:" << equi_size.height << endl;
	cout << "gw:" << gw << "\tgh:" << gh << endl;

	for (size_t p = 0; p < depth_points.size(); ++p)
	{
		Feature feat;
		double offset = {0.0};
		Vec3d ps = depth_points[p];
		Point2f pos;
		pos.x = ps[1];
		pos.y = ps[2];
		/*  alpha1-- left, alpha2--right
		*  alpha3-- up  , alpha4--down	 */
		double alpha1, alpha2, alpha3, alpha4;
		int c, r;
		c = (int)(pos.x / gw);
		if (c == num_col)
		{
			c--;
		}
		alpha2 = pos.x - c * gw;
		alpha2 = alpha2 / gw;

		r = (int)(pos.y / gh);
		if (r == num_row)
		{
			r--;
		}
		alpha4 = pos.y - r * gh;
		alpha4 = alpha4 / gh;

		feat.vertex[0] = nv_col * r + c;
		feat.vertex[1] = nv_col * r + c + 1;
		feat.vertex[2] = nv_col * (r + 1) + c;
		feat.vertex[3] = nv_col * (r + 1) + c + 1;
		Point2f test(15, 15);

		float b1, b2, b3;
		int idx = (nv_col + 1) * r + c;
		if (alpha2 + alpha4 < 1) //upertri
			Barycentric(pos, vertpos[idx], vertpos[idx + 1], vertpos[idx + nv_col + 1], b1, b2, b3);
		else //downtri
			Barycentric(pos, vertpos[idx + 1], vertpos[idx + nv_col + 1], vertpos[idx + nv_col + 2], b1, b2, b3);
		if (b1 > 1 || b1 < 0 || b2 > 1 || b2 < 0 || b3 > 1 || b3 < 0)
		{
			cout << p << endl;
			cout << alpha2 << "," << alpha4 << endl;
			cout << "0:" << vertpos[feat.vertex[0]].x << "," << vertpos[feat.vertex[0]].y << "\t1:" << vertpos[feat.vertex[1]].x << "," << vertpos[feat.vertex[1]].y << endl;
			cout << pos.x << "," << pos.y << endl;
			cout << "2:" << vertpos[feat.vertex[2]].x << "," << vertpos[feat.vertex[2]].y << "\t3:" << vertpos[feat.vertex[3]].x << "," << vertpos[feat.vertex[3]].y << endl;
			cout << b1 << "\t" << b2 << "\t" << b3 << endl;
			Barycentric(pos, vertpos[feat.vertex[0]], vertpos[feat.vertex[1]], vertpos[feat.vertex[2]], b1, b2, b3);
			cout << b1 << "\t" << b2 << "\t" << b3 << endl;
			Barycentric(pos, vertpos[feat.vertex[1]], vertpos[feat.vertex[2]], vertpos[feat.vertex[3]], b1, b2, b3);
			cout << b1 << "\t" << b2 << "\t" << b3 << endl;
		}
#if LOOP_GRID
		if (c == num_col - 1)
		{
			feat.vertex[1] = nv_col * r;
			feat.vertex[3] = nv_col * (r + 1);
			offset = align_data.img_data[0].scale_img.cols - 1;
		}
#endif
		vector<double> col_idx(3), val(3);
		if (alpha2 > alpha4)
		{ //upertri
			col_idx[0] = feat.vertex[0];
			col_idx[1] = feat.vertex[1];
			col_idx[2] = feat.vertex[2];
		}
		else
		{ //downtri
			col_idx[0] = feat.vertex[1];
			col_idx[1] = feat.vertex[2];
			col_idx[2] = feat.vertex[3];
		}
		val[0] = b1;
		val[1] = b2;
		val[2] = b3;

		for (size_t c_c = 0; c_c < col_idx.size(); ++c_c)
		{
			constraint[0] = row_count;
			constraint[1] = col_idx[c_c];
			constraint[2] = val[c_c];
			matrix_val.push_back(constraint);
			verticecount[col_idx[c_c]] += 1;
		}

		row_count++;

#if NON_LOOP_GRID
		if (dval < dval2)
			b.push_back(anchor_points[p].x * w);
		else
			b.push_back((anchor_points[p].x + width[i]) * w);
#endif
#if LOOP_GRID
//b.push_back(anchor_x[closest] * w - (val[1]+val[3]) * offset[i]);
#endif
		b.push_back(ps[0]);
		//p+=100;
	}
	w = 1.0;
	vector<double> col_idx_2(2), val_2(2);
	for (int i = 0; i < num_vert; i++)
	{
		//Ex,Ey
		cout << verticecount[i] << "\t";
		if ((i + 1) % num_col == 0)
			cout << endl;
		bool flag = true;
		if ((verticecount[i] == 0) && (flag))
		{
			bool isEdge, isVer, isHor;

			if (i >= num_col * num_row)
			{ //----Down edge (Horizotal)----
				isEdge = true;
				isVer = false;
				isHor = true;
			}
			else if (i < num_col)
			{ //---- Up edge (Horizotal)----
				isEdge = false;
				isVer = false;
				isHor = true;
			}
			else
			{
				isEdge = false;
				isVer = true;
				isHor = false;
			}

			if (isHor)
			{
				val_2[0] = w / 2;
				val_2[1] = -w / 2;

				//x
				col_idx_2[0] = i + 1;
				col_idx_2[1] = i - 1;

				if (i % num_col == 0)
					col_idx_2[1] = i + num_col - 1;

				if (i % num_col == num_col - 1)
					col_idx_2[0] = i - num_col + 1;

				for (int s = 0; s < 2; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);
			}

			if (isVer)
			{
				val_2[0] = w / 2;
				val_2[1] = -w / 2;
//x
#if LOOP_GRID
				col_idx_2[0] = i + 1;
				col_idx_2[1] = i - 1;

				if (i % num_col == 0)
					col_idx_2[1] = i + num_col - 1;

				if (i % num_col == num_col - 1)
					col_idx_2[0] = i - num_col + 1;

#endif

				for (int s = 0; s < 2; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);

				//y
				col_idx_2[0] = i + num_col;
				col_idx_2[1] = i - num_col;

				for (int s = 0; s < 2; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_2[s];
					constraint[2] = val_2[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);
			}
		}
	}
}
void Optimisor::distConstraint(vector<vector<double>> &matrix_val, vector<double> &b, int img, int &row_count)
{
	// This implementation is based on the following paper:
	// ��A shape preserving approach to image resizing��
	vector<DistortionData> distort_data;
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int num_quad = num_row * num_col;
	int num_vert = (num_row + 1) * (num_col + 1);
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_quad = num_row * num_col;
	int num_vert = (num_row + 1) * num_col;
#endif
	distort_data.resize(num_quad);
	int idx = 0;
	// ---- get idx_vertex
	// 0 �w 1
	// |   |
	// 2 �w 3
	for (int r = 0; r < num_row; ++r)
	{
		for (int c = 0; c < num_col; ++c)
		{
			DistortionData &distort = distort_data[idx];
			distort.idx_ver_r.resize(4);
			distort.idx_ver_c.resize(4);
			distort.idx_ver_r[0] = r;
			distort.idx_ver_r[1] = r;
			distort.idx_ver_r[2] = r + 1;
			distort.idx_ver_r[3] = r + 1;
			distort.idx_ver_c[0] = c;
			distort.idx_ver_c[1] = c + 1;
			distort.idx_ver_c[2] = c;
			distort.idx_ver_c[3] = c + 1;
#if LOOP_GRID
			if (c == num_col - 1)
			{
				distort.idx_ver_c[1] = 0;
				distort.idx_ver_c[3] = 0;
			}
#endif
			idx++;
		}
	}
	// ---- get Cp
	idx = 0;

	Mat tmp1, tmp2, tmp3;
	Mat I_mat = Mat::zeros(8, 8, CV_32FC1);
	Mat Ap = Mat::zeros(8, 4, CV_32FC1);

	ImageMesh &mesh = align_data.mesh_data[img].ori_mesh;
	vector<double> constraint(3), col_idx(8), val(8);
	for (int r = 0; r < num_row; ++r)
	{
		for (int c = 0; c < num_col; ++c)
		{

			for (int i = 0; i < 4; ++i)
			{
				int idx_r = distort_data[idx].idx_ver_r[i];
				int idx_c = distort_data[idx].idx_ver_c[i];
				Ap.at<float>(i, 0) = mesh[idx_r][idx_c].x;
				Ap.at<float>(i, 1) = -mesh[idx_r][idx_c].y;
				Ap.at<float>(i, 2) = 1;
				Ap.at<float>(i, 3) = 0;
				Ap.at<float>(i + 4, 0) = mesh[idx_r][idx_c].y;
				Ap.at<float>(i + 4, 1) = mesh[idx_r][idx_c].x;
				Ap.at<float>(i + 4, 2) = 0;
				Ap.at<float>(i + 4, 3) = 1;
			}
			setIdentity(I_mat);	// I
			transpose(Ap, tmp2);   // Ap_t (tmp2)
			tmp1 = tmp2 * Ap;	  // Ap_t*Ap
			invert(tmp1, tmp1, 0); // (Ap_t*Ap)^-1 (tmp1)
			tmp2 = tmp1 * tmp2;	// [(Ap_t*Ap)^-1]Ap_t
			tmp3 = Ap * tmp2;	  // Ap[(Ap_t*Ap)^-1]Ap_t
			distort_data[idx].Cp = tmp3 - I_mat;
			for (int r_c = 0; r_c < distort_data[idx].Cp.rows; ++r_c)
			{
				for (int c_c = 0; c_c < distort_data[idx].Cp.cols; ++c_c)
				{
					//val[c_c] = distort_data[idx].Cp.at<float>(r_c, c_c) * align_data.dist_weight / (double)num_quad;
					val[c_c] = distort_data[idx].Cp.at<float>(r_c, c_c) * align_data.dist_weight;
					int idx_v = c_c % 4;
// -- idx_ver = r*numVer_Col + c
#if NON_LOOP_GRID
					int idx_ver = distort_data[idx].idx_ver_r[idx_v] * (num_col + 1) + distort_data[idx].idx_ver_c[idx_v];
#endif
#if LOOP_GRID
					int idx_ver = distort_data[idx].idx_ver_r[idx_v] * num_col + distort_data[idx].idx_ver_c[idx_v];
#endif
					col_idx[c_c] = (c_c < 4) ? num_vert * 2 * img + idx_ver : num_vert * 2 * img + idx_ver + num_vert;
				}

				for (size_t c_c = 0; c_c < col_idx.size(); ++c_c)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx[c_c];
					constraint[2] = val[c_c];
					matrix_val.push_back(constraint);
				}
				row_count++;
				b.push_back(0);
			}
			idx++;

			tmp1.release();
			tmp2.release();
			tmp3.release();
		}
	}
}

void Optimisor::smoothConstraint(vector<vector<double>> &matrix_val, vector<double> &b, int img, int &row_count)
{

	vector<double> constraint(3), col_idx_3(3), val_3(3);
	vector<double> col_idx_4(4), val_4(4);
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int num_vert = (num_row + 1) * (num_col + 1);
	// count #contraints of shape term
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = (num_row + 1) * num_col;
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
	//double w = align_data.shape_weight / n_shape;
	double w = align_data.shape_weight;
	for (int i = 0; i < num_vert; ++i)
	{
		bool isEdge, isVer, isHor;
#if NON_LOOP_GRID
		if ((i == 0) || (i == num_vert - 1) || (i == (num_col + 1) * num_row) || (i == num_col))
			continue; //---- if i is corner, continue.
		if (i > (num_col + 1) * num_row)
		{ //----Down edge (Horizotal)----
			isEdge = true;
			isVer = false;
			isHor = true;
		}
		else if (i < num_col)
		{ //---- Up edge (Horizotal)----
			isEdge = false;
			isVer = false;
			isHor = (i == 0) ? false : true;
		}
		else if (i % (num_col + 1) == 0)
		{ //---- Left edge (Vertical)----
			isEdge = true;
			isVer = true;
			isHor = false;
		}
		else if (i % (num_col + 1) == num_col)
		{ //---- Right edge (Vertical)----
			isEdge = true;
			isVer = true;
			isHor = false;
		}
		else
		{
			isEdge = false;
			isVer = true;
			isHor = true;
		}
#endif
#if LOOP_GRID
		if (i >= num_col * num_row)
		{ //----Down edge (Horizotal)----
			isEdge = true;
			isVer = false;
			isHor = true;
		}
		else if (i < num_col)
		{ //---- Up edge (Horizotal)----
			isEdge = false;
			isVer = false;
			isHor = true;
		}
		else
		{
			isEdge = false;
			isVer = true;
			isHor = true;
		}
#endif

		if (isHor)
		{
			val_3[0] = w;
			val_3[1] = -2 * w;
			val_3[2] = w;
			double offset = 0.0;
			for (int n = 0; n < 2; ++n)
			{
				col_idx_3[0] = num_vert * 2 * img + num_vert * n + i + 1;
				col_idx_3[1] = num_vert * 2 * img + num_vert * n + i;
				col_idx_3[2] = num_vert * 2 * img + num_vert * n + i - 1;

#if LOOP_GRID
				if (i % num_col == 0)
				{
					offset = -align_data.img_data[0].scale_img.cols + 1;
					col_idx_3[2] = num_vert * 2 * img + num_vert * n + i + num_col - 1;
				}

				if (i % num_col == num_col - 1)
				{
					offset = align_data.img_data[0].scale_img.cols - 1;
					col_idx_3[0] = num_vert * 2 * img + num_vert * n + i - num_col + 1;
				}
#endif

				for (int s = 0; s < 3; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_3[s];
					constraint[2] = val_3[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
			}
#if NON_LOOP_GRID
			b.push_back(0);
			b.push_back(0);
#endif
#if LOOP_GRID
			b.push_back(-w * offset);
			b.push_back(0);
#endif
		}

		if (isVer)
		{
			val_3[0] = w;
			val_3[1] = -2 * w;
			val_3[2] = w;

			for (int n = 0; n < 2; ++n)
			{
#if NON_LOOP_GRID
				col_idx_3[0] = num_vert * 2 * img + num_vert * n + i - (num_col + 1);
				col_idx_3[1] = num_vert * 2 * img + num_vert * n + i;
				col_idx_3[2] = num_vert * 2 * img + num_vert * n + i + (num_col + 1);
#endif
#if LOOP_GRID
				col_idx_3[0] = num_vert * 2 * img + num_vert * n + i - num_col;
				col_idx_3[1] = num_vert * 2 * img + num_vert * n + i;
				col_idx_3[2] = num_vert * 2 * img + num_vert * n + i + num_col;
#endif

				for (int s = 0; s < 3; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_3[s];
					constraint[2] = val_3[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
			}
			b.push_back(0);
			b.push_back(0);
		}

		if (!isEdge)
		{
			val_4[0] = w;
			val_4[1] = -w;
			val_4[2] = -w;
			val_4[3] = w;
			for (int n = 0; n < 2; ++n)
			{
#if NON_LOOP_GRID
				col_idx_4[0] = num_vert * 2 * img + num_vert * n + i;
				col_idx_4[1] = num_vert * 2 * img + num_vert * n + i + 1;
				col_idx_4[2] = num_vert * 2 * img + num_vert * n + i + (num_col + 1);
				col_idx_4[3] = num_vert * 2 * img + num_vert * n + i + 1 + (num_col + 1);
#endif

#if LOOP_GRID
				col_idx_4[0] = num_vert * 2 * img + num_vert * n + i;
				col_idx_4[1] = num_vert * 2 * img + num_vert * n + i + 1;
				col_idx_4[2] = num_vert * 2 * img + num_vert * n + i + num_col;
				col_idx_4[3] = num_vert * 2 * img + num_vert * n + i + 1 + num_col;
				if (i % num_col == num_col - 1)
				{
					col_idx_4[1] = num_vert * 2 * img + num_vert * n + i - num_col + 1;
					col_idx_4[3] = num_vert * 2 * img + num_vert * n + i + 1;
				}
#endif

				for (int s = 0; s < 4; ++s)
				{
					constraint[0] = row_count;
					constraint[1] = col_idx_4[s];
					constraint[2] = val_4[s];
					matrix_val.push_back(constraint);
				}
				row_count++;
			}
			b.push_back(0);
			b.push_back(0);
		}
	}
}

void Optimisor::smoothConstraint2(vector<vector<double>> &matrix_val, vector<double> &b, int &row_count, int num)
{
	vector<double> constraint(3), col_idx_3(3), val_3(3);
	vector<double> col_idx_4(4), val_4(4);
	vector<double> col_idx_5(5), val_5(5);
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int num_vert = (num_row + 1) * (num_col + 1);
	// count #contraints of shape term
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = (num_row + 1) * num_col;
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
	//double w = align_data.shape_weight / n_shape;
	Size img_size = align_data.img_data[0].warp_imgs[0].size();
	double test = min(img_size.height / num_row, img_size.width / num_col);
	test = 3000;
	double w = 1;
	for (int i = 0; i < num_vert; ++i)
	{
		bool isEdge, isVer, isHor;
#if NON_LOOP_GRID
		if ((i == 0) || (i == num_vert - 1) || (i == (num_col + 1) * num_row) || (i == num_col))
			continue; //---- if i is corner, continue.
		if (i > (num_col + 1) * num_row)
		{ //----Down edge (Horizotal)----
			isEdge = true;
			isVer = false;
			isHor = true;
		}
		else if (i < num_col)
		{ //---- Up edge (Horizotal)----
			isEdge = false;
			isVer = false;
			isHor = (i == 0) ? false : true;
		}
		else if (i % (num_col + 1) == 0)
		{ //---- Left edge (Vertical)----
			isEdge = true;
			isVer = true;
			isHor = false;
		}
		else if (i % (num_col + 1) == num_col)
		{ //---- Right edge (Vertical)----
			isEdge = true;
			isVer = true;
			isHor = false;
		}
		else
		{
			isEdge = false;
			isVer = true;
			isHor = true;
		}
#endif
#if LOOP_GRID
		if (i >= num_col * num_row)
		{ //----Down edge (Horizotal)----
			isEdge = true;
			isVer = false;
			isHor = true;
		}
		else if (i < num_col)
		{ //---- Up edge (Horizotal)----
			isEdge = false;
			isVer = false;
			isHor = true;
		}
		else
		{
			isEdge = false;
			isVer = true;
			isHor = false;
		}
#endif
		//isVer = false;
		if (isHor)
		{
			val_4[0] = w;
			val_4[1] = -w / 3;
			val_4[2] = -w / 3;
			val_4[3] = -w / 3;
			double offset = 0.0;
			col_idx_4[0] = i;
			col_idx_4[1] = i - 1;
			col_idx_4[2] = i + 1;
			if (isEdge)
				col_idx_4[3] = i - num_col;
			else
				col_idx_4[3] = i + num_col;
#if LOOP_GRID
			if (i % num_col == 0)
			{
				offset = -align_data.img_data[0].scale_img.cols + 1;
				col_idx_4[1] = i + num_col - 1;
			}

			if (i % num_col == num_col - 1)
			{
				offset = align_data.img_data[0].scale_img.cols - 1;
				col_idx_4[2] = i - num_col + 1;
			}
#endif

			for (int s = 0; s < 4; ++s)
			{
				constraint[0] = row_count;
				constraint[1] = num_vert * num + col_idx_4[s];
				constraint[2] = val_4[s];
				matrix_val.push_back(constraint);
			}
			row_count++;

#if NON_LOOP_GRID
			//b.push_back(0);
			b.push_back(0);
#endif
#if LOOP_GRID
			//b.push_back(-w * offset);
			b.push_back(0);
#endif
		}

		if (isVer)
		{
			val_5[0] = -w / 4;
			val_5[1] = -w / 4;
			val_5[2] = w;
			val_5[3] = -w / 4;
			val_5[4] = -w / 4;

#if NON_LOOP_GRID
			col_idx_3[0] = num_vert * n + i - (num_col + 1);
			col_idx_3[1] = num_vert * n + i;
			col_idx_3[2] = num_vert * n + i + (num_col + 1);
#endif
#if LOOP_GRID
			col_idx_5[0] = i - num_col;
			col_idx_5[1] = i - 1;
			col_idx_5[2] = i;
			col_idx_5[3] = i + 1;
			col_idx_5[4] = i + num_col;

			if (i % num_col == 0)
				col_idx_5[1] = i + num_col - 1;

			if (i % num_col == num_col - 1)
				col_idx_5[3] = i - num_col + 1;

#endif

			for (int s = 0; s < 5; ++s)
			{
				constraint[0] = row_count;
				constraint[1] = num_vert * num + col_idx_5[s];
				constraint[2] = val_5[s];
				matrix_val.push_back(constraint);
			}
			row_count++;

			b.push_back(0);
		}
	}
}
void Optimisor::smoothConstraint3(vector<vector<double>> &matrix_val, vector<double> &b, int &row_count)
{
	vector<double> constraint(3), col_idx_3(3), val_3(3);
	vector<double> col_idx_4(4), val_4(4);
	vector<double> col_idx_5(5), val_5(5);
	vector<double> col_idx_6(6), val_6(6);
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int num_vert = (num_row + 1) * (num_col + 1);
	// count #contraints of shape term
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = (num_row + 1) * num_col;
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
	//double w = align_data.shape_weight / n_shape;
	Size img_size = align_data.img_data[0].warp_imgs[0].size();
	double test = min(img_size.height / num_row, img_size.width / num_col);
	test = 3000;
	double w = 1;
	for (int i = 0; i < num_vert; ++i)
	{
		bool isEdge, isVer, isHor;
#if NON_LOOP_GRID
		if ((i == 0) || (i == num_vert - 1) || (i == (num_col + 1) * num_row) || (i == num_col))
			continue; //---- if i is corner, continue.
		if (i > (num_col + 1) * num_row)
		{ //----Down edge (Horizotal)----
			isEdge = true;
			isVer = false;
			isHor = true;
		}
		else if (i < num_col)
		{ //---- Up edge (Horizotal)----
			isEdge = false;
			isVer = false;
			isHor = (i == 0) ? false : true;
		}
		else if (i % (num_col + 1) == 0)
		{ //---- Left edge (Vertical)----
			isEdge = true;
			isVer = true;
			isHor = false;
		}
		else if (i % (num_col + 1) == num_col)
		{ //---- Right edge (Vertical)----
			isEdge = true;
			isVer = true;
			isHor = false;
		}
		else
		{
			isEdge = false;
			isVer = true;
			isHor = true;
		}
#endif
#if LOOP_GRID
		if (i >= num_col * num_row)
		{ //----Down edge (Horizotal)----
			isEdge = true;
			isVer = false;
			isHor = true;
		}
		else if (i < num_col)
		{ //---- Up edge (Horizotal)----
			isEdge = false;
			isVer = false;
			isHor = true;
		}
		else
		{
			isEdge = false;
			isVer = true;
			isHor = false;
		}
#endif
		//isVer = false;
		if (isHor)
		{
			val_5[0] = w;
			val_5[1] = -w / 4;
			val_5[2] = -w / 4;
			val_5[3] = -w / 4;
			val_5[4] = -w / 4;
			double offset = 0.0;
			col_idx_4[0] = i;
			col_idx_4[1] = i - 1;
			col_idx_4[2] = i + 1;
			if (isEdge)
			{
				col_idx_4[3] = i - num_col;
				col_idx_4[4] = i - num_col;
			}
			else
			{
				col_idx_4[3] = i + num_col;
			}
#if LOOP_GRID
			if (i % num_col == 0)
			{
				offset = -align_data.img_data[0].scale_img.cols + 1;
				col_idx_4[1] = i + num_col - 1;
			}

			if (i % num_col == num_col - 1)
			{
				offset = align_data.img_data[0].scale_img.cols - 1;
				col_idx_4[2] = i - num_col + 1;
			}
#endif

			for (int s = 0; s < 4; ++s)
			{
				constraint[0] = row_count;
				constraint[1] = col_idx_4[s];
				constraint[2] = val_4[s];
				matrix_val.push_back(constraint);
			}
			row_count++;

#if NON_LOOP_GRID
			//b.push_back(0);
			b.push_back(0);
#endif
#if LOOP_GRID
			//b.push_back(-w * offset);
			b.push_back(0);
#endif
		}

		if (isVer)
		{
			val_5[0] = -w / 4;
			val_5[1] = -w / 4;
			val_5[2] = w;
			val_5[3] = -w / 4;
			val_5[4] = -w / 4;

#if NON_LOOP_GRID
			col_idx_3[0] = num_vert * n + i - (num_col + 1);
			col_idx_3[1] = num_vert * n + i;
			col_idx_3[2] = num_vert * n + i + (num_col + 1);
#endif
#if LOOP_GRID
			col_idx_5[0] = i - num_col;
			col_idx_5[1] = i - 1;
			col_idx_5[2] = i;
			col_idx_5[3] = i + 1;
			col_idx_5[4] = i + num_col;

			if (i % num_col == 0)
				col_idx_5[1] = i + num_col - 1;

			if (i % num_col == num_col - 1)
				col_idx_5[3] = i - num_col + 1;

#endif

			for (int s = 0; s < 5; ++s)
			{
				constraint[0] = row_count;
				constraint[1] = col_idx_5[s];
				constraint[2] = val_5[s];
				matrix_val.push_back(constraint);
			}
			row_count++;

			b.push_back(0);
		}
	}
}

void Optimisor::temperalSmoothConstraint(vector<vector<double>> &matrix_val, vector<double> &b, int &row_count, int max_vert)
{
	vector<double> constraint(3), col_idx_3(3), val_3(3);
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int num_vert = (num_row + 1) * (num_col + 1);
	// count #contraints of shape term
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = (num_row + 1) * num_col;
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
	//double w = align_data.shape_weight / n_shape;
	Size img_size = align_data.img_data[0].warp_imgs[0].size();
	double test = min(img_size.height / num_row, img_size.width / num_col);
	test = 3000;
	double w = 1;
	int tw = 2;
	int num_frame = max_vert / num_vert;
	vector<double> col_idx_t(2 * tw + 1), val_t(2 * tw + 1);
	for (int i = 0; i < max_vert; ++i)
	{

		int count = -1;
		for (int t = -tw; t <= tw; t++)
		{
			col_idx_t[t + tw] = num_vert * t + i;
			if ((col_idx_t[t + tw] >= 0) && (col_idx_t[t + tw] < max_vert))
				count++;
		}
		for (int t = 0; t < 2 * tw + 1; t++)
		{
			if (col_idx_t[t] == i)
				val_t[t] = 1;
			else if ((col_idx_t[t] >= 0) && (col_idx_t[t] < max_vert))
				val_t[t] = -(1 / count);
		}

		for (int s = 0; s < 2 * tw + 1; ++s)
		{
			if ((col_idx_t[s] >= 0) && (col_idx_t[s] < max_vert))
			{
				constraint[0] = row_count;
				constraint[1] = col_idx_t[s];
				constraint[2] = val_t[s];
				matrix_val.push_back(constraint);
			}
		}
		row_count++;
		b.push_back(0);
	}
}
void Optimisor::originConstraint(vector<vector<double>> &matrix_val, vector<double> &b, int img, int &row_count)
{
#define RECTANGLE 0
#define LOOP 1
	// fix four corners

	int num_vert_row = align_data.mesh_data[0].ori_mesh.size();
	int num_vert_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = num_vert_row * num_vert_col;

	vector<int> col_idx;
	vector<Point2f> boundary;
	Point2f origin = align_data.img_data[img].pos;
	int w = align_data.img_data[img].scale_img.cols;
	int h = align_data.img_data[img].scale_img.rows;
#if NON_LOOP_GRID
	float gw = (float)(align_data.img_data[img].scale_img.cols - 1) / (float)(num_vert_col - 1);
#endif
#if LOOP_GRID
	float gw = (float)(align_data.img_data[img].scale_img.cols - 1) / (float)(num_vert_col);
#endif
	float gh = (float)(align_data.img_data[img].scale_img.rows - 1) / (float)(num_vert_row - 1);

	// north pole fixation
	for (int i = 0; i < num_vert_col; ++i)
	{
		col_idx.push_back(2 * num_vert * img + i);									   // 1st row
		col_idx.push_back(2 * num_vert * img + (num_vert_row - 1) * num_vert_col + i); // last row
		float cpos = i * gw;
		cpos = (cpos > w - 1) ? w - 1 : cpos;
		boundary.push_back(origin + Point2f(cpos, 0));
		boundary.push_back(origin + Point2f(cpos, h - 1));
	}
#if RECTANGLE
	// longitude 0 and 360 fixation
	for (int i = 1; i < num_vert_row - 1; ++i)
	{
		col_idx.push_back(2 * num_vert * img + i * num_vert_col);
		col_idx.push_back(2 * num_vert * img + i * num_vert_col + num_vert_col - 1);
		float rpos = i * gh;
		rpos = (rpos > h - 1) ? h - 1 : rpos;
		boundary.push_back(origin + Point2f(0, rpos));
		boundary.push_back(origin + Point2f(w - 1, rpos));
	}
#endif

	double val = align_data.origin_weight / (double)align_data.img_data.size();
	for (int i = 0; i < boundary.size(); ++i)
	{
		vector<double> constraint(3);
#if RECTANGLE
		constraint[0] = row_count++;
		constraint[1] = col_idx[i]; // X coordinate
		constraint[2] = val;
		matrix_val.push_back(constraint);
		b.push_back(boundary[i].x * val);
#endif
		constraint[0] = row_count++;
		constraint[1] = col_idx[i] + num_vert; // Y coordinate
		constraint[2] = val;
		matrix_val.push_back(constraint);
		b.push_back(boundary[i].y * val);
	}
#if NON_LOOP_GRID
#if LOOP
	// longitude continuity of 0 and 360
	vector<Point2f> longitude_0, longitude_360;
	vector<int> idx_0, idx_360;
	for (int i = 0; i < num_vert_row - 1; ++i)
	{
		idx_0.push_back(2 * num_vert * img + i * num_vert_col);
		idx_360.push_back(2 * num_vert * img + i * num_vert_col + num_vert_col - 1);
	}
	for (int i = 0; i < idx_0.size(); ++i)
	{
		vector<double> constraint(3);

		constraint[0] = row_count;
		constraint[1] = idx_0[i]; // X coordinate
		constraint[2] = -val;
		matrix_val.push_back(constraint);
		constraint[0] = row_count;
		constraint[1] = idx_360[i];
		constraint[2] = val;
		matrix_val.push_back(constraint);
		b.push_back(val * (w - 1));
		row_count++;

		constraint[0] = row_count;
		constraint[1] = idx_0[i] + num_vert; // Y coordinate
		constraint[2] = -val;
		matrix_val.push_back(constraint);
		constraint[0] = row_count;
		constraint[1] = idx_360[i] + num_vert;
		constraint[2] = val;
		matrix_val.push_back(constraint);
		b.push_back(0);
		row_count++;
	}
#endif
#endif
}

void Optimisor::lengthConstraint(vector<vector<double>> &matrix_val, vector<double> &b, int img, int &row_count)
{
	int isEdge = 0; // is the most right or the most bottom edge
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int num_vert = (num_row + 1) * (num_col + 1);
	double n_length = (num_col * (num_row + 1) + (num_col + 1) * num_row) * align_data.img_data.size();
	int nv_row = num_row + 1;
	int nv_col = num_col + 1;
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = (num_row + 1) * num_col;
	double n_length = (num_col * (num_row + 1) + (num_col + 1) * num_row) * align_data.img_data.size();
	int nv_row = num_row + 1;
	int nv_col = num_col;
#endif
	//double w = align_data.length_weight / n_length;
	double w = align_data.length_weight * 2;
	// compute grid size
	float grid_width, grid_height;
	grid_width = (float)(align_data.img_data[img].scale_img.cols - 1) / (float)num_col;
	grid_height = (float)(align_data.img_data[img].scale_img.rows - 1) / (float)num_row;

	for (int r = 0; r < nv_row; ++r)
	{
		for (int c = 0; c < nv_col; ++c)
		{
			double offset = 0.0;
			/*			v1 ---- v2
			*			|		 |
			*  v(x, y)  |		 |
			*			v3 ---- v4
			* tmpw: v2(x)-v1(x) = gridW, v4(x)-v3(x) = gridW 
			* tmph: v3(y)-v1(y) = gridH, v4(y)-v2(y) = gridH
			*/
			double tmpw_b, tmph_b;
			vector<double> constraint(3), col_idx(2);
			vector<double> length_value(2);

			length_value[0] = -w;
			length_value[1] = w;

			if (c != nv_col - 1)
			{
				col_idx[0] = 2 * img * num_vert + r * nv_col + c;
				col_idx[1] = 2 * img * num_vert + r * nv_col + c + 1;
				isEdge = -1;
			}
			else
			{
#if NON_LOOP_GRID
				col_idx[0] = 2 * img * num_vert + r * nv_col + c;
				col_idx[1] = 2 * img * num_vert + r * nv_col + c - 1;
				isEdge = 1;
#endif
#if LOOP_GRID
				col_idx[0] = 2 * img * num_vert + r * nv_col + c;
				col_idx[1] = 2 * img * num_vert + r * nv_col;
				isEdge = -1;
				offset = align_data.img_data[img].scale_img.cols - 1;
#endif
			}

			for (int s = 0; s < 2; ++s)
			{
				constraint[0] = row_count;
				constraint[1] = col_idx[s];
				constraint[2] = isEdge * (-1) * length_value[s];
				matrix_val.push_back(constraint);
			}
			row_count++;

			if (r != num_row)
			{
				col_idx[0] = 2 * img * num_vert + num_vert + r * nv_col + c;
				col_idx[1] = 2 * img * num_vert + num_vert + (r + 1) * nv_col + c;
				isEdge = -1;
			}
			else
			{
				col_idx[0] = 2 * img * num_vert + num_vert + r * nv_col + c;
				col_idx[1] = 2 * img * num_vert + num_vert + (r - 1) * nv_col + c;
				isEdge = 1;
			}

			for (int s = 0; s < 2; ++s)
			{
				constraint[0] = row_count;
				constraint[1] = col_idx[s];
				constraint[2] = isEdge * (-1) * length_value[s];
				matrix_val.push_back(constraint);
			}
			row_count++;
#if NON_LOOP_GRID
			tmpw_b = grid_size.width * w;
#endif
#if LOOP_GRID
			tmpw_b = (grid_width - offset) * w;
#endif
			tmph_b = grid_height * w;
			b.push_back(tmpw_b); // tmpw
			b.push_back(tmph_b); // tmph
		}
	}
}

void Optimisor::transformSphere2equi(Eigen::VectorXd &equi, Eigen::VectorXd all_depth)
{
	Size equi_size = align_data.img_data[0].warp_imgs[0].size();
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int num_vert = (num_row + 1) * (num_col + 1);
	// count #contraints of shape term
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = (num_row + 1) * num_col;
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdvequi2Sphere
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
	vector<Mat> camera_mat(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		camera_mat[i] = Mat::zeros(4, 4, CV_64FC1);
		buildCameraMatrix(cameras[i], camera_mat[i]);
	}
	vector<Mat> Rc(cameras.size());
	vector<Mat> Tc(cameras.size());
	vector<Mat> NTc(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		Rc[i] = Mat::zeros(3, 3, CV_64FC1);
		Tc[i] = Mat::zeros(3, 1, CV_64FC1);
		NTc[i] = Mat::zeros(3, 1, CV_64FC1);
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				Rc[i].at<double>(r, c) = camera_mat[i].at<double>(r, c);
		for (int r = 0; r < 3; ++r)
			Tc[i].at<double>(r, 1) = camera_mat[i].at<double>(r, 3);
		// compute origin offset of each camera
		NTc[i] = Tc[i];
		Tc[i] = Rc[i] * Tc[i];
		//Tc[i] = Tc[i].reshape(3, 1);
	}

	/*float gw = equi_size.width/(float)num_row;
	float gh = equi_size.height/(float)num_col;

	Mat image = Mat::zeros(equi_size.height, equi_size.width, CV_8UC3);
	
	for(int i=0; i<equi_size.width; i++){
		for(int j=0; j<equi_size.height; j++){
			int r = int(j/gh);
			int c = int(i/gw);
			
			Vec3b intensity = 0;
			uchar blue = intensity.val[0];
			uchar green = intensity.val[1];
			uchar red = intensity.val[2];

			//Do stuff with blue, green, and red

			image.at<Vec3b>(j,i) = Vec3b(blue,green,red);
		}
	}*/
	float gw = 2 * CV_PI / (float)num_row;
	float gh = CV_PI / (float)num_col;

	for (int i = 0; i < num_vert; i++)
	{
		int r = i / num_col;
		int c = i % num_col;
		Vec3d ps;

		double depth = all_depth[i], theta = r * gh, phi = c * gw;

		ps[0] = depth * sin(phi) * sin(theta);
		ps[1] = -depth * cos(theta);
		ps[2] = depth * cos(phi) * sin(theta);

		double x = double(num_col) / 4;
		double mid[4] = {0, x, 2 * x, 3 * x};
		double base = 500;

		depth = abs(10000 - depth);
		for (int j = 0; j < cameras.size(); j++)
		{
			Mat temp = Mat::zeros(3, 1, CV_64FC1);
			temp.at<double>(0, 0) = ps[0];
			temp.at<double>(1, 0) = ps[1];
			temp.at<double>(2, 0) = ps[2];

			//temp=(Rc[j]*temp + Tc[j]);
			//temp=temp + 1000*Tc[j];
			Vec3d tempps;
			tempps = temp.col(0);

			tempps /= cv::norm(tempps);
			Point2f pe;
			sphere2Equi(tempps, pe, equi_size);
			if (j == 0)
			{
				if ((i % num_col) < mid[2])
					pe.x -= int(depth / base);
				else if ((i % num_col) > mid[2])
					pe.x += int(depth / base);
			}
			else
			{
				if ((i % num_col) < mid[j])
					pe.x += int(depth / base);
				else if ((i % num_col) > mid[j])
					pe.x -= int(depth / base);
			}

			/*((i%num_col)==0)
				pe.x = 0;*/
			equi[i + j * num_vert * 2] = pe.x;
			equi[i + num_vert + j * num_vert * 2] = pe.y;
		}
	}
}

void Optimisor::StichBySphere(Eigen::VectorXd sphere)
{
	Size equi_size = align_data.img_data[0].warp_imgs[0].size();
	int num_row = align_data.mesh_data[0].ori_mesh.size() - 1;
#if NON_LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size() - 1;
	int num_vert = (num_row + 1) * (num_col + 1);
	// count #contraints of shape term
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdv
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
#if LOOP_GRID
	int num_col = align_data.mesh_data[0].ori_mesh[0].size();
	int num_vert = (num_row + 1) * num_col;
	int h = (num_col - 1) * (num_row + 1);							//df/dudu
	int v = (num_col + 1) * (num_row - 1);							//df/dvdvequi2Sphere
	int ne = num_row + num_col + (num_col - 1) * (num_row - 1) - 1; //df/dudv
	double n_shape = (h + v + ne) * align_data.img_data.size();
#endif
	vector<Mat> camera_mat(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		camera_mat[i] = Mat::zeros(4, 4, CV_64FC1);
		buildCameraMatrix(cameras[i], camera_mat[i]);
	}
	vector<Mat> Rc(cameras.size());
	vector<Mat> Tc(cameras.size());
	vector<Mat> NTc(cameras.size());
	for (int i = 0; i < cameras.size(); ++i)
	{
		Rc[i] = Mat::zeros(3, 3, CV_64FC1);
		Tc[i] = Mat::zeros(3, 1, CV_64FC1);
		NTc[i] = Mat::zeros(3, 1, CV_64FC1);
		for (int r = 0; r < 3; ++r)
			for (int c = 0; c < 3; ++c)
				Rc[i].at<double>(r, c) = camera_mat[i].at<double>(r, c);
		for (int r = 0; r < 3; ++r)
			Tc[i].at<double>(r, 1) = camera_mat[i].at<double>(r, 3);
		// compute origin offset of each camera
		NTc[i] = Tc[i];
		Tc[i] = Rc[i] * Tc[i];
		//Tc[i] = Tc[i].reshape(3, 1);
		cout << Vec3d(Tc[i])[0] << " " << Vec3d(Tc[i])[1] << " " << Vec3d(Tc[i])[2] << endl;
	}

	//caculate all vertex 3d pos on sphere
	vector<Vec3d> V;

	float gw = 2 * CV_PI / (float)num_row;
	float gh = CV_PI / (float)num_col;
	for (int i = 0; i < num_vert; i++)
	{
		int r = i / num_col;
		int c = i % num_col;
		Vec3d X;

		double depth = sphere[i], theta = r * gh, phi = c * gw;

		X[0] = depth * sin(phi) * sin(theta);
		X[1] = -depth * cos(theta);
		X[2] = depth * cos(phi) * sin(theta);

		V.push_back(X);
	}

	gw = equi_size.width / (float)num_row;
	gh = equi_size.height / (float)num_col;

	Mat image = Mat::zeros(equi_size.height, equi_size.width, CV_8UC3);
	Mat imagetemp[4];
	imagetemp[0] = Mat::zeros(equi_size.height, equi_size.width, CV_8UC3);
	imagetemp[1] = Mat::zeros(equi_size.height, equi_size.width, CV_8UC3);
	imagetemp[2] = Mat::zeros(equi_size.height, equi_size.width, CV_8UC3);
	imagetemp[3] = Mat::zeros(equi_size.height, equi_size.width, CV_8UC3);

	for (int i = 10; i < equi_size.width; i++)
	{
		for (int j = 10; j < equi_size.height; j++)
		{
			int r = int(j / gh);
			int c = int(i / gw);

			Vec3d D;
			Point2f pe(i, j);
			equi2Sphere(pe, D, equi_size);

			Vec3d v0 = V[r * num_col + c];
			Vec3d v1 = V[r * num_col + c + 1];
			Vec3d v2 = V[(r + 1) * num_col + c];

			double t = FastIntersection(D, Vec3d(0, 0, 0), v0, v1, v2);
			if (t < 0)
			{
				v0 = V[(r + 1) * num_col + c + 1];
				t = FastIntersection(D, Vec3d(0, 0, 0), v0, v1, v2);
			}

			if (t < 0)
			{
				r += 1;
				if (r < 10)
				{
					v0 = V[r * num_col + c];
					v1 = V[r * num_col + c + 1];
					v2 = V[(r + 1) * num_col + c];
					t = FastIntersection(D, Vec3d(0, 0, 0), v0, v1, v2);
					if (t < 0)
					{
						v0 = V[(r + 1) * num_col + c + 1];
						t = FastIntersection(D, Vec3d(0, 0, 0), v0, v1, v2);
					}
				}

				r -= 2;
				if (r > -1)
				{
					v0 = V[r * num_col + c];
					v1 = V[r * num_col + c + 1];
					v2 = V[(r + 1) * num_col + c];
					t = FastIntersection(D, Vec3d(0, 0, 0), v0, v1, v2);
					if (t < 0)
					{
						v0 = V[(r + 1) * num_col + c + 1];
						t = FastIntersection(D, Vec3d(0, 0, 0), v0, v1, v2);
					}
				}
			}

			Vec3d ps = t * D;

			Vec3b intensity(0, 0, 0);

			//cout<< "t: " << t<<endl;
			//cout<< "r: " << r<< " c: " <<c<<endl;
			//cout<< ps[0]<<" "<<ps[1]<<" "<<ps[2]<<endl;
			for (int k = 0; k < cameras.size(); k++)
			{
				Vec3d temp = ps - Vec3d(Tc[k]);
				//Vec3d temp = ps;
				temp /= norm(temp);
				sphere2Equi(temp, pe, equi_size);
				//cout<< temp[0]<<" "<<temp[1]<<" "<<temp[2]<<endl;
				//cout<< "x: " << int(abs(pe.x))<< " y: " <<int(abs(pe.y))<<endl;
				Vec3b tempI = align_data.img_data[k].img.at<Vec3b>(int(abs(pe.y)), int(abs(pe.x)));
				intensity += 0.5 * tempI;
				//cout<< "k: "<<k<<endl;
				//cout<< "x: " << int(pe.x)<< " y: " <<int(pe.y)<<endl;
				uchar bluet = tempI.val[0];
				uchar greent = tempI.val[1];
				uchar redt = tempI.val[2];
				imagetemp[k].at<Vec3b>(j, i) = Vec3b(bluet, greent, redt);
			}
			uchar blue = intensity.val[0];
			uchar green = intensity.val[1];
			uchar red = intensity.val[2];

			//Do stuff with blue, green, and red

			image.at<Vec3b>(j, i) = Vec3b(blue, green, red);
		}
	}

	cv::imwrite("temp.jpg", image);
	cv::imwrite("temp1.jpg", imagetemp[0]);
	cv::imwrite("temp2.jpg", imagetemp[1]);
	cv::imwrite("temp3.jpg", imagetemp[2]);
	cv::imwrite("temp4.jpg", imagetemp[3]);
}

double Optimisor::FastIntersection(cv::Vec3d D, cv::Vec3d O, cv::Vec3d V0, cv::Vec3d V1, cv::Vec3d V2)
{
	Vec3d E1 = V1 - V0;
	Vec3d E2 = V2 - V0;
	Vec3d T = O - V0;
	Vec3d Q = T.cross(E1);
	Vec3d P = D.cross(E2);
	double t, u, v;
	double dominater = P.dot(E1);
	t = Q.dot(E2) / dominater;
	u = P.dot(T) / dominater;
	v = Q.dot(D) / dominater;
	/*cout<< "v0: " << V0[0]<< " " << V0[1]<< " " << V0[2]<<endl;
	cout<< "v1: " << V1[0]<< " " << V1[1]<< " " << V1[2]<<endl;
	cout<< "v2: " << V2[0]<< " " << V2[1]<< " " << V2[2]<<endl;
	cout<< "D: " << D[0]<< " " << D[1]<< " " << D[2]<<endl;
	cout<< "O: " << O[0]<< " " << O[1]<< " " << O[2]<<endl;*/
	//cout<< "t: " << t<< "u: " << u<< "v: " << v<<endl;

	if (u >= 0 && v >= 0 && (u + v) <= 1)
		return t;
	else
		return -1;
}
void Optimisor::drawDispMaps(vector<Mat> &disp_maps, string &filename)
{

	// draw disparity maps
	for (size_t i = 0; i < disp_maps.size(); ++i)
	{
		Mat disp_images = Mat::zeros(disp_maps[i].size(), CV_8UC1);
		float max_disp = 0, min_disp = disp_maps[i].cols;
		for (int r = 0; r < disp_maps[i].rows; ++r)
		{
			for (int c = 0; c < disp_maps[i].cols; ++c)
			{
				//if(equi_masks[2*i].at<uchar>(r, c) < 128)
				//continue;
				float dval = disp_maps[i].at<float>(r, c);
				max_disp = (max_disp < dval) ? dval : max_disp;
				min_disp = (min_disp > dval) ? dval : min_disp;
			}
		}
		/*
		//debug 
		max_disp = 0.003;
		*/
		cout << "maximum disparity:" << max_disp << ", minimum disparity:" << min_disp << endl;
		for (int r = 0; r < disp_maps[i].rows; ++r)
		{
			for (int c = 0; c < disp_maps[i].cols; ++c)
			{
				float dval = disp_maps[i].at<float>(r, c);
				disp_images.at<uchar>(r, c) = (uchar)(255 * (dval - min_disp) / (max_disp - min_disp));
			}
		}
		Mat colormap;
		applyColorMap(disp_images, colormap, COLORMAP_JET);
		// apply visibility mask to the disp map
		Mat mask = align_data.img_data[i].scale_mask;
		for (int r = 0; r < mask.rows; ++r)
			for (int c = 0; c < mask.cols; ++c)
				colormap.at<Vec3b>(r, c) = (mask.at<uchar>(r, c) < 128) ? Vec3b(0, 0, 0) : colormap.at<Vec3b>(r, c);
		stringstream sstr;
		sstr << filename << "-" << i << ".jpg";
		imwrite(sstr.str(), colormap);
	}
}

void Optimisor::drawMesh(ImageMesh &mesh, Mat &mesh_img)
{
	for (size_t r = 0; r < mesh.size(); ++r)
	{
		for (size_t c = 0; c < mesh[r].size(); ++c)
		{
			Point2f pt[3];

			pt[0] = mesh[r][c];
			if (r < mesh.size() - 1)
				pt[1] = mesh[r + 1][c];
			if (c < mesh[r].size() - 1)
				pt[2] = mesh[r][c + 1];
			if (r < mesh.size() - 1)
				line(mesh_img, pt[0], pt[1], Scalar(0, 255, 0), 5);
			if (c < mesh[r].size() - 1)
				line(mesh_img, pt[0], pt[2], Scalar(0, 255, 0), 5);
		}
	}
}

void Optimisor::Barycentric(Point2f p, Point2f a, Point2f b, Point2f c, float &u, float &v, float &w)
{
	Vec2f v0 = b - a, v1 = c - a, v2 = p - a;
	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = d00 * d11 - d01 * d01;
	v = (d11 * d20 - d01 * d21) / denom;
	w = (d00 * d21 - d01 * d20) / denom;
	u = 1.0f - v - w;
}
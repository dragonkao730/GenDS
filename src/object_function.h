#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <Eigen/Eigen>
#include "linear_solve.h"

using namespace std;
using namespace Eigen;

#define PI (atan(1) * 4)

struct GridInfo
{
    int n_frame;

    double image_w;
    double image_h;

    int n_rect_row;
    int n_rect_col;

    double rect_w;
    double rect_h;

    int n_vertex_row;
    int n_vertex_col;

    double rect_diagonal_slop;

    GridInfo(const int n_frame,
             const int n_rect_row,
             const int n_rect_col)
    {
        this->n_frame = n_frame;
        image_w = 2.0 * PI;
        image_h = PI;
        this->n_rect_row = n_rect_row;
        this->n_rect_col = n_rect_col;
        rect_w = image_w / n_rect_col;
        rect_h = image_h / n_rect_row;
        n_vertex_row = n_rect_row + 1;
        n_vertex_col = n_rect_col + 1;
        rect_diagonal_slop = rect_h / rect_w;
    }
};

inline void
AddCoefficient(Constrain &constrain,
               const GridInfo &grid_info,
               int frame_index,
               int vertex_row_index,
               int vertex_col_index,
               double coefficient)
{
    // frame
    assert(frame_index >= 0 && frame_index < grid_info.n_frame);
    // row
    assert(vertex_row_index >= 0 && vertex_row_index < grid_info.n_vertex_row);
    // col
    while (vertex_col_index < 0) // loop grid
        vertex_col_index += grid_info.n_vertex_col - 1;
    vertex_col_index %= grid_info.n_vertex_col - 1;
    // all
    int param_index =
        frame_index * grid_info.n_vertex_row * (grid_info.n_vertex_col - 1) +
        vertex_row_index * (grid_info.n_vertex_col - 1) +
        vertex_col_index;
    constrain.coefficients.push_back(pair<int, double>(param_index, coefficient));
};

vector<Constrain> GetSpatialSmoothConstraint(const GridInfo &grid_info);

vector<Constrain> GetTemporialSmoothConstraint(const GridInfo &grid_info, const int sild_window_w);

struct DepthPoint
{
    double theta;
    double phi;
    double depth;
    int frame_index;
};


vector<Constrain> GetDepthConstraint(const GridInfo &grid_info,
                                     const vector<DepthPoint> &depth_point_list);


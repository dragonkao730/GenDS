#include "object_function.h"

/*
inline Vector3d GetBarycentricVector(const ThetaPhiCoor &p,
                                     const ThetaPhiCoor &a,
                                     const ThetaPhiCoor &b,
                                     const ThetaPhiCoor &c)
{
    Vector2d v0 = b - a, v1 = c - a, v2 = p - a;
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;
    return Vector3d(u, v, w);
}

inline BarycentricCoor ThetaPhiCoor2BarycentricCoor(const ThetaPhiCoor &p,
                                                    const GridInfo &grid_info)
{
    const int rect_row_index = p(1) / grid_info.rect_h;
    const int rect_col_index = p(0) / grid_info.rect_w;

    assert(rect_row_index >= grid_info.n_rect_row);
    assert(rect_col_index >= grid_info.n_rect_col);

    // grid_vertex

    // gv[0] - gv[1]
    // |     \     |
    // gv[2] - gv[3]

    ThetaPhiCoor gv[4];
#define GRID_TP_COOR(i, j) ThetaPhiCoor((rect_col_index + i) * grid_info.rect_w, (rect_row_index + j) * grid_info.rect_h)
    gv[0] = GRID_TP_COOR(0, 0);
    gv[1] = GRID_TP_COOR(1, 0);
    gv[2] = GRID_TP_COOR(0, 1);
    gv[3] = GRID_TP_COOR(1, 1);
#undef GRID_TP_COOR

    // slop
    const double slop = (p(1) - gv[0](1)) / (p(0) - gv[0](0));

    // barycentric coor
    BarycentricCoor barycentric_coor;
    if (slop > grid_info.rect_diagonal_slop) // lower
    {
        barycentric_coor.vertex_row_index = Vector3i(rect_row_index, rect_row_index + 1, rect_row_index + 1);
        barycentric_coor.vertex_col_index = Vector3i(rect_col_index, rect_col_index, rect_row_index + 1);
        barycentric_coor.coefficient = GetBarycentricVector(p, gv[0], gv[2], gv[3]);
    }
    else // upper
    {
        barycentric_coor.vertex_row_index = Vector3i(rect_row_index, rect_row_index, rectrow_index + 1);
        barycentric_coor.vertex_col_index = Vector3i(rect_col_index, rect_col_index + 1, rect_row_index + 1);
        barycentric_coor.coefficient = GetBarycentricVector(p, gv[0], gv[1], gv[3]);
    }

    return barycentric_coor;
};

vector<Constrain> GetDepthConstraint(const vector<DepthPoint> &depth_point_list,
                                     const GridInfo &grid_info)
{
    const int n_depth_point = depth_point_list.size();

    // barycentric
    vector<BarycentricCoor> barycentric_coor_list(n_depth_point);
    for(int i=0; i<n_depth_point; i++) // parallel???
    {
        barycentric_coor_list[i] =
            ThetaPhiCoor2BarycentricCoor(depth_point_list.theta_phi, grid_info);
    }

    // constrain
    vector<Constrain> constrain_list(n_depth_point);
    for(int i=0; i<n_depth_point; i++) // parallel???
    {
        Constrain& constrain = constrain_list[i];
        for(int j=0; j<3; j++)
        {
            const int vertex_row_index = barycentric_coor_list[i].vertex_row_index(j);
            const int vertex_col_index = barycentric_coor_list[i].vertex_col_index(j);
            const int frame_index = depth_point_list[i].frame_index;
            const double coefficient = barycentric_coor_list[i].coefficient;
            ADD_COEF(vertex_row_index, vertex_col_index, frame_index, coefficient);
        }
        constrain.b = depth_point_list[i].depth;
    }
    
    return constrain_list;
}
*/

#define ADD_COEF(fi, vri, vci, coe) AddCoefficient(constrain, grid_info, fi, vri, vci, coe)

vector<Constrain> GetSpatialSmoothConstraint(const GridInfo &grid_info)
{
    vector<Constrain> constrain_list;
    for (int frame_index = 0; frame_index < grid_info.n_frame; frame_index++)
        for (int vertex_row_index = 0; vertex_row_index < grid_info.n_vertex_row; vertex_row_index++)
            for (int vertex_col_index = 0; vertex_col_index < grid_info.n_vertex_col - 1; vertex_col_index++)
            {
                Constrain constrain;
                if (0 == vertex_row_index) // top row
                {
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index, 1.0);
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index + 1, -1.0 / 3.0);
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index - 1, -1.0 / 3.0);
                    ADD_COEF(frame_index, vertex_row_index + 1, vertex_col_index, -1.0 / 3.0);
                }
                else if (grid_info.n_vertex_row - 1 == vertex_row_index) // bottom row
                {
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index, 1.0);
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index + 1, -1.0 / 3.0);
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index - 1, -1.0 / 3.0);
                    ADD_COEF(frame_index, vertex_row_index - 1, vertex_col_index, -1.0 / 3.0);
                }
                else // else
                {
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index, 1.0);
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index + 1, -0.25);
                    ADD_COEF(frame_index, vertex_row_index, vertex_col_index - 1, -0.25);
                    ADD_COEF(frame_index, vertex_row_index + 1, vertex_col_index, -0.25);
                    ADD_COEF(frame_index, vertex_row_index - 1, vertex_col_index, -0.25);
                }
                constrain_list.push_back(constrain);
            }
    return constrain_list;
}

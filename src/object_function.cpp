#include "object_function.h"

vector<Constrain> GetSpatialSmoothConstraint(const GridInfo &grid_info)
{
    vector<Constrain> constrain_list;
    for (int frame_index = 0; frame_index < grid_info.n_frame; frame_index++)
        for (int vertex_row_index = 0; vertex_row_index < grid_info.n_vertex_row; vertex_row_index++)
            for (int vertex_col_index = 0; vertex_col_index < grid_info.n_vertex_col - 1; vertex_col_index++)
            {
                Constrain constrain;
#define ADD_COEF(fi, vri, vci, coe) AddCoefficient(constrain, grid_info, fi, vri, vci, coe)
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
#undef ADD_COEF
                constrain_list.push_back(constrain);
            }
    return constrain_list;
}

vector<Constrain> GetTemporialSmoothConstraint(const GridInfo &grid_info, const int sild_window_w)
{
    const int l_sid_w = sild_window_w / 2;
    const int r_sid_w = sild_window_w - l_sid_w - 1;
    vector<Constrain> constrain_list;
    for (int frame_index = 0; frame_index < grid_info.n_frame; frame_index++)
    {
        const int s_frame_index = max(0, frame_index - l_sid_w);
        const int e_frame_index = min(grid_info.n_frame, frame_index + r_sid_w + 1);
        const double mid_coef = 1.0;
        const double sid_coef = -1.0 / (e_frame_index - s_frame_index - 1);
        for (int vertex_row_index = 0; vertex_row_index < grid_info.n_vertex_row; vertex_row_index++)
        {
            for (int vertex_col_index = 0; vertex_col_index < grid_info.n_vertex_col - 1; vertex_col_index++)
            {
                Constrain constrain;
#define ADD_COEF(fi, coe) AddCoefficient(constrain, grid_info, fi, vertex_row_index, vertex_col_index, coe)
                for (int i = s_frame_index; i < frame_index; i++)
                    ADD_COEF(i, sid_coef);
                ADD_COEF(frame_index, mid_coef);
                for (int i = frame_index + 1; i < e_frame_index; i++)
                    ADD_COEF(i, sid_coef);
#undef ADD_COEF
                constrain_list.push_back(constrain);
            }
        }
    }
    return constrain_list;
}

/*

struct BarycentricCoor
{
    Vector3i vertex_row_index;
    Vector3i vertex_col_index;
    Vector3d coefficient;
};

*/

// coefficient
inline Vector3d
GetBarycentric(const Vector2d &p, // theta phi
               const Vector2d &a,
               const Vector2d &b,
               const Vector2d &c)
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

// row_index, col_index, coefficient
inline tuple<int[3], int[3], double[3]>
ThetaPhi2Barycentric(const GridInfo &grid_info,
                     const double theta,
                     const double phi)
{
    const int rect_row_index = min(grid_info.n_rect_row - 1, int(phi / grid_info.rect_h));
    const int rect_col_index = min(grid_info.n_rect_col - 1, int(theta / grid_info.rect_w));
    // gv = grid vertex
    //
    // gv[0] - gv[1] < upper
    // |     \     |
    // gv[2] - gv[3]
    // ^
    // lower
    Vector2d gv[4];
#define GET_GV(i, j) Vector2d((rect_col_index + i) * grid_info.rect_w, (rect_row_index + j) * grid_info.rect_h)
    gv[0] = GET_GV(0, 0);
    gv[1] = GET_GV(1, 0);
    gv[2] = GET_GV(0, 1);
    gv[3] = GET_GV(1, 1);
#undef GET_GV
    // slop
    const double slop = (phi - gv[0](1)) / (theta - gv[0](0));
    // barycentric coor
    tuple<int[3], int[3], double[3]> barycentric;
#define SET_BARYCENTRIC(i, v1, v2, v3) \
    get<i>(barycentric)[0] = v1;       \
    get<i>(barycentric)[1] = v2;       \
    get<i>(barycentric)[2] = v3;
    if (slop > grid_info.rect_diagonal_slop) // lower
    {
        SET_BARYCENTRIC(0, rect_row_index, rect_row_index + 1, rect_row_index + 1); // vertex_row_index
        SET_BARYCENTRIC(1, rect_col_index, rect_col_index, rect_col_index + 1);     // vertex_col_index
        Vector3d coefficient = GetBarycentric(Vector2d(theta, phi), gv[0], gv[2], gv[3]);
        SET_BARYCENTRIC(2, coefficient(0), coefficient(1), coefficient(2)); // coefficient
    }
    else // upper
    {
        SET_BARYCENTRIC(0, rect_row_index, rect_row_index, rect_row_index + 1);     // vertex_row_index
        SET_BARYCENTRIC(1, rect_col_index, rect_col_index + 1, rect_col_index + 1); // vertex_col_index
        Vector3d coefficient = GetBarycentric(Vector2d(theta, phi), gv[0], gv[1], gv[3]);
        SET_BARYCENTRIC(2, coefficient(0), coefficient(1), coefficient(2)); // coefficient
    }
#undef SET_BARYCENTRIC
    return barycentric;
};

vector<Constrain> GetDepthConstraint(const GridInfo &grid_info,
                                     const vector<DepthPoint> &depth_point_list)
{
    const int n_depth_point = depth_point_list.size();
    vector<Constrain> constrain_list(n_depth_point);
    for (int depth_point_index = 0; depth_point_index < n_depth_point; depth_point_index++) // parallel???
    {
        const DepthPoint &depth_point = depth_point_list[depth_point_index];
        Constrain &constrain = constrain_list[depth_point_index];
        // barycentric = row_index, col_index, coefficient
        tuple<int[3], int[3], double[3]> barycentric = ThetaPhi2Barycentric(grid_info,
                                                                            depth_point.theta,
                                                                            depth_point.phi);
        int *vertex_row_index = get<0>(barycentric);
        int *vertex_col_index = get<1>(barycentric);
        double *coefficient = get<2>(barycentric);
        // add coefficient
        const int frame_index = depth_point.frame_index;

        static int testest = 0;

        for (int i = 0; i < 3; i++)
        {
            if(testest == 10)
                cout<<vertex_row_index[i]<<"&&"<<vertex_col_index[i]<<endl;
            AddCoefficient(constrain,
                           grid_info,
                           frame_index,
                           vertex_row_index[i],
                           vertex_col_index[i],
                           coefficient[i]);
        }

        testest++;

        constrain.b = depth_point.depth;
    }
    return constrain_list;
}

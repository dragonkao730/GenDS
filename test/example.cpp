#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <cmath>

#include <gends.h>



using namespace std;

#define PI (atan(1) * 4)

vector<string> GetFilePathsInDir(const char *dir_path)
{
    DIR *dir;
    dirent *ent;
    char filename[256];
    vector<string> file_paths;
    if ((dir = opendir(dir_path)) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..") || !strcmp(ent->d_name, ".DS_Store"))
                continue;
            sprintf(filename, "%s/%s", dir_path, ent->d_name);
            cout << filename << endl;
            file_paths.push_back(string(filename));
        }
    }
    return file_paths;
}

vector<vector<FeaturePair>>
GetFeaturePairList(const string &corrs_dir_path,
                   const double img_w,
                   const double img_h)
{
    const vector<string> corrs_file_list = GetFilePathsInDir(corrs_dir_path.c_str());
    vector<vector<FeaturePair>> all_feature_pair;
    for (auto &corrs_file : corrs_file_list)
    {
        fstream fs(corrs_file.c_str(), fstream::in);
        string line;
        vector<FeaturePair> feature_pair_one_frame;
        while (getline(fs, line))
        {
            if (line.compare(0, 3, "c n") == 0)
            {
                stringstream ss;
                ss << line;
                string s;

                FeaturePair corr;
                while (ss >> s)
                {
                    if (s.compare(0, 1, "n") == 0)
                        corr.first.camera_index = atoi(&s[1]);
                    else if (s.compare(0, 1, "N") == 0)
                        corr.second.camera_index = atoi(&s[1]);
                    else if (s.compare(0, 1, "x") == 0)
                        corr.first.theta_phi(0) = atof(&s[1]) * 2.0 * PI / img_w;
                    else if (s.compare(0, 1, "y") == 0)
                        corr.first.theta_phi(1) = atof(&s[1]) * PI / img_h;
                    else if (s.compare(0, 1, "X") == 0)
                        corr.second.theta_phi(0) = atof(&s[1]) * 2.0 * PI / img_w;
                    else if (s.compare(0, 1, "Y") == 0)
                        corr.second.theta_phi(1) = atof(&s[1])  * PI / img_h;
                }

                feature_pair_one_frame.push_back(corr);
            }
        }
        fs.close();
        all_feature_pair.push_back(feature_pair_one_frame);
        if (all_feature_pair.size() >= 20)
            break;
    }
    return all_feature_pair;
}

PolyCamera GetCamera(const string &filename)
{
    fstream fs(filename.c_str(), fstream::in);
    int num_cam;
    fs >> num_cam;
    float pos[4][3];
    float ori[4][3];
    float up[4][3];
    float fovy, ratio;
    for (int c = 0; c < num_cam; ++c)
    {
        fs >> pos[c][0] >> pos[c][1] >> pos[c][2];
        fs >> ori[c][0] >> ori[c][1] >> ori[c][2];
        fs >> up[c][0] >> up[c][1] >> up[c][2];
        fs >> fovy >> ratio;
    }
    Vector3d v4[4];
    for (int i = 0; i < 4; i++)
        v4[i] = Vector3d(pos[i][0], pos[i][1], pos[i][2]);
    return PolyCamera(v4[0], v4[1], v4[2], v4[3]);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Example usage:" << endl;
        cout << "gends data_path" << endl;
        return -1;
    }

    const string data_path = argv[1];
    cout << "data_path = " << data_path << endl;

    const string corrs_dir_path = data_path + "/corrs";
    cout << "corrs_dir_path = " << corrs_dir_path << endl;

    const string polycam_config_path = data_path + "/cams_config/polycam.cam";
    cout << "polycam_config_path = " << polycam_config_path << endl;

    const vector<vector<FeaturePair>> feature_pair_list = GetFeaturePairList(corrs_dir_path, 2000, 1000);
    const PolyCamera ploy_camera = GetCamera(polycam_config_path);

    Tensor<double, 3> output = GenerateDeformableSphere(feature_pair_list,
                                                        ploy_camera);

    cout << output.dimension(0) << endl;
    cout << output.dimension(1) << endl;
    cout << output.dimension(2) << endl;

    for (int frame_index = 0; frame_index < output.dimension(0); frame_index++)
    {
        cout << "frame_index = " << frame_index << endl;
        for (int row_index = 0; row_index < output.dimension(1); row_index++)
        {
            for (int col_index = 0; col_index < output.dimension(2) + 1; col_index++)
            {
                int real_col_index = col_index % output.dimension(2);
                cout << output(frame_index, row_index, real_col_index) << endl;
            }
        }
    }
    return 0;
}
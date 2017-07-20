#include <string>
#include <vector>
#include <iostream>
#include <dirent.h>

#include "alignment.h"

using namespace std;

string img_dir = "aquila/0421";
string mask_dir = "mask/aquila";
string feat_dir = "feature_all";

vector<string> img_names;
vector<string> mask_names;
vector<string> feat_names;

vector<string> getFilePathsInDir(const char *dir_path)
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

int main(int argc, char **argv)
{
	const string usage = "Example usage:\ngends corrs_dir_path polycam_config_path imgs_dir_path masks_dir_path\n ";
	if (argc != 5)
	{
		cout << usage << endl;
		return 0;
	}

	const string corrs_dir_path = argv[1];
	const string polycam_config_path = argv[2];
	const string imgs_dir_path = argv[3];
	const string masks_dir_path = argv[4];

	cout << "corrs_dir_path = " << corrs_dir_path << endl;
	cout << "polycam_config_path = " << polycam_config_path << endl;
	cout << "imgs_dir_path = " << imgs_dir_path << endl;
	cout << "masks_dir_path = " << masks_dir_path << endl;

	vector<string> corrs_file_paths = getFilePathsInDir(corrs_dir_path.c_str());
	vector<string> imgs_file_paths = getFilePathsInDir(imgs_dir_path.c_str());
	vector<string> masks_file_paths = getFilePathsInDir(masks_dir_path.c_str());
	
	Alignment aligner(	imgs_file_paths,
						masks_file_paths,
						corrs_file_paths,
						polycam_config_path);
	aligner.aggregation();

	return 0;
}
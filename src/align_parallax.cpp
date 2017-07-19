#include "alignment.h"
//#include <opencv2\core\core.hpp>
//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
#include "dirent.h"
#include <iostream>
#include "align_data.h"
//#include <time.h>

using namespace std;
//using namespace cv;

string img_dir = "aquila/0421";
string mask_dir = "mask/aquila";
string feat_dir = "feature_all";

vector<string> img_names;
vector<string> mask_names;
vector<string> feat_names;

void getInputFiles(const char* imgdir){
	DIR* dir;
	dirent* ent;
	char filename[256];
	if((dir=opendir(imgdir)) != NULL ){
		while( (ent=readdir(dir))!= NULL){
			if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
				continue;
			sprintf(filename, "%s/%s", imgdir, ent->d_name);
			cout << filename<<endl;
			img_names.push_back(string(filename));
		}
	}
}
void getMaskFiles(const char* maskdir){
	DIR* dir;
	dirent* ent;
	char filename[256];
	if((dir=opendir(maskdir)) != NULL ){
		while( (ent=readdir(dir))!= NULL){
			if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
				continue;
			sprintf(filename, "%s/%s", maskdir, ent->d_name);
			cout << filename<<endl;
			mask_names.push_back(string(filename));
		}
	}
}
void getFeatFiles(const char* maskdir){
	DIR* dir;
	dirent* ent;
	char filename[256];
	if((dir=opendir(maskdir)) != NULL ){
		while( (ent=readdir(dir))!= NULL){
			if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
				continue;
			sprintf(filename, "%s/%s", maskdir, ent->d_name);
			cout << filename<<endl;
			feat_names.push_back(string(filename));
		}
	}
}

int main(int argc, char** argv){
	//unsigned long startTime = clock();
	getInputFiles(img_dir.c_str());
	getMaskFiles(mask_dir.c_str());
	getFeatFiles(feat_dir.c_str());
	//string feat_file = "feature/equi-corr-sparse.txt";
	string feat_file = "feature/equi-feats.txt";
	string cam_file = "auxiliary/polycam.cam";
	AlignData align_data;
	//Alignment  aligner(img_names, mask_names, feat_file, cam_file);
	Alignment  aligner(img_names, mask_names, feat_names, cam_file);
	aligner.aggregation();
	//unsigned long endTime = clock();
	//cout << "Toatal Time: " << (endTime-startTime)/1000.0 << endl;
	//system("pause");
	return 0;
}
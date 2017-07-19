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
	
	getInputFiles(img_dir.c_str());
	getMaskFiles(mask_dir.c_str());
	getFeatFiles(feat_dir.c_str());
	
	string feat_file = "feature/equi-feats.txt";
	string cam_file = "auxiliary/polycam.cam";
	
	Alignment  aligner(img_names, mask_names, feat_names, cam_file);
	aligner.aggregation();
	
	return 0;
}
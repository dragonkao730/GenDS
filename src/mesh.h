#ifndef GLOBAL_VARIABLES_H
#define GLOBAL_VARIABLES_H

#include <opencv2/highgui/highgui.hpp>
#include <string>
using namespace cv;

struct Grid{
	float GRID_WIDTH;
	float GRID_HEIGHT;
};
struct Mesh{
	vector< vector<CvPoint2D32f> > in_mesh;
	vector< vector<CvPoint2D32f> > out_mesh;
	CvMat *mat_out_mesh;
};
struct Match{
	// index of match pair
	Match(){
		matchPic.resize(2);
		matchI.resize(2);
	}
	vector<int> matchPic;
	vector<int> matchI;

	//frank:
	//ex: match pair of img3 and img5
	//then matchPic[0] = 3, matchPic[1] = 5
	//     matchI[0] = 5, matchI[1] = 3
};
struct MatFA{
	CvMat *mat_p;
	CvMat *mat_q;
	int idx_p, idx_q;
};
// frank: each image is associated with a Data
struct Data{
	vector<Mesh> data_mesh; // frank: size is always Global::numPic
	Mesh result_mesh;
	int min_x; int min_y;
	int max_x; int max_y;
	IplImage *img; // frank: original image
	IplImage *scale_img; // frank: scaled (down) image
	vector<IplImage*> warp_img; // frank: size is always Global::numPic
};
struct Feature{
	Feature(){
		idx_quad = -1;
		pos.x = -1;		pos.y = -1;
		vertex.resize(4);	alpha.resize(4);
	}
	CvPoint2D32f pos;
	int idx_quad;
	// ---- block information----
	//	(0)-(1)
	//   |   |
	//  (2)-(3)
	vector<int> vertex; // vertex number
	vector<double> alpha;

	//frank:
	//records which quad and vertices surround the feature
	//idx_quad: 1D idx of quad
	//vertex: 1D indices of the 4 surrounding vertices
	//alpha: corresponding coeff for interp.
};
struct featurePic{
	// featurePic: numPic [idx_pic]
	// row: numPic [idx_i]
	// col: N [numFeat]
	vector< vector<Feature> > featInfo;
	vector<bool> isExist;
	vector<CvMat*> matFA;
	vector<CvMat*> matFb;

	// frank:
	// ex: "static vector<featurePic> featPic;"
	// featPic[idx_pic].featInfo[idx_i] is a vector of "Struct Feature" for image pair (idx_pic, idx_i)
};

class Global{
	public:
		static bool skip_SIFT;			static bool skip_ALIGN;
		static string detection_type;	static string descriptor_type;		static string matcher_type;
		static string inputPath;		static string outputPath;
		static string sift_list;		static string align_list;
		static string warpA;			static string warpB;			static string output;
		static int IDX_ORIGIN;			static int IDX_END;
		static double END_WIDTH;		static double END_HEIGHT;
		static float SCALE;			static int numPic;
		static int numVer_Col;		static int numVer_Row;
		static int coarse2fine_Count;
		static int ugly_Count;		static int stop_Count;

		static int n_feature;		static int n_shape;			static int n_quad;
		static int n_origin;		static int n_length;		static int n_hor_ver;
		static int SCREEN_WIDTH;		static int SCREEN_HEIGHT;
		static float minxAll, minyAll;

		static vector<featurePic> featPic;
		static vector< vector<int> > matchList;
		static vector<Match> matchPair;
		static vector<MatFA> matFeat;
		static vector<Mesh> meshPic;

};

#endif //GLOBAL_VARIABLES_H
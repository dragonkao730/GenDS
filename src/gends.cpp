
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

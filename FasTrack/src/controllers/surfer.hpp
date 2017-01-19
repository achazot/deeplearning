#ifndef SURFER_HPP 
#define SURFER_HPP

#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include <limits>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std; 

class Surfer 
{
	public:
		Surfer();
		Surfer(int hessian);
		void initialize(Mat object);
		float match(Mat& scene, Rect position);

	private:
		Mat ref_object; 
		int hessian_value = 800; 	
		std::vector<KeyPoint> obj_keypoints, scene_keypoints; 
		Mat obj_descriptors, scene_descriptors; 
};

#endif // SURFER_HPP 
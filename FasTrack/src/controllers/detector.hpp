#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <dirent.h>
#include <fstream>
#include <list>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../models/detectorResult.hpp"

using namespace caffe;
using namespace std;

#define amax(a, b) (((a)>(b)) ? (a) : (b))
#define amin(a, b) (((a)<(b)) ? (a) : (b))

class Detector
{
public:
	Detector ( );
	void initialize(const string& model_file, const string& weights_file, const string& labels_file);
	void Detection(const string& im_name);
	void Detection(cv::Mat cv_img);
	void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
	void boxes_sort(int num, const float* pred, float* sorted_pred);
	void read_classes(const string& path);
	bool initialized();
	vector<DetectorResult> getResults ( );

private:
	vector<DetectorResult> m_results;
	vector<string> m_classes;
	Net<float> *net_;
	bool m_initialized;
};


#endif

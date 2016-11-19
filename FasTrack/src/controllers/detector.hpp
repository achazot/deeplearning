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

using namespace caffe;
using namespace std;

#define max(a, b) (((a)>(b)) ? (a) : (b))
#define min(a, b) (((a)<(b)) ? (a) : (b))
const int class_num=21;

class Detector
{
public:
	Detector ( );
	void initialize(const string& model_file, const string& weights_file, const string& labels_file);
	void Detection(const string& im_name);
	void Detection(cv::Mat cv_img, const string& im_name);
	void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
	void vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH, int nclass);
	void boxes_sort(int num, const float* pred, float* sorted_pred);
	void read_classes(const string& path);
	bool initialized();

private:
	vector<string> m_classes;
	Net<float> *net_;
	bool m_initialized;
};


#endif

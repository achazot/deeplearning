#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "trackermodel.hpp"

class Tracker
{
public:
	Tracker ( );
	cv::Rect track ( cv::Mat img );
	void train ( cv::Mat img, cv::Rect pos );
private:

};




#endif // TRACKER_HPP

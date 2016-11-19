#ifndef TRACKERMODEL_HPP
#define TRACKERMODEL_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class TrackerModel
{
public:
	TrackerModel();
private:
	cv::Size m_size;
	cv::HOGDescriptor m_hog;
	cv::Mat m_fghist;
	cv::Mat m_bghist;
};

#endif // TRACKERMODEL_HPP

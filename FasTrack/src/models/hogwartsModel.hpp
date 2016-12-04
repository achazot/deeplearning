#ifndef HOGWARTSMODEL_HPP
#define HOGWARTSMODEL_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class HogwartsModel
{
private:
	cv::Size m_size;
	cv::HOGDescriptor m_hog;
	cv::Mat m_fghist;
	cv::Mat m_bghist;

public:
	HogwartsModel ( cv::Mat img );
	~HogwartsModel ( );
	float compare ( HogwartsModel with );
	void update ( HogwartsModel with, float learnCoeff );
	cv::Size size ( );
	cv::HOGDescriptor hog ( );
	cv::Mat fghist ( );
	cv::Mat bghist ( );
};

#endif // HOGWARTSMODEL_HPP

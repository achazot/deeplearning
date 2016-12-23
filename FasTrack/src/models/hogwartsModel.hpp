#ifndef HOGWARTSMODEL_HPP
#define HOGWARTSMODEL_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <iostream>
#include <math.h>

#define ABS(X) ( (X) > 0 ? (X) : (-(X)) )

class HogwartsModel
{
private:
	cv::Mat m_hog;
	cv::Mat m_fghist;
	cv::Mat m_bghist;

public:
	HogwartsModel ( );
	HogwartsModel ( cv::Mat img );
	~HogwartsModel ( );
	double compare ( HogwartsModel with );
	void update ( HogwartsModel with, float learnCoeff );
	cv::Mat hog ( );
	cv::Mat fghist ( );
	cv::Mat bghist ( );
};

#endif // HOGWARTSMODEL_HPP

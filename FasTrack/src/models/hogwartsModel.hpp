#ifndef HOGWARTSMODEL_HPP
#define HOGWARTSMODEL_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <iostream>
#include <math.h>

#define ABS(X) ( (X) > 0 ? (X) : (-(X)) )

class HogwartsModel
{
private:
	cv::Mat m_hog;
	cv::Mat m_fghist[3];
	cv::Mat m_bghist[3];

public:
	HogwartsModel ( );
	HogwartsModel ( cv::Mat img, cv::Rect pos, bool calcHist = false );
	~HogwartsModel ( );
	double compareHOG ( HogwartsModel with );
	double compareHist ( cv::Mat with );
	void update ( HogwartsModel with, float FGLearn, float BGLearn, float HoGLearn );
	cv::Mat computeHistogram( cv::Mat image, cv::Mat mask, int size );
	cv::Mat hog ( );
	cv::Mat* fghist ( );
	cv::Mat* bghist ( );
};

#endif // HOGWARTSMODEL_HPP

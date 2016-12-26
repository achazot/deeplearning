#ifndef HOGWARTS_HPP
#define HOGWARTS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "../models/hogwartsModel.hpp"

class Hogwarts
{
private:
	HogwartsModel m_currentModel;
	cv::Rect m_previousPosition;
public:
	Hogwarts ( cv::Mat img, cv::Rect position );
	cv::Rect update ( cv::Mat img );
};

#endif // HOGWARTS_HPP

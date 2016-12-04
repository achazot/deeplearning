#include "hogwartsModel.hpp"

HogwartsModel::HogwartsModel ( cv::Mat img )
{

}

HogwartsModel::~HogwartsModel ( )
{

}

float HogwartsModel::compare ( HogwartsModel with )
{

	return 0.f;
}

void HogwartsModel::update ( HogwartsModel with, float learnCoeff )
{

}

cv::Size HogwartsModel::size ( )
{
	return m_size;
}

cv::HOGDescriptor HogwartsModel::hog ( )
{
	return m_hog;
}

cv::Mat HogwartsModel::fghist ( )
{
	return m_fghist;
}

cv::Mat HogwartsModel::bghist ( )
{
	return m_bghist;
}

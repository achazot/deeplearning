#include "hogwartsModel.hpp"


HogwartsModel::HogwartsModel ( )
{
}

HogwartsModel::HogwartsModel ( cv::Mat img )
{
	cv::HOGDescriptor hog(cv::Size(32,16), cv::Size(8,8), cv::Size(4,4), cv::Size(4,4), 9);
	std::vector<cv::Point> locations;
	std::vector<float> descs;
	cv::Mat grey;
	cv::cvtColor(img, grey, CV_RGB2GRAY);

	hog.compute(grey, descs, cv::Size(0,0), cv::Size(0,0), locations);

	m_hog = cv::Mat(descs.size(),1,CV_32FC1);
	memcpy(m_hog.data,descs.data(),descs.size()*sizeof(float));
}

HogwartsModel::~HogwartsModel ( )
{

}

double HogwartsModel::compare ( HogwartsModel with )
{
	double distance = 0;
	distance = cv::norm(m_hog, with.hog(), cv::NORM_L2, cv::noArray());
	return distance;
}

void HogwartsModel::update ( HogwartsModel with, float learnCoeff )
{
	m_hog = (1.f-learnCoeff)*m_hog + with.hog() * learnCoeff;
}

cv::Mat HogwartsModel::hog ( )
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

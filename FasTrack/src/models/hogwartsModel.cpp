#include "hogwartsModel.hpp"


HogwartsModel::HogwartsModel ( )
{
}

HogwartsModel::HogwartsModel ( cv::Mat img )
{
	cv::HOGDescriptor hog(cv::Size(32,16), cv::Size(8,8), cv::Size(4,4), cv::Size(4,4), 9);
	std::vector<cv::Point> locations;
	cv::Mat grey;
	cv::cvtColor(img, grey, CV_RGB2GRAY);

	hog.compute(grey, m_hog, cv::Size(0,0), cv::Size(0,0), locations);

}

HogwartsModel::~HogwartsModel ( )
{

}

double HogwartsModel::compare ( HogwartsModel with )
{
	double distance = 0;
	cv::Mat A(m_hog.size(),1,CV_32FC1);
	memcpy(A.data,m_hog.data(),m_hog.size()*sizeof(float));
	cv::Mat B(with.hog().size(),1,CV_32FC1);
	memcpy(B.data,with.hog().data(),with.hog().size()*sizeof(float));

	cv::Mat C = A-B;
	C = C.mul(C);
	cv::sqrt(C, C);
	cv::Scalar rr = cv::sum(C);
	distance = rr(0);

	return distance;
}

void HogwartsModel::update ( HogwartsModel with, float learnCoeff )
{

}

cv::Size HogwartsModel::size ( )
{
	return m_size;
}

std::vector<float> HogwartsModel::hog ( )
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

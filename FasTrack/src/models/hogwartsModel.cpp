#include "hogwartsModel.hpp"


HogwartsModel::HogwartsModel ( )
{
}

HogwartsModel::HogwartsModel ( cv::Mat img, cv::Rect pos, bool calcHist )
{
	// HoG computing
	cv::HOGDescriptor hog(cv::Size(16,16), cv::Size(8,8), cv::Size(4,4), cv::Size(4,4), 9);
	std::vector<cv::Point> locations;
	std::vector<float> descs;
	cv::Mat grey;

	cv::cvtColor(img(pos), grey, CV_RGB2GRAY);

	hog.compute(grey, descs, cv::Size(0,0), cv::Size(0,0), locations);

	m_hog = cv::Mat(descs.size(),1,CV_32FC1);
	memcpy(m_hog.data,descs.data(),descs.size()*sizeof(float));


	if (calcHist)
	{
		// histogram computing
		int nbins = 32;
		std::vector<cv::Mat> frame_planes;
		cv::Mat img2 = img;

		cv::split( img2, frame_planes );

		cv::Mat fg_mask = cv::Mat( img.size().height, img.size().width, CV_8U, cv::Scalar(0) );
		fg_mask(cv::Rect(42,42,44,44)).setTo(cv::Scalar::all(255));

		cv::Mat bg_mask = cv::Mat( img.size().height, img.size().width, CV_8U, cv::Scalar(255) );
		bg_mask(pos).setTo(cv::Scalar::all(0));

		for (int i=0;i<3;i++) m_bghist[i] = computeHistogram( frame_planes[i], bg_mask, nbins );
		for (int i=0;i<3;i++) m_fghist[i] = computeHistogram( frame_planes[i], fg_mask, nbins );
	}

}

HogwartsModel::~HogwartsModel ( )
{

}

double HogwartsModel::compareHOG ( HogwartsModel with )
{
	double distance = 0;
	distance = cv::norm(m_hog, with.hog(), cv::NORM_L2, cv::noArray());
	return distance;
}

double HogwartsModel::compareHist ( cv::Mat with )
{
	double distance = 0;

	int nbins = 32;
	float lambda = 0.001f;
	double histScore[3];

	cv::Mat roi_mask = cv::Mat( with.size().height, with.size().width, CV_8U, cv::Scalar(255) );
	std::vector<cv::Mat> frame_planes;
	cv::split( with, frame_planes );
	cv::Mat roi_hist[3];
	for (int i=0;i<3;i++) roi_hist[i] = computeHistogram( frame_planes[i], roi_mask, nbins );

	for (int i=0;i<3;i++)
	{
		histScore[i] = 0;
		for (int ib = 0; ib < nbins; ib++)
		{
			histScore[i] += (roi_hist[i].at<float>(ib) * (m_fghist[i].at<float>(ib) / (m_fghist[i].at<float>(ib) + m_bghist[i].at<float>(ib) + lambda)) );
		}
	}
	distance = histScore[2];// + histScore[1] + histScore[2];

	return distance;
}

void HogwartsModel::update ( HogwartsModel with, float learnCoeff1, float learnCoeff2)
{
	m_hog = (1.f-learnCoeff1)*m_hog + with.hog() * learnCoeff1;

	for (int i=0;i<3;i++) m_bghist[i] = m_bghist[i] * (1.f-learnCoeff2) + learnCoeff2 * with.bghist()[i];
	for (int i=0;i<3;i++) m_fghist[i] = m_fghist[i] * (1.f-learnCoeff2) + learnCoeff2 * with.fghist()[i];
}

cv::Mat HogwartsModel::hog ( )
{
	return m_hog;
}

cv::Mat* HogwartsModel::fghist ( )
{
	return m_fghist;
}

cv::Mat* HogwartsModel::bghist ( )
{
	return m_bghist;
}

cv::Mat HogwartsModel::computeHistogram( cv::Mat image, cv::Mat mask, int size )
{
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	cv::Mat hist;
	cv::calcHist( &image, 1, 0, mask, hist, 1, &size, &histRange, true, false );
	//normalize( hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );
	return hist;
}

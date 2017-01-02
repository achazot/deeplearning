
#include "hogwarts.hpp"

Hogwarts::Hogwarts ( cv::Mat img, cv::Rect position )
{
	// get smaller window surrounding object
	cv::Rect swPos (cv::Point(MAX(0, position.x - position.width * 0.5f),
					          MAX(0, position.y - position.height * 0.5f) ),
					cv::Point(MIN(img.size().width, position.x + position.width * 1.5f),
					          MIN(img.size().height, position.y + position.height * 1.5f)));
	cv::Mat subWindow = img(swPos);

	// resize to 128x128
	cv::resize(subWindow, subWindow, cv::Size(128, 128));

	// create first tracking model (calculate HoG and color histograms)
	m_currentModel = HogwartsModel(subWindow, cv::Rect(32, 32, 64, 64), true);
	// set previous position
	m_previousPosition = position;
}

cv::Rect Hogwarts::update ( cv::Mat img )
{
	// TODO: floating point operations should be transformed into byte ops
	// TODO: compute histogram integral response better


	// get smaller window surrounding object
	cv::Rect swPos (cv::Point(MAX(0, m_previousPosition.x - m_previousPosition.width * 0.5f),
					          MAX(0, m_previousPosition.y - m_previousPosition.height * 0.5f) ),
					cv::Point(MIN(img.size().width, m_previousPosition.x + m_previousPosition.width * 1.5f),
					          MIN(img.size().height, m_previousPosition.y + m_previousPosition.height * 1.5f)));
	cv::Mat subWindow = img(swPos);

	// resize to 128x128
	cv::resize(subWindow, subWindow, cv::Size(128, 128));

	// calc inverse position transform
	float ixscale = swPos.width / 128.f;
	float iyscale = swPos.height / 128.f;




	cv::Mat hoglhmap(64,64, CV_64FC1, cv::Scalar(DBL_MAX));
	cv::Mat hislhmap(128,128, CV_64FC1, cv::Scalar(0));

	int step = 2;

	for ( int x = 0; x < 128; x+=step )
	for ( int y = 0; y < 128; y+=step )
	{
		double res = m_currentModel.compareHist(subWindow(cv::Rect(x,y,step,step)));

		for (int ix = 0; ix < step; ix++)
		for (int iy = 0; iy < step; iy++)
		{
			if (y + iy >= hislhmap.size().height || x + ix >= hislhmap.size().width ) continue;
			hislhmap.at<double>(y + iy, x + ix) = res;
		}
	}

	cv::Mat hislresp(128, 128, CV_64FC1, cv::Scalar(0));
	cv::Mat hislintg;
	cv::normalize( hislhmap, hislhmap, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
	cv::imshow("hislhmap", hislhmap);
	cv::moveWindow("hislhmap",500,0);

	// cv::integral(hislhmap, hislintg);

	// histogram response based on previous image size (here fixed as 64)
	/*
	for (int ix = 1; ix < 127; ix ++)
	for (int iy = 1; iy < 127; iy ++)
	{
		cv::Point O(0,0);
		cv::Point A(ix, iy);
		cv::Point B(MIN(ix+64, 127), iy);
		cv::Point C(ix, MIN(iy+64, 127));
		cv::Point D(MIN(ix+64,127), MIN(iy+64,127));
		cv::Scalar Asum = cv::sum(hislintg(cv::Rect(O, A)));
		cv::Scalar Bsum = cv::sum(hislintg(cv::Rect(O, B)));
		cv::Scalar Csum = cv::sum(hislintg(cv::Rect(O, C)));
		cv::Scalar Dsum = cv::sum(hislintg(cv::Rect(O, D)));
		hislresp.at<double>(ix, iy) = (Asum[0] + Dsum[0] - Bsum[0] - Csum[0]);///((D.x-A.x)*(D.y-A.y));
	}
	*/
	cv::GaussianBlur(hislhmap,hislresp, cv::Size(63,63), -1);
	cv::normalize( hislresp, hislresp, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

	cv::imshow("hislresp", hislresp);
	cv::moveWindow("hislresp",500,200);

	for ( int x = 0; x < 64; x+=step )
	for ( int y = 0; y < 64; y+=step )
	{
		HogwartsModel newmodel = HogwartsModel (subWindow, cv::Rect(x, y, 64, 64));
		double res = m_currentModel.compareHOG(newmodel);

		for (int ix = 0; ix < step; ix++)
		for (int iy = 0; iy < step; iy++)
		{
			if (y + iy >= hoglhmap.size().height || x + ix >= hoglhmap.size().width ) continue;
			hoglhmap.at<double>(y + iy, x + ix) = res;
		}
	}
	cv::normalize( hoglhmap, hoglhmap, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
	cv::Mat hoglresp(128,128, CV_64FC1, cv::Scalar(0));
	for ( int x = 0; x < 64; x++ )
	for ( int y = 0; y < 64; y++ )
		hoglresp.at<double>(y + 32, x + 32) = 1.0 - hoglhmap.at<double>(y, x);

	cv::imshow("hoglresp", hoglresp);
	cv::moveWindow("hoglresp",500,400);

	cv::Mat finalmap(128,128, CV_64FC1, cv::Scalar(0));
	finalmap = hoglresp * 0.7f + hislresp * 0.3f;
	//cv::normalize( finalmap, finalmap, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
	//cv::GaussianBlur(finalmap,finalmap, cv::Size(9,9), 0);
	cv::imshow("finalmap", finalmap);
	cv::moveWindow("finalmap",500,600);

	// cv::waitKey(0);

	double bres = 0;
	int bx = 0;
	int by = 0;

	for ( int x = 0; x < 128; x++ )
	for ( int y = 0; y < 128; y++ )
	{
		double res = finalmap.at<double>(y, x);
		if ( res > bres)
		{
			bres = res;
			bx = x;
			by = y;
		}
	}

	m_previousPosition = cv::Rect((bx-64) * ixscale + m_previousPosition.x, (by-64) * iyscale + m_previousPosition.y, m_previousPosition.width, m_previousPosition.height);

	m_currentModel.update(HogwartsModel (subWindow, cv::Rect(32, 32, 64, 64), true), 0.00f, 0.04f);

	return m_previousPosition;
}


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




	cv::Mat hislhmap(128,128, CV_64FC1, cv::Scalar(0));

	int step = 4;

	for ( int x = 0; x < 128; x+=step )
	for ( int y = 0; y < 128; y+=step )
	{
		double res = m_currentModel.compareHist(subWindow(cv::Rect(x,y,step,step)));

		for (int ix = 0; ix < step; ix++)
		for (int iy = 0; iy < step; iy++)
		{
			hislhmap.at<double>(y + iy, x + ix) = res;
		}
	}

	cv::Mat hislresp(128, 128, CV_64FC1, cv::Scalar(0));
	//cv::Mat hislintg;
	cv::normalize( hislhmap, hislhmap, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
	cv::imshow("hislhmap", hislhmap);
	cv::moveWindow("hislhmap",500,0);

	// cv::integral(hislhmap, hislintg);
	// cv::normalize( hislintg, hislintg, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

	//cv::Mat grad = cv::imread("test.png", CV_LOAD_IMAGE_GRAYSCALE);

	// histogram response based on previous image size (here fixed as 64)
	/*
	for (int ix = 0; ix < 96; ix ++)
	for (int iy = 0; iy < 96; iy ++)
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

	cv::GaussianBlur(hislhmap,hislresp, cv::Size(63,63), 0);

	cv::normalize( hislresp, hislresp, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

	cv::imshow("hislresp", hislresp);
	cv::moveWindow("hislresp",500,200);


	// Multithreaded HoG calculation over whole subwindow (128x128) using patches of 64x64

	cv::Mat hoglhmap(64,64, CV_64FC1, cv::Scalar(0));
	std::thread t[16];

	cv::Mat hogmatrix[4][4];
	for ( int x = 0; x < 4; x++ )
	for ( int y = 0; y < 4; y++ )
	{
		hogmatrix[x][y] = cv::Mat (64,64, CV_64FC1, cv::Scalar(0));
		t[x*4+y] = std::thread(hogThread, x * 4, y * 4, m_currentModel, subWindow, std::ref(hogmatrix[x][y]), step);
	}

	for ( int x = 0; x < 4; x++ )
	for ( int y = 0; y < 4; y++ )
	{
		t[x*4+y].join();
		hoglhmap = hoglhmap + hogmatrix[x][y];
	}

	cv::normalize( hoglhmap, hoglhmap, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

	cv::Mat hoglresp(128,128, CV_64FC1, cv::Scalar(0));
	for ( int x = 0; x < 64; x++ )
	for ( int y = 0; y < 64; y++ )
		hoglresp.at<double>(y + 32, x + 32) = 1.0 - hoglhmap.at<double>(y, x);

	cv::imshow("hoglresp", hoglresp);
	cv::moveWindow("hoglresp",500,400);


	// Mix the two feature likelihood maps
	cv::Mat finalmap(128,128, CV_64FC1, cv::Scalar(0));
	finalmap = hoglresp * 0.4f + hislresp * 0.6f;

	cv::imshow("finalmap", finalmap);
	cv::moveWindow("finalmap",500,600);

	// search best value
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

	// save new position
	m_previousPosition = cv::Rect((bx-64) * ixscale + m_previousPosition.x, (by-64) * iyscale + m_previousPosition.y, m_previousPosition.width, m_previousPosition.height);

	// train
	m_currentModel.update(HogwartsModel (subWindow, cv::Rect(bx-32, by-32, 64, 64), true), 0.0f, 0.04f);

	return m_previousPosition;
}

void Hogwarts::hogThread(int x, int y, HogwartsModel cur, cv::Mat img, cv::Mat &res, int step)
{

	for (int ix = 0; ix < 16; ix+=step)
	for (int iy = 0; iy < 16; iy+=step)
	{
		HogwartsModel newmodel = HogwartsModel (img, cv::Rect(x*4+ix, y*4+iy, 64, 64));
		double dres = newmodel.compareHOG(cur);

		for (int jx = 0; jx < step; jx++)
		for (int jy = 0; jy < step; jy++)
		{
			if (y*4+iy+jy > res.size().height || x*4+ix+jx > res.size().width) continue;
			res.at<double>(y*4+iy+jy, x*4+ix+jx) = dres;
		}
	}
}

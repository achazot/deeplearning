
#include "hogwarts.hpp"

Hogwarts::Hogwarts ( cv::Mat img, cv::Rect position )
{
	float xscale = 64.f / position.width;
	float yscale = 64.f / position.height;
	m_previousPosition = position;
	cv::Rect rescaledPos = cv::Rect(position.x * xscale, position.y * yscale, 64, 64);
	cv::resize(img, img, cv::Size(img.size().width * xscale, img.size().height * yscale));
	m_currentModel = HogwartsModel(img, rescaledPos);
}

cv::Rect Hogwarts::update ( cv::Mat img )
{
	// TODO:resize image and do fixed operations

	float xscale = 64.f / m_previousPosition.width;
	float yscale = 64.f / m_previousPosition.height;

	cv::resize(img, img, cv::Size(img.size().width * xscale, img.size().height * yscale));


	double bres = DBL_MAX;
	int bx = 0;
	int by = 0;

	int bpt1x = m_previousPosition.x * xscale - 32;
	int bpt1y = m_previousPosition.y * yscale - 32;
	int bpt2x = m_previousPosition.x * xscale + 96;
	int bpt2y = m_previousPosition.y * yscale + 96;

	bpt1x = bpt1x < 0 ? 0 : bpt1x;
	bpt1y = bpt1y < 0 ? 0 : bpt1y;
	bpt2x = bpt2x > img.size().width - 64 ? img.size().width - 64 : bpt2x;
	bpt2y = bpt2y > img.size().height - 64 ? img.size().height - 64 : bpt2y;

	// dessiner courbe différences
	for ( int x = bpt1x; x < bpt2x - 9; x+=8 )
	for ( int y = bpt1y; y < bpt2y - 9; y+=8 )
	{
		HogwartsModel newmodel = HogwartsModel (img, cv::Rect(x, y, 64, 64));
		double res = m_currentModel.compare(newmodel);

		if ( res < bres )
		{
			bres = res;
			bx = x;
			by = y;
		}
	}

	m_previousPosition = cv::Rect(bx / xscale, by / yscale, m_previousPosition.width, m_previousPosition.height);

	m_currentModel.update(HogwartsModel (img, cv::Rect(bx, by, 64, 64)), 0.000001f);

	return m_previousPosition;
}

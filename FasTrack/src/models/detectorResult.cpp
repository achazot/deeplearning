#include "detectorResult.hpp"

DetectorResult::DetectorResult ( cv::Rect position, int detclass, float score )
{
	m_position = position;
	m_detclass = detclass;
	m_score = score;
}

DetectorResult::~DetectorResult ( )
{

}

cv::Rect DetectorResult::position ( )
{
	return m_position;
}

int DetectorResult::detclass ( )
{
	return m_detclass;
}

float DetectorResult::score ( )
{
	return m_score;
}

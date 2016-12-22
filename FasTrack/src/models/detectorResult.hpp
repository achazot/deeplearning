#ifndef DETECTORRESULT_HPP
#define DETECTORRESULT_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class DetectorResult
{
private:
	cv::Rect m_position;
	int m_detclass;
	float m_score;

public:
	DetectorResult ( cv::Rect position, int detclass, float m_score );
	~DetectorResult ( );
	cv::Rect position ( );
	int detclass ( );
	float score ( );
};

#endif // DETECTOR_HPP

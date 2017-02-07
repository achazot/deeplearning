#ifndef IMGSOURCE_HPP
#define IMGSOURCE_HPP

#include <list>
#include <iostream>
#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class ImgSource
{
	public:
		ImgSource();
		ImgSource(int device);
		ImgSource(string path);
		~ImgSource();
		Mat operator >> (Mat &sink);
		bool empty();
	private:
		bool m_useCam;
		list<string> m_files;
		VideoCapture m_cap;
		bool m_error;
};



#endif

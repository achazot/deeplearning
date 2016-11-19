#include <stdio.h>  // for snprintf
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <dirent.h>
#include <fstream>
#include <list>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tools.hpp"
#include "controllers/detector.hpp"

using namespace caffe;
using namespace std;


int main(int argc, char* argv[])
{
	string net1_weights_file = "/home/dibi/sources/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel";
	string net2_weights_file = "/home/dibi/sources/py-faster-rcnn/data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel";

	string net1_model_file = "/home/dibi/sources/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt";
	string net2_model_file = "/home/dibi/sources/py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt";


	#ifdef CPU_ONLY
		Caffe::set_mode(Caffe::CPU);
	#else
		int GPUID=0;
		Caffe::SetDevice(GPUID);
		Caffe::set_mode(Caffe::GPU);
	#endif

	Detector det = Detector(net2_model_file, net2_weights_file);
	det.read_classes("networks/labels.txt");

	if (argc > 1 && strcmp(argv[1], "-s")==0)
	{
		string seqpath = argv[2];

		bool playing = false;

		list<string> files;
		DIR *dir;
		struct dirent *ent;
		if ( (dir = opendir ( seqpath.c_str() ) ) != NULL )
		{
			while ( ( ent = readdir (dir) ) != NULL )
			{
				string fname(ent->d_name);
				if ( fname.find( ".jpg" ) != string::npos )
					files.push_back( seqpath +"/"+ fname );
			}
		 	closedir (dir);
		}
		else
		{
		  perror ("");
		  return EXIT_FAILURE;
		}
		files.sort( tools::compare_nocase );
		for (string f : files)
		{
			cout <<  f << endl;
			det.Detection(f);
		}
	}
	else if (argc > 1 && strcmp(argv[1], "-c")==0)
	{
		cv::VideoCapture cap;
		if(!cap.open(0))
			return 0;
		bool running = true;

		cv::Mat src;
		int k = 0;
		while(running)
		{
			cap >> src;
			char kt[64];
			sprintf(kt, "cam%08d.jpg", k++);
			det.Detection(src, kt);
			if( cv::waitKey(1) == 1048603 ) running = false;
		}
	}
	else
	{
		cout << "ready." << endl;
		string word;
		while(getline(cin,word))
		{
			if (word.compare("exit") == 0 || word.compare("quit") == 0)
				break;
			else if (word.empty())
				continue;
			else
			{
				clock_t begin = clock();
				det.Detection(word);
				cout << "done in " << to_string(double(clock() - begin) / CLOCKS_PER_SEC) << endl;
			}
		}
	}

	return EXIT_SUCCESS;
}

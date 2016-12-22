#include <iostream>
#include <caffe/caffe.hpp>

#include "controllers/parser.hpp"
#include "controllers/detector.hpp"

using namespace std;


int main ( int agrc, char *argv[] )
{

  cout << "OpenCV version : " << CV_VERSION << endl;
  cout << "Major version : " << CV_MAJOR_VERSION << endl;
  cout << "Minor version : " << CV_MINOR_VERSION << endl;
  cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;
 
  if ( CV_MAJOR_VERSION < 3)
  {
      // Old OpenCV 2 code goes here. 
  } else
  {
      // New OpenCV 3 code goes here. 
  }	// init
	#ifdef CPU_ONLY
		Caffe::set_mode(Caffe::CPU);
	#else
		int GPUID=0;
		Caffe::SetDevice(GPUID);
		Caffe::set_mode(Caffe::GPU);
	#endif


	Parser parser = Parser();
	Detector detector = Detector();

	// read input
	cout << "\x1b[32mFasTrack\x1b[0m pre-alpha 1." << endl;
	cout << "Please enter a command" << endl << " > ";
	string line;
	while(getline(cin,line))
	{
		if (!line.empty())
		{
			bool cmdres = false;
			bool quit = false;

			switch(parser.parseCommand(line))
			{
				case Parser::CMD_LOADNET:
				{
					string caffemodel;
					string prototxt;
					string labels;
					cmdres = parser.getLoadnetArgs( caffemodel, prototxt, labels );
					if (cmdres)
					{
						cout << "cm: " << caffemodel << endl << "pt: " << prototxt << endl << "labels: " << labels << endl;
						detector.initialize(prototxt, caffemodel, labels);
					}
					break;
				}

				case Parser::CMD_DETECT:
				{
					string image_path;
					cmdres = parser.getDetectArgs(image_path);
					if (detector.initialized())
						detector.Detection(image_path);
					else
						cout << "please load a network first" << endl;

					break;
				}

				case Parser::CMD_QUIT:
					quit = true;
					break;

				case Parser::CMD_AMBIGUOUS:
					cout << "ambiguous command" << endl;
					break;

				case Parser::CMD_UNKNOWN:
				default:
					cout << "unknown command" << endl;
					break;
			}

			if (quit)
				break;

			if (!cmdres)
			{
				cout << "error" << endl;
			}
		}


		cout << " > ";
	}
}

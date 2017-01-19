#include <iostream>
#include <list>

#include "controllers/parser.hpp"
#include "controllers/detector.hpp"
#include "controllers/surfer.hpp"
#include "controllers/hogwarts.hpp"
#include "models/detectorResult.hpp"

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
	cout << "\x1b[32mFasTrack\x1b[0m pre-alpha 2." << endl;
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

				case Parser::CMD_TRACK:
				{
					list<string> files;
					cmdres = parser.getTrackArgs( files );
					if (cmdres)
					{

						cv::Mat img = cv::imread(files.front());
						cv::Rect objpos;


						if (detector.initialized())
						{
							detector.Detection(img);
							cout << to_string(detector.getResults().size()) << " matches found." << endl;
							if (detector.getResults().size() <= 0 ) break;
							objpos = detector.getResults().front().position();
						}
						else
						{
							// cout << "please load a network first" << endl;
							// break;
							// objpos = cv::Rect(468, 511, 60, 123); // crossing
							objpos = cv::Rect(186, 212, 54, 111); // basketball
							// objpos = cv::Rect(446, 173, 73, 205); // bottles
							// objpos = cv::Rect(512, 228, 79, 26); // birds1
						}


						// init tracker on first image
						Surfer surfer;
						surfer.initialize(img(objpos));
						Hogwarts tracker = Hogwarts( img, objpos );
						files.pop_front();

						cv::imshow("ref",img(objpos));
						rectangle(img, objpos, cv::Scalar(0,255,0), 2);
						cv::imshow("vis",img);
						cv::waitKey(10);

						int frNum = 1;
						for (string f : files)
						{
							img = cv::imread(f);

							// update tracker on new image
							objpos = tracker.update( img );
							float score = surfer.match(img, objpos);
							cout << "Score : " << score << endl;

							if (score < 0.1f)
							{
								//TODO: redetect
								surfer.initialize(img(objpos));
							}

							rectangle(img, objpos, cv::Scalar(0,255,0), 2);
							// putText(img, std::to_string(detector.getResults().front().detclass()),
							//	cvPoint(objpos.x, objpos.y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(128,128,128), 1, CV_AA);
							cv::imshow("vis",img);
							cv::waitKey(10);
						}
					}
					break;
				}

				case Parser::CMD_DETECT:
				{
					string image_path;
					cmdres = parser.getDetectArgs(image_path);
					if (detector.initialized())
					{
						cv::Mat cv_img = cv::imread(image_path);
						if(cv_img.empty())
						{
								std::cout<<"no such file"<<endl;
								break;
						}
						detector.Detection(image_path);
						cout << to_string(detector.getResults().size()) << " matches found." << endl;
						int n = 0;
						for (DetectorResult r : detector.getResults())
						{
							cout << to_string(r.position().x) << " " << to_string(r.position().y) << " " << to_string(r.position().width) << " " << to_string(r.position().height) << endl;
							rectangle(cv_img, r.position(), cv::Scalar(0,255,0), 1.5);
							putText(cv_img, std::to_string(n++) + ":" + std::to_string(r.detclass()),
								cvPoint(r.position().x, r.position().y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(128,128,128), 1, CV_AA);
							cv::imshow("vis",cv_img);
							cv::waitKey(1);
						}
					}
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
				cout << "syntax error" << endl;
			}
		}


		cout << " > ";
	}
}

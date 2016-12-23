#include <iostream>
#include <list>

#include "controllers/parser.hpp"
#include "controllers/detector.hpp"
#include "controllers/hogwarts.hpp"
#include "models/detectorResult.hpp"

using namespace std;


int main ( int agrc, char *argv[] )
{
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

				case Parser::CMD_TRACK:
				{
					// 468, 511, 60, 123

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
							objpos = detector.getResults().front().position();
						}
						else
						{
							//cout << "please load a network first" << endl;
							//break;
							//objpos = cv::Rect(468, 511, 60, 123); // crossing
							objpos = cv::Rect(186, 212, 54, 111); // basketball
						}


						// init tracker on first image
						Hogwarts tracker = Hogwarts( img, objpos );
						files.pop_front();

						rectangle(img, objpos, cv::Scalar(0,255,0), 2);
						cv::imshow("vis",img);
						cv::waitKey(10);

						for (string f : files)
						{
							img = cv::imread(f);
							// update tracker on new image
							objpos = tracker.update( img );

							rectangle(img, objpos, cv::Scalar(0,255,0), 2);
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
						for (DetectorResult r : detector.getResults())
						{
							//cout << to_string(r.position().x) << " " << to_string(r.position().y) << " " << to_string(r.position().width) << " " << to_string(r.position().height) << endl;
							rectangle(cv_img, r.position(), cv::Scalar(0,255,0), 2);
							cv::imshow("vis",cv_img);
							cv::waitKey(50);
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

#include <iostream>
#include <list>

#include "controllers/parser.hpp"
#include "controllers/detector.hpp"
#include "controllers/surfer.hpp"
#include "controllers/hogwarts.hpp"
#include "controllers/imgSource.hpp"
#include "models/detectorResult.hpp"

using namespace std;
using namespace cv;

/* ===================================================================================
 * ================================= MAIN ============================================
 * ===================================================================================
 *
 * Point d'entrée, crée une interface utilisateur en ligne de commande (CLI) et
 * instancie les objets responsables de la détection et du tracking
 */

int main ( int agrc, char *argv[] )
{
	// Verification de la version d'OpenCV
	cout << "OpenCV version : " << CV_VERSION << endl;

	// Instanciation des objets communs (parser d'entrées et détecteur)
	Parser parser = Parser();
	Detector detector = Detector();
	ImgSource imgSource;
	float confidence_treshold = 0.5;
	float nms_treshold = 0.3;
	int class_spec = 15;

	Mat cur_img;
	Mat vis_img;


	// Boucle de lecture de l'entrée CLI
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
				// Commande de chargement d'un réseau de neurones
				case Parser::CMD_LOADNET:
				{
					string caffemodel;
					string prototxt;
					string labels;
					cmdres = parser.getLoadnetArgs( caffemodel, prototxt, labels );
					if (cmdres)
					{
						cout << "cm: " << caffemodel << endl << "pt: " << prototxt << endl << "labels: " << labels << endl;
						detector.initialize(prototxt, caffemodel, labels, confidence_treshold, nms_treshold);
					}
					break;
				}

				// Commande de tracking
				case Parser::CMD_TRACK:
				{
					int obj_num;
					Rect force_pos;
					Rect objpos;
					int baseclass;
					cmdres = parser.getTrackArgs( obj_num, force_pos );

					cout << to_string(obj_num) << endl;

					if (!cmdres) break;

					if (obj_num >= 0)
					{
						if (obj_num >= detector.getResults().size())
						{
							cout << "No known object of index " << to_string(obj_num)<< " from detection." << endl;
							break;
						}
						objpos = detector.getResults().at(obj_num).position();
						baseclass = detector.getResults().at(obj_num).detclass();
					}
					else
					{
						objpos = force_pos;
						baseclass = class_spec; //TODO: spec track parser
					}

					// objpos = cv::Rect(468, 511, 60, 123); // crossing
					// objpos = cv::Rect(186, 212, 54, 111); // basketball
					// objpos = cv::Rect(446, 173, 73, 205); // bottles 446 173 73 205
					// objpos = cv::Rect(512, 228, 79, 26); // birds1
					// pedestrian 326 417 13 37

					// Initialise le tracker sur la première image
					Surfer surfer(400);
					surfer.initialize(cur_img(objpos));
					Hogwarts tracker = Hogwarts( cur_img, objpos );

					vis_img = cur_img.clone();

					rectangle(vis_img, objpos, cv::Scalar(0,255,0), 2);
					cv::imshow("vis", vis_img);
					cv::waitKey(10);

					// ln networks/ZF_test

					int frNum = 1;
					bool running = true;
					int surf_failing = 0;
					while (running && !imgSource.empty())
					{

						imgSource >> cur_img;
						vis_img = cur_img.clone();

						// update tracker on new image
						objpos = tracker.update( cur_img );
						float score = surfer.match(cur_img, objpos);
						cout << "Score : " << score << endl;

						if (score == -1) surf_failing ++;
						else surf_failing = 0;

						if ((score < 0.2f && score != -1) || (surf_failing >= 5))
						{
							frNum = 1;
							vector<DetectorResult> resultSet;


							for (int l = 0; l < 5; l++)
							{
								detector.setConfTresh(confidence_treshold/(float)l);
								detector.Detection(cur_img);
								for (DetectorResult res : detector.getResults())
									if (res.detclass() == baseclass) resultSet.push_back(res);

								cout << to_string(resultSet.size()) << " objects found." << endl;
								if (resultSet.size()!=0) break;
							}

							cout << "position1: " << to_string(objpos.x) << " " << to_string(objpos.y)
								<< " " << to_string(objpos.width) << " " << to_string(objpos.height) << endl;
							Rect last_objpos = objpos;


							float best_score = -1;
							float best_dist = 20000;
							int n=0;
							for (DetectorResult res : resultSet)
							{
								int n = 0;
								if (res.detclass() == baseclass)
								{
									 char sc[8];
									 sprintf(sc, "%03f", res.score());
									 cout << to_string(n) << ": " << detector.getClass(res.detclass()) << " (" << sc << ") position: "
									 	<< to_string(res.position().x) << " " << to_string(res.position().y)
									 	<< " " << to_string(res.position().width) << " " << to_string(res.position().height) << endl;
									 rectangle(vis_img, res.position(), cv::Scalar(0,255,0), 1.5);
									 //TODO: mettre rectangle transparent dessous
									 putText(vis_img, std::to_string(n) + ":" + detector.getClass(res.detclass()),
									 	cvPoint(res.position().x, res.position().y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,100), 1, CV_AA);
									 imshow("vis",vis_img);
									n++;

									float resScore = surfer.match(cur_img, res.position());

									rectangle(vis_img, res.position(), cv::Scalar(255,0,0), 1);
									putText(vis_img, to_string(n++) + ":" + to_string(resScore),
										cvPoint(res.position().x, res.position().y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,100), 1, CV_AA);

									if (resScore <= 0 && best_score == -1)
									{
										float x1 = objpos.x;
										float y1 = objpos.y;
										float x2 = res.position().x;
										float y2 = res.position().y;
										float x3 = objpos.x + objpos.width;
										float y3 = objpos.y + objpos.height;
										float x4 = res.position().x + res.position().width;
										float y4 = res.position().y + res.position().height;
										float dist = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))+sqrt((x3-x4)*(x3-x4)+(y3-y4)*(y3-y4));
										if (best_dist > dist)
										{
											best_dist = dist;
											objpos = res.position();
										}
									}

									if (resScore > best_score)
									{
										objpos = res.position();
										best_score = resScore;
									}
								}
							}
							cout << "best score:" << to_string(best_score) << endl;
							cout << "position3: " << to_string(objpos.x) << " " << to_string(objpos.y)
								<< " " << to_string(objpos.width) << " " << to_string(objpos.height) << endl;

							if (objpos.x == -1 && objpos.y == -1 && objpos.width == -1 && objpos.height == -1)
							{
								cout << "Object potentially lost." << endl;
								//running = false;
								objpos = last_objpos;
								break;
							}
							else
							{
								surfer.initialize(cur_img(objpos));
								tracker = Hogwarts( cur_img, objpos );
							}
						}
						else if (score > 0.6f)
						{
							if (objpos.x > vis_img.size().width) objpos.x = vis_img.size().width - 1;
							if (objpos.y > vis_img.size().height) objpos.y = vis_img.size().height - 1;

							if (objpos.x < 0) objpos.x = 0;
							if (objpos.y < 0) objpos.y = 0;

							if (objpos.x + objpos.width > vis_img.size().width) objpos.width = vis_img.size().width - objpos.x;
							if (objpos.y + objpos.height > vis_img.size().height) objpos.height = vis_img.size().height - objpos.y;
							surfer.initialize(cur_img(objpos));
						}

						if (objpos.x > vis_img.size().width) objpos.x = vis_img.size().width - 1;
						if (objpos.y > vis_img.size().height) objpos.y = vis_img.size().height - 1;

						if (objpos.x < 0) objpos.x = 0;
						if (objpos.y < 0) objpos.y = 0;

						if (objpos.x + objpos.width > vis_img.size().width) objpos.width = vis_img.size().width - objpos.x;
						if (objpos.y + objpos.height > vis_img.size().height) objpos.height = vis_img.size().height - objpos.y;

						rectangle(vis_img, objpos, cv::Scalar(0,255,0), 2);
						// putText(img, std::to_string(detector.getResults().front().detclass()),
						//	cvPoint(objpos.x, objpos.y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(128,128,128), 1, CV_AA);
						cv::imshow("vis",vis_img);
						cv::waitKey(10);

						detector.setConfTresh(confidence_treshold);
					}
					break;
				}

				// Commande de détection d'une seule image de la source
				case Parser::CMD_DETECT:
				{
					cmdres = parser.getDetectArgs();

					if (!detector.initialized())
					{
						cout << "Please load a network first" << endl;
						break;
					}
					if (imgSource.empty())
					{
						cout << "Please select a source first" << endl;
						break;
					}

					detector.Detection(cur_img);
					vis_img = cur_img.clone();
					cout << to_string(detector.getResults().size()) << " matches found." << endl;
					int n = 0;
					for (DetectorResult r : detector.getResults())
					{
						char sc[8];
						sprintf(sc, "%03f", r.score());
						cout << to_string(n) << ": " << detector.getClass(r.detclass()) << " (" << sc << ") position: "
							<< to_string(r.position().x) << " " << to_string(r.position().y)
							<< " " << to_string(r.position().width) << " " << to_string(r.position().height) << endl;
						rectangle(vis_img, r.position(), cv::Scalar(0,255,0), 1.5);
						//TODO: mettre rectangle transparent dessous
						putText(vis_img, std::to_string(n++) + ":" + detector.getClass(r.detclass()),
							cvPoint(r.position().x, r.position().y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,100), 1, CV_AA);
						imshow("vis",vis_img);
						waitKey(100);
					}

					break;
				}


				// Commande de sélection de la source (dossier, webcam ou video)
				case Parser::CMD_SOURCE:
				{
					string path;
					int device;
					cmdres = parser.getSourceArgs( path, device );
					if (!cmdres) break;

					if (device >= 0) imgSource = ImgSource(device);
					else imgSource = ImgSource(path);

					if (imgSource.empty()) cout << "Source is empty." << endl;

					imgSource >> cur_img;

					break;
				}

				case Parser::CMD_FEED:
				{
					cmdres = true;
					for (int i = 0; i< 10; i++)
						imgSource >> cur_img;
					break;
				}

				// définition des paramètres
				case Parser::CMD_SET:
				{
					cmdres = parser.getSetArgs(confidence_treshold, nms_treshold, class_spec);
					detector.setConfTresh(confidence_treshold);
					detector.setNmsTresh(nms_treshold);
					break;
				}

				// Autres
				case Parser::CMD_QUIT:
					quit = true;
					break;

				case Parser::CMD_AMBIGUOUS:
					cout << "ambiguous command" << endl;
					cmdres = true;
					break;

				case Parser::CMD_UNKNOWN:
				default:
					cmdres = true;
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

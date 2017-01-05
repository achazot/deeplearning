#include "surfer.hpp"



Surfer::Surfer(){ }
Surfer::Surfer(int hessian){ hessian_value = hessian; }

void Surfer::initialize(Mat& object)
{
	Mat object_gray;
	cvtColor(object, object_gray, CV_RGB2GRAY);
	Ptr<Feature2D> surf = SURF::create(hessian_value);	
	surf->detectAndCompute(object, Mat(), obj_keypoints, obj_descriptors); 
}

float Surfer::match(Mat& scene, Rect position)
{
	Mat scene_gray;
	cvtColor(scene, scene_gray, CV_RGB2GRAY);
	Ptr<Feature2D> surf = SURF::create(hessian_value);	
	surf->detectAndCompute(scene, Mat(), scene_keypoints, scene_descriptors);
	
	BFMatcher matcher; 
	std::vector<DMatch> matches;
	matcher.match(obj_descriptors, scene_descriptors, matches); 

	std::sort(matches.begin(), matches.end());
	std::vector< DMatch > good_matches; 
	for(int i = 0; i < (matches.size() > 5 ? 5 : matches.size()); i ++)
		good_matches.push_back( matches[i] );

    Rect scene_subwindow; 
    scene_subwindow.x = MAX(position.x - position.width / 2, 0);
    scene_subwindow.y = MAX(position.y - position.height / 2, 0);
    scene_subwindow.width = position.width * 2;
    scene_subwindow.height = position.height * 2;

    float in_subwindow = 0;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        if( scene_subwindow.x <= scene_keypoints[ good_matches[i].trainIdx ].pt.x
            && scene_keypoints[ good_matches[i].trainIdx ].pt.x <= scene_subwindow.x + scene_subwindow.width
            && scene_subwindow.y <= scene_keypoints[ good_matches[i].trainIdx ].pt.y
            && scene_keypoints[ good_matches[i].trainIdx ].pt.y <= scene_subwindow.y + scene_subwindow.height )
                in_subwindow++;        
    }
    
    return in_subwindow / good_matches.size();
}
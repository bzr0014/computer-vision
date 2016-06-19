#ifndef OBJECT_COUNTER_H
#define OBJECT_COUNTER_H

#include "cv.h"
#include "highgui.h"
#include <ml.h>
#include <math.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

class objectCounter {
private:
	Mat image, gray, markers, output, training_image, wshed;
	int count;
	//vector<KeyPoint> train_key_points, output_key_points;
	vector<KeyPoint> train_image_keypoints;
	Mat train_image_desc;
	int numImages = 0;
	int hasMarkers = false;
public:
	Mat mask;
	vector<Mat> separate_images;
	vector<vector<Point> > contours;
	vector<Vec4i> main_hierarchy;
	objectCounter(Mat); //constructor
	void get_markers(); //function to get markers for watershed segmentation
	void get_markers2();
	int count_objects(); //function to implement watershed segmentation and count catchment basins
	void detect_objects();
	void train_image(Mat train_image);
	vector<DMatch> match_images(Mat query_image, Mat mask);
	
};

#endif

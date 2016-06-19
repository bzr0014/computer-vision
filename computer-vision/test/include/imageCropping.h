#ifndef IMAGE_CROPPING_H
#define IMAGE_CROPPING_H

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

class ImageCropping {
public:
	// Global variables
	// Flags updated according to left mouse button activity
	bool ldown = false, lup = false;
	// Original image
	Mat img;
	// Starting and ending points of the user's selection
	Point corner1, corner2;
	// ROI
	Rect box;

	Mat crop;
	// Callback function for mouse events

	ImageCropping();
	void mouse_callback(int event, int x, int y, int, void *);
};

#endif

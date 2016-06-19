#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

#ifndef _TESTTEMP_H_
#define _TESTTEMP_H_

class utilities
{
public:
	utilities();
	static double calc_area(vector<Point> points);
	static double calc_area(Point* points, int size);
	static vector<Point> point2f_to_point(vector<Point2f> src);
	static vector<Rect> calc_bounding_rect(vector<vector<Point> > contours);
	static vector<Point> rect_to_contour(Rect rect);
	static vector<double> calc_statistics(vector<double> values);
	static vector<double> normalize_values(vector<double> values);
	static vector<double> normalize_values(vector<double> values, vector<double> statistics);
	static vector<Point> rotatedRect_to_contour(RotatedRect rect);
	static vector<DMatch> match_images(Mat query_image, Mat train_image, Mat mask);
};
#endif

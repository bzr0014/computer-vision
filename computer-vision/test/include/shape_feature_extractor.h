#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <math.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include<hash_map>
#include <map>
#include <cassert>

using namespace cv;

#ifndef SHAPE_FEATURE
#define SHAPE_FEATURE
class shape_feature_extractor {
public:
	int size;
	Mat image;
	//major contours
	vector<vector<Point> > contours, good_contours,
		approx, min_rects_contour, hulls;
	vector<Rect> bounding_rects;
	vector<RotatedRect> min_rects;
	vector<Point> centers;
	vector<double> radii;
	//different shape areas
	double image_area;
	vector<double> contours_area;
	vector<double> good_contours_areas,
		bounding_rects_areas,
		approx_areas,
		min_rects_areas,
		hulls_areas,
		circles_areas;
	//features
	vector<vector<double> > features;
	vector<string> feature_labels;
	std::map<string, int> labels;

	shape_feature_extractor(Mat _imge);
	shape_feature_extractor();
	void getImage(Mat _img);
	void get_contours();
	void get_contours(int i);
	void extract_features();
	void match_shapes(shape_feature_extractor that);

private:
	void filter_contours();
	void get_shapes();
	void get_shape_areas();
	void add_to_features(vector<double> feature, string label);
	vector<double> calc_hull_to_min_rect();
	vector<double> calc_min_rect_length_to_width();
	vector<double> calc_rectagularity();
	vector<vector<double> > calc_hu_moments();
	vector<double> calc_elongatedness();
	
};
#endif

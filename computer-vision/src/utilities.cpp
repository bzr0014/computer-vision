#include <../include/utilities.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

template<class T> vector<T> array_to_vector(T* src, int src_size) {
	vector<T> dst;
	for (int i = 0; i < src_size; i++) {
		dst.push_back(src[i]);
	}
	return dst;
}
utilities::utilities(){}
double utilities::calc_area(vector<Point> points) {
	double area = 0;
	int size = points.size();
	for (int i = 0; i < size; i++) {
		area = area + .5 * (points[i].x * points[(i + 1) % size].y - points[(i + 1) % size].x * points[i].y);
	}
	return area >= 0 ? area : -1 * area;
}
double utilities::calc_area(Point* points, int size) {
	double area = 0;
	for (int i = 0; i < size; i++) {
		area = area + .5 * (points[i].x * points[(i + 1) % size].y - points[(i + 1) % size].x * points[i].y);
	}
	return area >= 0 ? area : -1 * area;
}
vector<Rect> utilities::calc_bounding_rect(vector<vector<Point> > contours) {
	int size = contours.size();
	vector<Rect> bounding_rect(size);
	vector<vector<Point> > approx_poly(size);
	for (int i = 0; i < size; i++) {
		approxPolyDP(Mat(contours[i]), approx_poly[i], 3, true);
		bounding_rect[i] = boundingRect(Mat(approx_poly[i]));
	}
	return bounding_rect;
}
vector<Point> utilities::rect_to_contour(Rect rect) {
	Point p1, p2, p3, p4;
	p1 = Point(rect.x, rect.y);
	p2 = Point(rect.x + rect.width, rect.y);
	p3 = Point(rect.x + rect.width, rect.y + rect.height);
	p4 = Point(rect.x, rect.y + rect.height);
	Point mardas[] = { p1, p2, p3, p4 };
	vector<Point> ret;
	for (int i = 0; i < 4; i++) {
		ret.push_back(mardas[i]);
	}
	return ret;
}
vector<double> utilities::calc_statistics(vector<double> values) {
	if (!values.size()) return vector<double>({ 0, 0 });
	double mean = 0;
	for (int i = 0; i < values.size(); i++) {
		mean += values[i];
	}
	mean = mean / values.size();

	double variance = 0;

	for (int i = 0; i < values.size(); i++) {
		variance += pow(values[i] - mean, 2);
	}
	variance = variance / values.size();
	double stdev = pow(variance, .5);
	double mardas[] = { mean, stdev };
	vector<double> statistics;
	for (int i = 0; i < 2; i++) {
		statistics.push_back(mardas[i]);
	}


	return statistics;

}
vector<double> utilities::normalize_values(vector<double> values) {
	vector<double> statistics = calc_statistics(values);
	for (int i = 0; i < values.size(); i++) {
		values[i] = (values[i] - statistics[0]) / statistics[1];
	}
	return values;
}
vector<double> utilities::normalize_values(vector<double> values, vector<double> statistics) {
	for (int i = 0; i < values.size(); i++) {
		values[i] = (values[i] - statistics[0]) / statistics[1];
	}
	return values;
}
vector<Point> utilities::point2f_to_point(vector<Point2f> src) {
	vector<Point> dst;
	for (int i = 0; i < src.size(); i++){
		int x = (int)src[i].x;
		int y = (int)src[i].y;
		dst.push_back(Point(x, y));
	}
	return dst;
}
vector<Point> utilities::rotatedRect_to_contour(RotatedRect rect) {
	// min bounding rectangles
	Point2f temp_points[4];
	rect.points(temp_points);
	vector<Point2f> fl_points = array_to_vector(temp_points, 4);
	vector<Point> min_rect_pts = utilities::point2f_to_point(fl_points);
	return min_rect_pts;
}
vector<DMatch> utilities::match_images(Mat query_image, Mat training_image, Mat mask) {
	cv::SiftFeatureDetector detector(300);
	SiftDescriptorExtractor extractor;
	vector<KeyPoint> train_image_keypoints;
	detector.detect(training_image, train_image_keypoints);
	
	Mat train_image_desc;
	extractor.compute(training_image, train_image_keypoints, train_image_desc);
	vector<KeyPoint> query_image_keypoints;
	detector.detect(query_image, query_image_keypoints, mask);
	Mat query_image_desc;
	extractor.compute(query_image, query_image_keypoints, query_image_desc);

	FlannBasedMatcher matcher;
	vector<vector<DMatch> > matches;
	matcher.knnMatch(query_image_desc, train_image_desc, matches, 2);

	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[0][0].distance < .5 * matches[0][1].distance)
			good_matches.push_back(matches[0][0]);
	}

	Mat img_matches;
	//if (good_matches.size() > 0) {
	drawMatches(query_image, query_image_keypoints, training_image, train_image_keypoints, good_matches, img_matches);
	//}
	imshow("matches", img_matches);
	waitKey(0);
	return good_matches;
}

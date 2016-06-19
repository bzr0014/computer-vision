#include <iostream>
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv/ml.h>
#include <math.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <../include/object_counter.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

objectCounter::objectCounter(Mat _image) {
	image = _image.clone();
	cvtColor(image, gray, CV_BGR2GRAY);
	imshow("image", image);
}
void objectCounter::get_markers() {
	// equalize histogram of image to improve contrast
	Mat im_e; equalizeHist(gray, im_e);
	//imshow("im_e", im_e);
	// dilate to remove small black spots
	Mat strel =
	getStructuringElement(MORPH_ELLIPSE, Size(9,
	9));
	Mat im_d; dilate(im_e, im_d, strel);
	//imshow("im_d", im_d);
	// open and close to highlight objects
	strel =
	getStructuringElement(MORPH_ELLIPSE, Size(19,
	19));
	Mat im_oc; morphologyEx(im_d, im_oc,
	MORPH_OPEN, strel);
	morphologyEx(im_oc, im_oc, MORPH_CLOSE,
	strel);
	//imshow("im_oc", im_oc);
	// adaptive threshold to create binary	image
	Mat th_a; adaptiveThreshold(im_oc, th_a,
	255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
	105, 0);
	//imshow("th_a", th_a);
	// erode binary image twice to separate	regions
	Mat th_e; erode(th_a, th_e, strel,
	Point(-1, -1), 2);
	//imshow("th_e", th_e);
	vector<vector<Point> > c, contours;
	vector<Vec4i> hierarchy;
	findContours(th_e, c, hierarchy,
	CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	// remove very small contours
	for(int idx = 0; idx >= 0; idx =
	hierarchy[idx][0])
	if(contourArea(c[idx]) > 20)
	contours.push_back(c[idx]);
	cout << "Extracted " << contours.size() <<
	" contours" << endl;
	count = contours.size();
	markers.create(image.rows, image.cols,
	CV_32SC1);
	for(int idx = 0; idx < contours.size();
	idx++)
	drawContours(markers, contours, idx,
	Scalar::all(idx + 1), -1, 8);
}
void objectCounter::get_markers2() {
	if (hasMarkers) return;
	// equalize histogram of image to improve contrast
	Mat im_e; 
	equalizeHist(gray, im_e);
	//imshow("im_e", im_e);
	// dilate to remove small black spots
	Mat strel =
		getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	Mat im_d; dilate(im_e, im_d, strel);
	imshow("im_d", im_d);
	// open and close to highlight foreground objects
	strel =
		getStructuringElement(MORPH_ELLIPSE, Size(15,
		15));
	Mat im_oc;
	morphologyEx(im_d, im_oc, MORPH_OPEN, strel);
	morphologyEx(im_oc, im_oc, MORPH_CLOSE,	strel);
	imshow("im_oc", im_oc);
	// adaptive threshold to create binary image
	Mat th_a;
	adaptiveThreshold(im_oc, th_a, 255,
			ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
			105, 0);
	imshow("th_a", th_a);
	// erode binary image twice to separate	regions
	Mat canny_image;
	blur(im_e, im_e, Size(5, 5));
	Canny(im_e, canny_image, 100, 50);
	strel =
		getStructuringElement(MORPH_RECT, Size(7,
		7));
	erode(canny_image, canny_image, strel);
	erode(canny_image, canny_image, strel);
	morphologyEx(canny_image, canny_image, MORPH_CLOSE, strel);
	canny_image = Scalar::all(255) - canny_image;
	//morphologyEx(canny_image, canny_image, MORPH_CLOSE, strel);
	//imshow("canny", canny_image);
	//strel =
	//	getStructuringElement(MORPH_ELLIPSE, Size(9,
	//	9));
	Mat th_e;
	erode(th_a, th_e, strel,
		Point(-1, -1), 2);
	//imshow("th_e", th_e);
	vector<Vec4i> hierarchy;
	th_a = min(canny_image, th_a);
	//imshow("th_a", th_a);
	vector<vector<Point> > c;
	findContours(canny_image, c, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	// remove very small contours
	for (int idx = 0; idx >= 0; idx =
		hierarchy[idx][0])
		if (contourArea(c[idx]) > 500)
			contours.push_back(c[idx]);
	main_hierarchy = hierarchy;
	std::cout << "Extracted " << contours.size() <<
		" contours" << endl;
	count = contours.size();
	markers.create(image.rows, image.cols,
		CV_32SC1);
	for (int idx = 0; idx < contours.size();
		idx++)
		drawContours(markers, contours, idx,
		Scalar::all(idx + 1), -1, 8);
	hasMarkers = true;
}

int objectCounter::count_objects() {
	if (numImages) {
		imshow("Segmentation", wshed);
		return numImages;
	}
	watershed(image, markers);
	// colors generated randomly to make the output look pretty
	vector<Vec3b> colorTab;
	for (int i = 0; i < count; i++) {
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);
		colorTab.push_back(Vec3b((uchar)b,
			(uchar)g, (uchar)r));
	}

	vector<uchar> colorTab_sequence;
	for (int i = 0; i < count; i++) {
		colorTab_sequence.push_back(uchar(i + 1));
	}
	// watershed output image
	Mat raghas(markers.size(), CV_8UC3);
	wshed = raghas;
	Mat wshed_sequence(markers.size(), CV_8UC1);
	// paint the watershed output image
	for (int i = 0; i < markers.rows; i++)
		for (int j = 0; j < markers.cols; j++) {
		int index = markers.at<int>(i, j);
		if (index == -1)
			wshed.at<Vec3b>(i, j) = Vec3b(255,
			255, 255);
		else if (index <= 0 || index > count)
			wshed.at<Vec3b>(i, j) = Vec3b(0, 0,
			0);
		else {
			wshed.at<Vec3b>(i, j) =
				colorTab[index - 1];
			wshed_sequence.at<uchar>(i, j) =
				colorTab_sequence[index - 1];
		}
		}

	for (int i = 0; i < count; i++) {
		separate_images.push_back(Mat::zeros(image.size(), image.type()));
	}

	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++) {
			int index = markers.at<int>(i, j);
			if (index >= 0 && index < count) {
				separate_images.at(index).at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
			}
		}
	}

	// superimpose the watershed image with 50%	transparence on the grayscale original image
	Mat imgGray; cvtColor(gray, imgGray,
		CV_GRAY2BGR);
	wshed = wshed*0.5 + imgGray*0.5;
	imshow("Segmentation", wshed);
	numImages = count;
	return count;
}
void objectCounter::train_image(Mat train_image_in) {
	training_image = train_image_in.clone();
	SiftFeatureDetector detector(300);
	detector.detect(training_image, train_image_keypoints);
	SiftDescriptorExtractor extractor;
	extractor.compute(training_image, train_image_keypoints, train_image_desc);
}
vector<DMatch> objectCounter::match_images(Mat query_image, Mat mask) {
	SiftFeatureDetector detector(300);
	vector<KeyPoint> query_image_keypoints;
	detector.detect(query_image, query_image_keypoints, mask);

	SiftDescriptorExtractor extractor;
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
	if (good_matches.size() > 0) {
		drawMatches(query_image, query_image_keypoints, training_image, train_image_keypoints, good_matches, img_matches);
		imshow("matches", img_matches);
	}
	return good_matches;
}

// Program to crop images using GUI mouse callbacks
// Author: Samarth Manoj Brahmbhatt, University of Pennsylvania

#include <iostream>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../include/categorizer.h"
#include "../include/shape_feature_extractor.h"
#include "../include/object_counter.h"
#include "../include/utilities.h"

using namespace std;
using namespace cv;

RNG rng(255);
double REFERENCE_LENGTH = 10.01;;

int main() {
	Mat img = imread("/home/behnam/Downloads/M001 Screen Shot.png");
	Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
	
    //find only green and purple pixels in the picture and get a mask of them
    for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (
				( img.at<Vec3b>(i, j)[0] < 100 and img.at<Vec3b>(i, j)[1] > 150 and img.at<Vec3b>(i, j)[2] < 100)
				or
				( img.at<Vec3b>(i, j)[0] > 90 and img.at<Vec3b>(i, j)[1] < 50 and img.at<Vec3b>(i, j)[2] > 100)
				) {
				mask.at<uchar>(i, j) = 255;
			}
		}
	}

    //allContours stores all the contours
	vector<vector<Point>> allContours, contours;
	vector<Vec4i> hierarchy;
	findContours(mask, allContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
    //just choose the contours that have more than 50 points and do not have child contours
	for (int i = 0; i < hierarchy.size(); i++) {
		if (hierarchy[i][2] > 0 or allContours[i].size() < 50) {
			continue;
		}
		contours.push_back(allContours[i]);
	}

    //draw the contours
	Mat drawing = Mat::zeros( img.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		//Scalar color = Scalar(255);
		drawContours( drawing, contours, i, color);
   	}
   	
    //find the center of the contours
   	vector<Point> centers;
   	for (int i = 0; i < contours.size(); i++) {
   		int sum_x = 0;
   		int sum_y = 0;
   		for (int j = 0; j < contours[i].size(); j++) {
	   		sum_x = sum_x + contours[i][j].x;
	   		sum_y = sum_y + contours[i][j].y;
	   	}
	   	sum_x = sum_x / contours[i].size();
	   	sum_y = sum_y / contours[i].size();
		circle(img, Point(sum_x, sum_y), 2, Scalar(100, 0, 100), 3, 8);
		centers.push_back(Point(sum_x, sum_y));
	}



    //get the lengh of the reference line:

	Mat img_canny;
	Mat mardas = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Canny(img, img_canny, 1000, 200);
	Mat str_el = getStructuringElement(CV_SHAPE_RECT, Size(3, 1));
	morphologyEx(img_canny, img_canny, MORPH_OPEN, str_el);
	vector<Vec4i> lines;
    HoughLinesP( img_canny, lines, 1, CV_PI/180, 30, 30, 20 );
    Mat img_clone = img.clone();
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( img_clone, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }

    // the previous step detected several lines, here we choose the longers one
    double longestLineLength = 0;
    int longestLineIndex = 0;
    for(int i = 0; i < lines.size(); i++) {
    	double xDist = lines[i][0] - lines[i][2];
    	double yDist = lines[i][1] - lines[i][3];
    	double length = (int) sqrt(pow(xDist, 2) + pow(yDist, 2));
    	if (longestLineLength < length) {
    		longestLineLength = length;
    		longestLineIndex = i;
    	}
    }


    //sort the centers based on x
    //this is used to detect the middle point
    for (int i = 1; i < centers.size(); i++) {
    	Point key = centers[i];
    	int j = i - 1;
    	while (j >= 0 and centers[j].x > key.x) {
    		centers[j + 1] = centers[j];
    		centers[j] = key;
    		j--;
    	}
    }

    int centerPointIndex = centers.size() / 2;
    Point centerPoint = centers[centerPointIndex];
    //set the middle point to (0, 0) so that it is ommited when we sort centers based on y later
    centers[centerPointIndex] = Point(0, 0);
    circle(img, centerPoint, 10, Scalar(100, 0, 100), 3, 8);

    //sort centers base on y
    for (int i = 1; i < centers.size(); i++) {
    	Point key = centers[i];
    	int j = i - 1;
    	while (j >= 0 and centers[j].y > key.y) {
    		centers[j + 1] = centers[j];
    		centers[j] = key;
    		j--;
    	}
    }

    
    //the adjacent points are the ones that are at almost the same height in the image
    //note that since we set the center point to (0, 0), then the loop starts from 1st index rather than index 0
    int size = centers.size();
    vector<vector<Point>> correspondingPoints;
    for (int i = 1; i < size; i=i+2) {
    	line(img, centers[i], centers[i+1], Scalar(128, 128, 255));
    	//store adjacent points in a separate vector for ease of use
        vector<Point> temp;
    	temp.push_back(centers[i]);
    	temp.push_back(centers[i+1]);
    	correspondingPoints.push_back(temp);
    }

    //find the length of each line and print them above the line
    for (int i = 0; i < correspondingPoints.size(); i++) {
    	double xDist = correspondingPoints[i][0].x - correspondingPoints[i][1].x;
    	double yDist = correspondingPoints[i][0].y - correspondingPoints[i][1].y;
    	double length = (int) sqrt(pow(xDist, 2) + pow(yDist, 2));
    	double actualLength = length / longestLineLength * REFERENCE_LENGTH;
    	Point textOrg = Point((correspondingPoints[i][0].x + correspondingPoints[i][1].x) / 2, (correspondingPoints[i][0].y + correspondingPoints[i][1].y) / 2 - 10);
        char text[15];
        sprintf(text, "%3.2f cm", actualLength);
    	putText(img, text, textOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, .6, Scalar(0, 00, 250), 2, 8);
    }
    imshow("canny", img_canny);
	imshow("lines", mardas);
	imshow("drawing", drawing);
	imshow("img", img);
    imshow("clone", img_clone);
	waitKey();

	return 0;
}

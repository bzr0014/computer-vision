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

int min_r = 0;
int min_g = 0;
int min_b = 0;
int max_r = 255;
int max_g = 255;
int max_b = 255;


Mat img = imread("/home/behnam/Downloads/MRI pics/M003 Traced MRI MS.png");
Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
void mardas() {
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    //find only green and purple pixels in the picture and get a mask of them
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (
                //green lines
                //( img.at<Vec3b>(i, j)[0] < 200 and img.at<Vec3b>(i, j)[1] > 200 and img.at<Vec3b>(i, j)[2] < 200)
                //or
                //purple lines
                //( img.at<Vec3b>(i, j)[0] > 90 and img.at<Vec3b>(i, j)[1] < 50 and img.at<Vec3b>(i, j)[2] > 100)
                //or
                //yellow lines
                ( img.at<Vec3b>(i, j)[0] > min_r and img.at<Vec3b>(i, j)[0] < max_r 
                    and img.at<Vec3b>(i, j)[1] > min_g and img.at<Vec3b>(i, j)[1] < max_g 
                    and img.at<Vec3b>(i, j)[2] > min_b and img.at<Vec3b>(i, j)[2] < max_b)
                ) {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }
    imshow("mask", mask);
    printf("%d\n", min_b);
}
void on_trackbar_r_min(int, void*) {
    mardas();
}
void on_trackbar_r_max(int, void*) {
    mardas();
}
void on_trackbar_g_min(int, void*) {
    mardas();
}
void on_trackbar_g_max(int, void*) {
    mardas();
}
void on_trackbar_b_min(int, void*) {
    mardas();
}
void on_trackbar_b_max(int, void*) {
    mardas();
}

int main() {
	//Mat img = imread("/home/behnam/Downloads/M001 Screen Shot.png");
    imshow("mask", mask);
    createTrackbar( "min red", "mask", &min_r, 255, on_trackbar_r_min);
    createTrackbar( "max red", "mask", &max_r, 255, on_trackbar_r_max);
    createTrackbar( "min green", "mask", &min_g, 255, on_trackbar_g_min);
    createTrackbar( "max green", "mask", &max_g, 255, on_trackbar_g_max);
    createTrackbar( "min blue", "mask", &min_b, 255, on_trackbar_b_min);
    createTrackbar( "max blue", "mask", &max_b, 255, on_trackbar_b_max);
    
    waitKey(0);
	return 0;
}

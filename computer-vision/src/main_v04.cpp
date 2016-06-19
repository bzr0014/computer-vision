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

using namespace cv;
using namespace std;

static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

static double sideRatio(Point p1, Point p2, Point p3) {
    double s1 = pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2);
    s1 = sqrt(s1);

    double s2 = pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2);
    s2 = sqrt(s2);
    if (s2 > s1) return s2 / s1;
    else return s1 / s2;
}


int squareType = 1;
int redRectangleType = 2;
int grayRectangleType = 3;
static bool checkSquare(vector<Point> approx, int type) {
    switch(type) {
    case (1) :
        if( approx.size() == 4 &&
            fabs(contourArea(Mat(approx))) > 1000 &&
            isContourConvex(Mat(approx)) )
        {
            double maxCosine = 0;

            for( int j = 2; j < 5; j++ )
            {
                // find the maximum cosine of the angle between joint edges
                double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                maxCosine = MAX(maxCosine, cosine);
            }
            //detect squares, not rectangles
            double side_ratio = sideRatio(approx[0], approx[1], approx[2]);

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.1 and side_ratio < 1.2)
               return true;
        }
        return false;
    case(3) :
        if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
        {
            double maxCosine = 0;

            for( int j = 2; j < 5; j++ )
            {
                // find the maximum cosine of the angle between joint edges
                double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                maxCosine = MAX(maxCosine, cosine);
            }
            //detect squares, not rectangles
            double side_ratio = sideRatio(approx[0], approx[1], approx[2]);

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.1 and side_ratio > 2 and side_ratio < 2.2)
                return true;
        }
        return false;
    case(2) :
        if( approx.size() == 4 &&
            fabs(contourArea(Mat(approx))) > 1000 &&
            isContourConvex(Mat(approx)) )
        {
            double maxCosine = 0;

            for( int j = 2; j < 5; j++ )
            {
                // find the maximum cosine of the angle between joint edges
                double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                maxCosine = MAX(maxCosine, cosine);
            }
            //detect squares, not rectangles
            double side_ratio = sideRatio(approx[0], approx[1], approx[2]);

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.1 and side_ratio < 1.5 and side_ratio > 1.3)
                return true;
        }
        return false;
    }
}

double findDist(Point p1, Point p2) {
    double dist = pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2);
    return sqrt(dist);
}

double distSquares(Rect a, Rect b) {
    int ax = a.x + a.width / 2;
    int ay = a.y + a.height /2;
    int bx = b.x + b.width / 2;
    int by = b.y + b.height / 2;
    double dist = pow(ax - bx, 2) + pow(ay - by, 2);
    dist = sqrt(dist);
    return dist;
}

Point findCenter(Rect a) {
    return Point(a.x + a.width / 2, a.y + a.height / 2);
}

Point findCenter(vector<Point> ps) {
    int x = 0;
    int y = 0;
    for (size_t i = 0; i < ps.size(); i++) {
        x += ps[i].x;
        y += ps[i].y;
    }
    x = x / ps.size();
    y = y / ps.size();
    return Point(x, y);
}

Point findCenter(vector<Rect> ps) {
    int x = 0;
    int y = 0;
    for (size_t i = 0; i < ps.size(); i++) {
        x += findCenter(ps[i]).x;
        y += findCenter(ps[i]).y;
    }
    x = x / ps.size();
    y = y / ps.size();
    return Point(x, y);
}

static double findAngle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    double radian = acos((dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10));
    double degree = radian * 180 / M_PI;
    return degree;
}

double compareHistogram(Mat src_base, Mat src_test1, int compare_method) {
    Mat hsv_base, hsv_test1;
    /// Convert to HSV
    cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
    cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );
    //cvtColor( src_test2, hsv_test2, COLOR_BGR2HSV );

    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };


    /// Histograms
    MatND hist_base;
    MatND hist_test1;
    //MatND hist_test2;

    /// Calculate the histograms for the HSV images
    calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
    normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
    normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

    /// Apply the histogram comparison methods
    //double base_base = compareHist( hist_base, hist_base, compare_method );
    //double base_half = compareHist( hist_base, hist_half_down, compare_method );
    double base_test1 = compareHist( hist_base, hist_test1, compare_method );
    return base_test1;
}


int main() {
    for (int num = 1; num < 13; num++) { //loop for all the pictures
        //if (num != 3) continue;
        char address[40];
        sprintf(address, "/home/behnam/Downloads/Dubaisi/%i.JPG", num);
        Mat img = imread(address);
        //pre-processing
        GaussianBlur( img, img, Size(31, 31), 2, 2 );
        pyrDown(img, img);
        Mat img_clone = img.clone();
        //extract hsv imag to filter the blue parts in order to find the blue squares
        Mat img_hsv = Mat::zeros(img.rows, img.cols, CV_8UC1);
        cvtColor(img, img_hsv, COLOR_BGR2HSV);
        Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
        //color filter the blue squares
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                if (img_hsv.at<Vec3b>(i, j)[0] < 140 and img_hsv.at<Vec3b>(i, j)[0] > 50) {
                    mask.at<uchar>(i, j) = 255;
                }
            }
        }
        Mat maskedImage = Mat::zeros(img.size(), img.type());
        add(img, maskedImage, maskedImage, mask);
        //extract basic features from the image.
        shape_feature_extractor sfe(maskedImage);
        vector<Rect> allSquares;
        //loop through bounding rects to find the blue circles
        for (size_t i = 0; i < sfe.size; i++) {
            Rect rect = sfe.bounding_rects[i];
            double ratio = ((double) rect.width) / rect.height;
            double rectagularity = sfe.good_contours_areas[i] / sfe.bounding_rects_areas[i];
            if (ratio > .9 and ratio < 1.1 and rectagularity > .8) {
                rectangle(img_clone, rect, Scalar(255, 100, 200), 8);
                allSquares.push_back(rect);
            }
        }
        //categorizing the blue circles.
        bool stopLoop = false;
        if (allSquares.size() == 0) stopLoop = true;
        Rect mainThree[3];
        vector<Rect> mainThrees[3];
        if (!stopLoop) mainThrees[0].push_back(allSquares[0]);
        for (size_t i = 1; i < allSquares.size(); i++) {
            for (int j = 0; j < 3; j++) {
                if (mainThrees[j].size() == 0) {
                    mainThrees[j].push_back(allSquares[i]);
                    break;
                }
                else {
                    double dist = distSquares(allSquares[i], mainThrees[j][0]);
                    if (dist < 10) {
                        mainThrees[j].push_back(allSquares[i]);
                        break;
                    }
                }
            }
        }
        int maxRectIndex[3];
        double maxArea[3];
        for (int i = 0; i < 3; i++) {
            if (mainThrees[i].size() == 0) {
                stopLoop = true;
                break;
            }
            maxArea[i] = img.rows * img.cols;
            for (size_t j = 0; j < mainThrees[i].size(); j++) {
                if (maxArea[i] > mainThrees[i][j].area()) {
                    maxArea[i] = mainThrees[i][j].area();
                    maxRectIndex[i] = j;
                }
            }
        }
        for (int i = 0; i < 3; i++) {
            mainThree[i] = mainThrees[i][maxRectIndex[i]];
        }
        if (stopLoop) {
            printf("three blue squares not found!\n");
            continue;
        }
        printf("size of all tha squares: %ld", allSquares.size());
        printf("number of rectangles in each category: %ld, %ld, %ld\n", mainThrees[0].size(), mainThrees[1].size(), mainThrees[2].size());
        printf("maximum area rectable indexes: (%d, %f), (%d, %f), (%d, %f)\n", maxRectIndex[0], maxArea[0],
                maxRectIndex[1], maxArea[1], maxRectIndex[2], maxArea[2]);
        //find features of the blue squares:
        printf("blue angles: ");
        int sharpEdglePointIndex = 0;
        for (int i = 0; i < 3; i++) {
            Point pt0 = findCenter(mainThree[i]);
            Point pt1 = findCenter(mainThree[(i + 1) % 3]);
            Point pt2 = findCenter(mainThree[(i + 2) % 3]);
            line(img_clone, pt0, pt1, Scalar(0, 100, 0), 5);
            line(img_clone, pt2, pt1, Scalar(0, 100, 0), 5);
            line(img_clone, pt2, pt0, Scalar(0, 100, 0), 5);
            double angle = findAngle(pt1, pt2, pt0);
            if (angle < 40) {
                sharpEdglePointIndex = i;
            }
            printf("%f, ", angle);
        }
        
        Rect temp = mainThree[sharpEdglePointIndex];
        mainThree[sharpEdglePointIndex] = mainThree[0];
        mainThree[0] = temp;

        printf("sharp point index: %d\n", sharpEdglePointIndex);
        circle(img_clone, findCenter(mainThree[0]), 5, Scalar(50, 0, 0), 5);

        //finding the gray rectangle
        shape_feature_extractor sfe2(img);
        vector<vector<Point>> grayRectangles;
        for (size_t i = 0; i < sfe2.size; i++) {
            Mat roi(img, sfe2.bounding_rects[i]);
            Mat tmplt = imread("/home/behnam/Downloads/Dubaisi/redRectangle.jpg");
            int tmplt_area = tmplt.rows * tmplt.cols;
            //area an histogram analysis
            double histAnalysis = compareHistogram(tmplt, roi, 0);
            RotatedRect rRect = sfe2.min_rects[i];
            Size2f size = rRect.size;
            double sideRatio = size.height / size.width; sideRatio = sideRatio >= 1 ? sideRatio : 1 / sideRatio;
            double area = sfe2.min_rects_areas[i];
            double area_ratio = area / mainThree[0].area();
            double rectangularity = sfe2.good_contours_areas[i] / sfe2.min_rects_areas[i];
            double angleCos = cos(rRect.angle / 180 * M_PI);
            double angle1 = findAngle(findCenter(mainThree[1]), rRect.center, findCenter(mainThree[0]));
            double angle2 = findAngle(findCenter(mainThree[2]), rRect.center, findCenter(mainThree[0]));
            double hull_ratio = utilities::calc_area(sfe2.hulls[i]) / area;
            if ( histAnalysis > .3
                and area_ratio > 1.2 and area_ratio < 2
                and sideRatio < 1.6 and sideRatio > 1.3 
                and angle1 > 10 and angle1 < 27
                and angle2 > 10 and angle2 < 27
                and rectangularity > .6
                //and num_h > 100
                //and num_s > 100
                ) {
                vector<Point> points = utilities::rotatedRect_to_contour(rRect);
                printf("area: %f\n", area);
                printf("area ratio %f\n", area_ratio);
                printf("side ratio: %f\n", sideRatio);
                printf("rectangularity: %f\n", rectangularity);
                printf("angle: %f, %f\n", angle1, angle2);
                printf("hull ratio %f\n", hull_ratio);
                printf("hist analysis: %f\n", histAnalysis);
                printf("****************\n");
                grayRectangles.push_back(points);
            }
        }

        drawContours(img_clone, grayRectangles, -1, Scalar(0, 0, 100), 5);

        imshow("image", img_clone);
        waitKey(0);
        printf("************************************************************************************\n");
    }
}

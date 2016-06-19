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

int squareType = 1;
int redRectangleType = 2;
int grayRectangleType = 3;

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

struct Specs {
    Mat img;
    Mat roi;
    Mat hsvRoi;
    Rect boundingRect;
    int dist;
    double angle;
    Rect mainThree[3];
    Point pts[3];
};

void show(Mat img) {
    imshow("test", img);
    waitKey(0);
}

int findSpecs(Mat img, Specs* tmpltSpecs) {
    
    Mat thresholded;
    float mainArea;
    Mat tmplt, dst;
    Rect tmpltRect, rct2, rct;
    int tempDist;
    //**********************************************************************************************************************************
    //pre-processing
    pyrDown(img, img); pyrDown(img, img);
    //GaussianBlur( img, img, Size(3, 3), 2, 2 );
    //
    // Mat element = getStructuringElement( MORPH_CROSS,
                                   // Size( 3, 3 ));
    //erode( img, img, element );
    Mat originalImg = img.clone();
    Mat img_clone = img.clone();
    //**********************************************************************************************************************************
    //extract hsv imag to filter the blue parts in order to find the blue squares
    //**********************************************************************************************************************************
    //filetering the blue parts
    Mat img_hsv = Mat::zeros(img.rows, img.cols, CV_8UC1);
    cvtColor(originalImg, img_hsv, COLOR_BGR2HSV);
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
        double rectangularity = sfe.good_contours_areas[i] / sfe.bounding_rects_areas[i];
        if (ratio > .9 and ratio < 1.1 and rectangularity > .6) {
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

    // for (size_t i = 0; i < allSquares.size(); i++) {
    //     rectangle(img_clone, allSquares[i], Scalar(100, 100, 100));
    // }
    // show(img_clone);

    int maxRectIndex[3] = {0, 0, 0};
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

    if (stopLoop) {
        printf("three blue squares not found!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return -1;
    }

    printf("%d, %d, %d\n", maxRectIndex[0], maxRectIndex[1], maxRectIndex[2]);

    for (int i = 0; i < 3; i++) {
        mainThree[i] = mainThrees[i][maxRectIndex[i]];
    }
    printf("mardas\n");

    
    printf("size of all tha squares: %ld", allSquares.size());
    printf("number of rectangles in each category: %ld, %ld, %ld\n", mainThrees[0].size(), mainThrees[1].size(), mainThrees[2].size());
    printf("maximum area rectable indexes: (%d, %f), (%d, %f), (%d, %f)\n", maxRectIndex[0], maxArea[0],
            maxRectIndex[1], maxArea[1], maxRectIndex[2], maxArea[2]);
    //find features of the blue squares:
    printf("blue angles: ");
    int sharpEdglePointIndex = 0;
    double mainAngle;
    for (int i = 0; i < 3; i++) {
        Point pt0 = findCenter(mainThree[i]);
        Point pt1 = findCenter(mainThree[(i + 1) % 3]);
        Point pt2 = findCenter(mainThree[(i + 2) % 3]);
        line(img_clone, pt0, pt1, Scalar(0, 100, 0), 5);
        line(img_clone, pt2, pt1, Scalar(0, 100, 0), 5);
        line(img_clone, pt2, pt0, Scalar(0, 100, 0), 5);
        double angleTemp = findAngle(pt1, pt2, pt0);
        if (angleTemp < 40) {
            sharpEdglePointIndex = i;
            mainAngle = angleTemp;
        }
        printf("%f, ", angleTemp);
    }
    
    Rect temp = mainThree[sharpEdglePointIndex];
    mainThree[sharpEdglePointIndex] = mainThree[0];
    mainThree[0] = temp;

    printf("sharp point index: %d\n", sharpEdglePointIndex);
    circle(img_clone, findCenter(mainThree[0]), 5, Scalar(50, 0, 0), 5);
    tmpltSpecs->img = img; tmplt = img;
    tmpltRect = mainThree[0];
    Point p1, p2, p3;
    p1 = findCenter(mainThree[0]); tmpltSpecs->pts[0] = p1;
    p2 = findCenter(mainThree[1]); tmpltSpecs->pts[1] = p2;
    p3 = findCenter(mainThree[2]); tmpltSpecs->pts[2] = p3;
    tempDist = findDist(p1, Point((p2.x + p3.x)/2, (p2.y + p3.y)/2)); tmpltSpecs->dist = tempDist;
    rct2 = Rect(p1.x - tempDist * 2.5, p1.y - tempDist, tempDist * 3.5, tempDist * 2); tmpltSpecs->boundingRect = rct2;
    tmpltSpecs->roi = Mat(img, rct2);
    tmpltSpecs->hsvRoi = Mat(img_hsv, rct2);

    Point horizontal_point = Point(p1.x - 10, p1.y);
    double horizontal_angle = findAngle(p2, horizontal_point, p1);
    double difference = -abs(horizontal_angle - mainAngle / 2);
    printf("main angle: %f, horisontal angle: %f, difference: %f\n", mainAngle, horizontal_angle, difference);
    Mat rot_mat = getRotationMatrix2D(p1, difference, 1);
    warpAffine(img, img, rot_mat, img.size());
    tmpltSpecs->img = img;
    for (int i = 0; i < 3; i++) {
        tmpltSpecs->mainThree[i] = mainThree[i];
    }
    return 1;
}

int main() {
    Mat thresholded;
    Mat tmplt, dst;
    Specs tmpltSpecs;
    // shape_feature_extractor sfe2(thresholded);
    char address[70] = "/home/behnam/Downloads/Dubaisi/Station_10/Speeder/speeder-good.JPG";
    // char address[70] = "/home/behnam/Downloads/Dubaisi/Station_10/Speeder/speeder-good.JPG";
    printf("file: %s", address);
    Mat img = imread(address);
    findSpecs(img, &tmpltSpecs);
    imshow("image", tmpltSpecs.img);
    
    waitKey(0);

    for (int num = 1; num < 5; num++) { //loop for all the pictures
        //if (num != 3) continue;
        char address[40];
        sprintf(address, "/home/behnam/Downloads/Dubaisi/Station_10/Speeder/speeder-defect%i.JPG", num);
        printf("file: %s", address);
        Mat img = imread(address);
        Specs dstSpecs;
        findSpecs(img, &dstSpecs);
        printf("source: %d, %d\n", tmpltSpecs.boundingRect.width, tmpltSpecs.boundingRect.height);
        printf("dst: %d, %d\n", dstSpecs.boundingRect.width, dstSpecs.boundingRect.height);
        Mat roid = dstSpecs.roi;
        Mat roic = tmpltSpecs.roi;
        Mat hsv_roid = dstSpecs.hsvRoi;
        Mat hsv_roic = tmpltSpecs.hsvRoi;

        if (roic.rows < roid.rows and roic.cols < roid.cols) {
            resize(roic, roic, Size(roic.cols, roic.rows));
            resize(roid, roid, Size(roic.cols, roic.rows));
            resize(hsv_roic, hsv_roic, Size(hsv_roic.cols, hsv_roic.rows));
            resize(hsv_roid, hsv_roid, Size(hsv_roic.cols, hsv_roic.rows));
        }
        else if (roic.rows > roid.rows and roic.cols > roid.cols) {
            resize(roic, roic, Size(roid.cols, roid.rows));
            resize(roid, roid, Size(roid.cols, roid.rows));
            resize(hsv_roic, hsv_roic, Size(hsv_roid.cols, hsv_roid.rows));
            resize(hsv_roid, hsv_roid, Size(hsv_roid.cols, hsv_roid.rows));
        }

        printf ("\n");

        imshow("destination", roid);
        
        Mat pertas = abs(roid - roic);
        Mat hsvPertas = abs(hsv_roid - hsv_roic);
        //divide(pertas, roic, pertas);
        //pertas = pertas * 255;
        // Mat h_split(pertas.rows, pertas.cols;
        Mat hsvSplit[3], rgbSplit[3];
        split(hsvPertas, hsvSplit);
        split(pertas, rgbSplit);
        Mat element = getStructuringElement(0, Size(13, 13));
        for (int i = 0; i < 3; i++) {
            inRange(hsvSplit[i], Scalar(100), Scalar(255), hsvSplit[i]);
            // erode(rgbSplit[i], rgbSplit[i], element);
            inRange(rgbSplit[i], Scalar(100), Scalar(255), rgbSplit[i]);
            // erode(hsvSplit[i], hsvSplit[i], element);
        }
        imshow("difference of hue", hsvSplit[0]);
        imshow("difference of saturation", hsvSplit[1]);
        imshow("difference of volume", hsvSplit[2]);

        imshow("difference of red", rgbSplit[0]);
        imshow("difference of green", rgbSplit[1]);
        imshow("difference of blue", rgbSplit[2]);
        imshow("template", roic);

        Mat andResult;
        bitwise_or(hsvSplit[0], hsvSplit[1], andResult);
        bitwise_or(rgbSplit[0], andResult, andResult);
        bitwise_or(rgbSplit[0], andResult, andResult);
        bitwise_or(rgbSplit[1], andResult, andResult);
        bitwise_or(rgbSplit[2], andResult, andResult);
        erode(andResult, andResult, element);
        imshow("or result", andResult);
        waitKey(0);

    }
}
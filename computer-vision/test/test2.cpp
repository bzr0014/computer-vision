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

int numPics = 0;
void show(Mat img) {
    char name[15] = "";
    sprintf(name, "test%d", numPics++);
    destroyWindow(name);
    sprintf(name, "test%d", numPics);
    imshow(name, img);
    waitKey(0);
}

Mat gray_mask, grayMaskedImage;
int findSpecs(Mat img, Specs* tmpltSpecs) {
    
    Mat thresholded;
    float mainArea;
    Mat tmplt, dst;
    Rect tmpltRect, rct2, rct;
    int tempDist;
     //pre-processing
    /// Apply the erosion operation
    pyrDown(img, img); pyrDown(img, img);
    //GaussianBlur( img, img, Size(3, 3), 2, 2 );
    medianBlur(img, img, 9);
    Mat element = getStructuringElement( MORPH_CROSS,
                                   Size( 3, 3 ));
    //erode( img, img, element );
    Mat originalImg = img.clone();
    Mat img_clone = img.clone();
    //extract hsv imag to filter the blue parts in order to find the blue squares
    Mat img_hsv = Mat::zeros(img.rows, img.cols, CV_8UC1);
    cvtColor(originalImg, img_hsv, COLOR_BGR2HSV);
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    if (gray_mask.empty()) gray_mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    else gray_mask.setTo(0);
    //color filter the blue squares
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img_hsv.at<Vec3b>(i, j)[0] < 140 and img_hsv.at<Vec3b>(i, j)[0] > 50) {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }
    // inRange(img_hsv, Scalar(140, 0, 0), Scalar(50, 255, 255), mask);
    inRange(img_hsv, Scalar(20, 9, 0), Scalar(110, 40, 255), gray_mask);
    element = getStructuringElement(0, Size(5, 5));
    dilate(gray_mask, gray_mask, element);
    if (grayMaskedImage.empty()) grayMaskedImage = Mat(img.rows, img.cols, img.type());
    else grayMaskedImage.setTo(0);
    add(img, 0, grayMaskedImage, gray_mask);
    cvtColor(grayMaskedImage, grayMaskedImage, CV_BGR2GRAY);
    show(grayMaskedImage);
    //find the gray circles
    vector<Vec3f> circles;
    //GaussianBlur(gray_mask, gray_mask, Size(5, 5), 2, 2);
    HoughCircles(grayMaskedImage, circles, CV_HOUGH_GRADIENT, 2, 50, 50, 90, 20, 50);
    bool goOn = true;
    if (circles.size() < 2) {
        goOn = false;
        printf("the circles not found!!!!!!!!!!!!\n");
        //return 0;
    }
    if (goOn) {
        vector<Vec3f> good_circles;
        good_circles.push_back(circles[0]);
        for (size_t i = 1; i < circles.size(); i++) {
            if (abs(circles[i][0] - good_circles[0][0]) < 10) {
                if (abs(circles[i][1] - good_circles[0][1]) > 20) {
                    good_circles.push_back(circles[i]);
                    break;
                }
            }
        }
        // gray_mask.release(); grayMaskedImage.release();
        printf("size of the circle: %ld\n", circles.size());
        for (size_t i = 0; i < good_circles.size(); i++) {
            circle(img_clone, Point((int) good_circles[i][0], (int) good_circles[i][1]), (int) good_circles[i][2], Scalar(100, 100, 100), 5);
        }
    }

    Mat maskedImage = Mat::zeros(img.size(), img.type());
    add(img, maskedImage, maskedImage, mask);
    //extract basic features from the image.
    shape_feature_extractor sfe(maskedImage);
    vector<Rect> allSquares;
    //loop through bounding rects to find the blue circles
    int max_area = - img.cols * img.rows;
    int conotourIndex = -1;
    for (size_t i = 0; i < sfe.size; i++) {
        Rect rect = sfe.bounding_rects[i];
        if (rect.height / (double) rect.width >= 2.5 && rect.height / (double) rect.width <= 2.9) {
            if (max_area < rect.area()) {
                max_area = rect.area();
                conotourIndex = i;
                allSquares.push_back(rect);
            }
        }
        // double ratio = ((double) rect.width) / rect.height;
        // double rectangularity = sfe.good_contours_areas[i] / sfe.bounding_rects_areas[i];
    }

    //categorizing the blue circles.
    bool stopLoop = false;
    if (allSquares.size() == 0) stopLoop = true;

    if (stopLoop) {
        printf("The blue rectangle not found\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return -1;
    }

    Rect mainRect = allSquares[allSquares.size() - 1];
    printf("main rect width is %d and its height is %d\n", mainRect.width, mainRect.height);
    rectangle(img_clone, mainRect, Scalar(255, 100, 200), 8);
    show(img_clone);
    
    tmpltSpecs->img = img; tmplt = img;
    tmpltRect = mainRect;
    Point p1, p2, p3;
    p1 = findCenter(mainRect);
    tempDist = max(mainRect.width, mainRect.height);
    rct2 = Rect(p1.x - tempDist, p1.y - tempDist / 2, tempDist * 2, tempDist); tmpltSpecs->boundingRect = rct2;
    tmpltSpecs->roi = Mat(img, rct2);
    tmpltSpecs->hsvRoi = Mat(img_hsv, rct2);

    // Point horizontal_point = Point(p1.x - 10, p1.y);
    //double horizontal_angle = findAngle(p2, horizontal_point, p1);
    RotatedRect rotated_rect = sfe.min_rects[conotourIndex];
    float blob_angle_deg = rotated_rect.angle;
    if (rotated_rect.size.width < rotated_rect.size.height) {
      blob_angle_deg = -blob_angle_deg;
    }
    else blob_angle_deg = -(90 + blob_angle_deg);
    Mat mapMatrix = getRotationMatrix2D(p1, blob_angle_deg, 1.0);
    // if (difference < -50) difference = (90 + difference);
    // else difference = -difference;
    printf("angle: %f\n", blob_angle_deg);
    warpAffine(img, img, mapMatrix, img.size());
    tmpltSpecs->img = img;
    show(img);
    return 1;
}

int main() {
    Mat thresholded;
    Mat tmplt, dst;
    Specs tmpltSpecs;
    // shape_feature_extractor sfe2(thresholded);
    char address[70] = "/home/behnam/Downloads/Dubaisi/Station_10/SUV/suv-good2.JPG";
    printf("file: %s", address);
    Mat img = imread(address);
    findSpecs(img, &tmpltSpecs);
    imshow("image", tmpltSpecs.img);
    
    waitKey(0);

    for (int num = 1; num < 5; num++) { //loop for all the pictures
        //if (num != 3) continue;
        char address[40];
        sprintf(address, "/home/behnam/Downloads/Dubaisi/Station_10/SUV/suv-defect%d.JPG", num);
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
        printf("*******************************************************\n");
        Mat hsvPertas = abs(hsv_roid - hsv_roic);
        printf("*******************************************************\n");
        //divide(pertas, roic, pertas);
        //pertas = pertas * 255;
        // Mat h_split(pertas.rows, pertas.cols;
        Mat hsvSplit[3], rgbSplit[3];
        split(hsvPertas, hsvSplit);
        split(pertas, rgbSplit);
        Mat element = getStructuringElement(0, Size(13, 13));
        for (int i = 0; i < 3; i++) {
            // inRange(hsvSplit[i], Scalar(50), Scalar(255), hsvSplit[i]);
            // erode(rgbSplit[i], rgbSplit[i], element);
            inRange(rgbSplit[i], Scalar(150), Scalar(255), rgbSplit[i]);
            // erode(hsvSplit[i], hsvSplit[i], element);
        }
        inRange(hsvSplit[0], Scalar(10), Scalar(255), hsvSplit[0]);
        inRange(hsvSplit[1], Scalar(50), Scalar(255), hsvSplit[1]);
        inRange(hsvSplit[2], Scalar(150), Scalar(255), hsvSplit[2]);

        imshow("difference of hue", hsvSplit[0]);
        imshow("difference of saturation", hsvSplit[1]);
        imshow("difference of volume", hsvSplit[2]);

        imshow("difference of red", rgbSplit[0]);
        imshow("difference of green", rgbSplit[1]);
        imshow("difference of blue", rgbSplit[2]);
        imshow("template", roic);

        Mat orResult, andResult;
        bitwise_or(hsvSplit[0], hsvSplit[1], orResult);
        bitwise_or(rgbSplit[0], orResult, orResult);
        bitwise_or(rgbSplit[0], orResult, orResult);
        bitwise_or(rgbSplit[1], orResult, orResult);
        bitwise_or(rgbSplit[2], orResult, orResult);

        bitwise_and(hsvSplit[0], hsvSplit[1], andResult);
        bitwise_and(rgbSplit[0], andResult, andResult);
        bitwise_and(rgbSplit[0], andResult, andResult);
        bitwise_and(rgbSplit[1], andResult, andResult);
        bitwise_and(rgbSplit[2], andResult, andResult);
        
        // erode(andResult, andResult, element);
        erode(orResult, orResult, element);
        imshow("or result", orResult);
        imshow("and result", andResult);
        waitKey(0);

    }
}
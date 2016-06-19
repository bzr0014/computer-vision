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

static void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage (it's got it all folks) to find\n"
    "squares in a list of images pic1-6.png\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
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

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
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
                        squares.push_back(approx);
                }
            }
        }
    }
}


static void findRectangles( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
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
                        squares.push_back(approx);
                }
            }
        }
    }
}

static void findRedRectangle( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
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
                        squares.push_back(approx);
                }
            }
        }
    }
}

// the function draws all the squares in the image
static void drawShapes( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA);
    }
    Mat toShow;
    pyrDown(image, toShow);
    imshow(wndname, toShow);
}

void showPyrredDown(string name, Mat img, int n) {
    Mat toShow = img.clone();
    for (int i = 0; i < n; i++) {
        pyrDown(img, toShow);
    }
    imshow(name, toShow);
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

RNG rng(255);
VideoCapture cap("/home/behnam/Downloads/Dubaisi/MOV_0370.mp4");
char c;

int main() {
    //Mat img = imread("/home/behnam/Downloads/M001 Screen Shot.png");
    //char num[] = "1";
    //for (int num = 1; num < 13; num++) {
    while (c != 'q') {
    char address[40];
    //sprintf(address, "/home/behnam/Downloads/Dubaisi/%i.JPG", num);
    Mat img;
    //img = imread(address);
    cap >> img;
    GaussianBlur( img, img, Size(31, 31), 2, 2 );
    pyrDown(img, img);
    //pyrDown(img, img);
    Mat img_hsv = Mat::zeros(img.rows, img.cols, CV_8UC1);
    cvtColor(img, img_hsv, COLOR_BGR2HSV);
    //img = img(Rect(5, 5, img.cols - 10, img.rows - 10));
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    
    //find only green and purple pixels in the picture and get a mask of them

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img_hsv.at<Vec3b>(i, j)[0] < 140 and img_hsv.at<Vec3b>(i, j)[0] > 50) {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }

    Mat res = Mat::zeros(img.rows, img.cols, CV_8UC3);   
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (mask.at<uchar>(i, j)) {
                res.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[0];
                res.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j)[1];
                res.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i, j)[2];
            }
        }
    }

    vector<vector<Point> > squares;
    
    GaussianBlur(res, res, Size(31, 31), 2, 2);
    findSquares(res, squares);
    drawShapes(img, squares);
    vector<Rect> mainThree, allSquares;
    for (int i = 0; i < 3; i++) {
        mainThree.push_back(Rect(0, 0, 1, 1));
    }

    for (size_t i = 0; i < squares.size(); i++) {
        Rect rect = boundingRect(squares[i]);
        allSquares.push_back(rect);
        bool resume = true;
        for (int i = 0; i < 3 and resume; i++) {
            if (mainThree[i].x == 0) {
                mainThree[i] = rect;
                resume = false;
            }
            else {
                if (distSquares(rect, mainThree[i]) < 100) {
                    if (rect.area() > mainThree[i].area()) mainThree[i] = rect;
                    resume = false;
                }
            }
        }
    }

    bool stopLoop = false;

    for (size_t i = 0; i < 3; i++) {
        rectangle(img, mainThree[i], Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 200)), -1);
        if(mainThree[i].x == 0) {
            stopLoop = true;
            continue;
        }
    }

    if (stopLoop) {
        printf("Three blue objects not detected!!\n");
        //continue;
    }

    vector<vector<Point>> grayRectangles;
    findRectangles(img, grayRectangles);
    Rect grayRectangle(0, 0, 1, 1);;
    for (size_t i = 0; i < grayRectangles.size(); i++) {
        Rect rect = boundingRect(grayRectangles[i]);
        if (rect.area() > grayRectangle.area() and rect.area() > mainThree[i].area()) grayRectangle = rect;
    }

    if (grayRectangle.x != 0) {
        Point center = findCenter(mainThree);
        double ratio = findDist(center, findCenter(mainThree[0])) / findDist(center, findCenter(grayRectangle));
        printf("ratio of gray: %f\n", ratio);
        Point p = findCenter(grayRectangle);
        if (true) {
            rectangle(img, grayRectangle, Scalar(0, 200, 0), 5);
            line(img, Point(p.x - 5, p.y - 5), Point(p.x, p.y), Scalar(0, 250, 0), 2);
            line(img, Point(p.x + 10, p.y - 10), Point(p.x, p.y), Scalar(0, 250, 0), 2);
        }
        else {
            rectangle(img, grayRectangle, Scalar(0, 0, 250), 5);
            line(img, Point(p.x - 10, p.y - 10), Point(p.x +10, p.y + 10), Scalar(0, 0, 250), 2);
            line(img, Point(p.x - 10, p.y + 10), Point(p.x + 10, p.y - 10), Scalar(0, 0, 250), 2);
        }
    }
    else {
        printf("Large gray Rectangle not recognized!\n");
        //continue;
    }



    vector<vector<Point>> redRectangles;
    findRedRectangle(img, redRectangles);
    Rect redRectangle(0, 0, 1, 1);;
    for (size_t i = 0; i < redRectangles.size(); i++) {
        Rect rect = boundingRect(redRectangles[i]);
        if (rect.area() > redRectangle.area() and rect.area() > mainThree[i].area()) redRectangle = rect;
    }
    //drawShapes(img, redRectangles);
    if (redRectangle.x != 0) {
        Point center = findCenter(mainThree);
        double ratio = findDist(center, findCenter(mainThree[0])) / findDist(center, findCenter(redRectangle));
        printf("ratio of red: %f\n", ratio);
        Point p = findCenter(redRectangle);
        if (true) {
            rectangle(img, redRectangle, Scalar(0, 200, 0), 5);
            line(img, Point(p.x - 5, p.y - 5), Point(p.x, p.y), Scalar(0, 250, 0), 2);
            line(img, Point(p.x + 10, p.y - 10), Point(p.x, p.y), Scalar(0, 250, 0), 2);
        }
        else {
            rectangle(img, redRectangle, Scalar(0, 0, 250), 5);
            line(img, Point(p.x - 10, p.y - 10), Point(p.x +10, p.y + 10), Scalar(0, 0, 250), 2);
            line(img, Point(p.x - 10, p.y + 10), Point(p.x + 10, p.y - 10), Scalar(0, 0, 250), 2);
        }
    }
    else {
        printf("Large red Rectangle not recognized!\n");
        //continue;
    }

    pyrDown(img, img);
    imshow("img", img);
    c = waitKey(30);
}
    return 0;
}


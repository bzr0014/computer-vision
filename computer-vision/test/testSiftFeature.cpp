// Program to illustrate SIFT keypoint and descriptor extraction, and matching using brute force
// Author: Samarth Manoj Brahmbhatt, University of Pennsylvania
 
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
 
using namespace cv;
using namespace std;
 
int main() {
    Mat train = imread("/home/behnam/Downloads/Dubaisi/redRectangle5.png"), train_g;
    pyrDown(train, train);
    cvtColor(train, train_g, CV_BGR2GRAY);
    medianBlur(train_g, train_g, 9);
    //detect SIFT keypoints and extract descriptors in the train image
    vector<KeyPoint> train_kp;
    Mat train_desc;
 
    SiftFeatureDetector featureDetector;
    featureDetector.detect(train_g, train_kp);
    SiftDescriptorExtractor featureExtractor;
    featureExtractor.compute(train_g, train_kp, train_desc);
 
    // Brute Force based descriptor matcher object
    BFMatcher matcher;
    vector<Mat> train_desc_collection(1, train_desc);
    matcher.add(train_desc_collection);
    matcher.train();
 
    // VideoCapture object
    // VideoCapture cap(0);
 
    unsigned int frame_count = 0;
 
    while(char(waitKey(1)) != 'q') {
        double t0 = getTickCount();
        Mat test, test_g;
        //cap >> test;
	test = imread("/home/behnam/Downloads/Dubaisi/Station_1/Speeder/4.JPG");
        if(test.empty())
            continue;
 	pyrDown(test, test); pyrDown(test, test);
        cvtColor(test, test_g, CV_BGR2GRAY);
 	medianBlur(test_g, test_g, 9);
        //detect SIFT keypoints and extract descriptors in the test image
        vector<KeyPoint> test_kp;
        Mat test_desc;
        featureDetector.detect(test_g, test_kp);
        featureExtractor.compute(test_g, test_kp, test_desc);
 
        // match train and test descriptors, getting 2 nearest neighbors for all test descriptors
        vector<vector<DMatch> > matches;
        matcher.knnMatch(test_desc, matches, 2);
 
        // filter for good matches according to Lowe's algorithm
        vector<DMatch> good_matches;
        for(int i = 0; i < matches.size(); i++) {
            if(matches[i][0].distance < 0.6 * matches[i][1].distance)
                good_matches.push_back(matches[i][0]);
        }
 
        Mat img_show;
        drawMatches(test, test_kp, train, train_kp, good_matches, img_show);
        imshow("Matches", img_show);
 
        cout << "Frame rate = " << getTickFrequency() / (getTickCount() - t0) << endl;
    }
 
    return 0;
}

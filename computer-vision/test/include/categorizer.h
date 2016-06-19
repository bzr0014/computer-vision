#ifndef CATEGORIZER_H
#define CATEGORIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include "../include/Config.h"

using namespace cv;
using namespace std;
using namespace boost::filesystem;

class categorizer {
    private:
        map<string, Mat> templates, objects, positive_data, negative_data; //maps from category names to data
        multimap<string, Mat> train_set; //training images, mapped by category name
        map<string, CvSVM> svms;         //trained SVMs, mapped by category name
        vector<string> category_names;   //names of the categories found in TRAIN_FOLDER
        int categories;                  //number of categories
        int clusters;                    //number of clusters for SURF features to build vocabulary
        Mat vocab;                       //vocabulary

        // Feature detectors and descriptor extractors
        Ptr<FeatureDetector> featureDetector;
        Ptr<DescriptorExtractor> descriptorExtractor;
        Ptr<BOWKMeansTrainer> bowtrainer;
        Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor;
        Ptr<FlannBasedMatcher> descriptorMatcher;

        void make_train_set();           //function to build the training set multimap
        void make_pos_neg();             //function to extract BOW features from training images and organize them into positive and negative samples
        string remove_extension(string); //function to remove extension from file name, used for organizing templates into categories
    public:
        categorizer(int);                //constructor
        void build_vocab();              //function to build the BOW vocabulary
        void train_classifiers();        //function to train the one-vs-all SVM classifiers for all categories
        void categorize(VideoCapture);   //function to perform real-time object categorization on camera frames
        Mat categorize(Mat);   //function to perform real-time object categorization on camera frames
        Mat categorize(Mat frame, float* score);
};

#endif

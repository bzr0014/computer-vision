#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace cv;


/// Global variables
Mat src, erosion_dst, dilation_dst;
std::unordered_map<std::string, int> u;
std::unordered_map<std::string, int> v;
/** Function Headers */
void Erosion( int, void* );
void Dilation( int, void* );
void cannation(int, void*);

/** @function main */
int main( int argc, char** argv )
{
  u["image_num"]= 1;
  u["erosion_elem"]= 0;
  u["erosion_size"]= 0;
  u["dilation_elem"]= 0;
  u["dilation_size"]= 0;
  u["gaussian_elem"]= 0;
  u["canny_first"]= 50;
  u["canny_second"]= 150;
  u["canny_kernel_size"]= 1;
  u["pyr_num"]= 2;
  u["areaRatio"]= 1;
  u["channel number"]= 0;
  u["interval"] = 5;
  u["num interval"] = 0; 
  u["hsv thresholding?"] = 1; v["hsv thresholding?"] = 1 - 1;
  u["incrementalThresholding"] = 0; v["incrementalThresholding"] = 1 - 1;

  v["erosion_elem"]= 2 - 1;
  v["erosion_size"]= 21 - 1;
  v["dilation_elem"]= 2 - 1;
  v["dilation_size"]= 21 - 1;
  v["gaussian_elem"]= 21 - 1;
  v["canny_first"]= 300 - 1;
  v["canny_second"]= 400 - 1;
  v["canny_kernel_size"]= 3 - 1;
  v["pyr_num"]= 4 - 1;
  v["image_num"]= 13 - 1;
  v["areaRatio"]= 100 - 1;
  v["channel number"]= 2 - 1;
  v["interval"] = 50 - 1;
  v["num interval"] = 80;

  // Iterate and print keys and values of unordered_map
  for( const auto& n : u ) {
     std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  }

  /// Load an image
  src = imread( argv[1] );

  if( !src.data )
  { return -1; }

  /// Create windows
  //namedWindow( "Erosion Demo", CV_WINDOW_AUTOSIZE );
  //namedWindow( "Dilation Demo", CV_WINDOW_AUTOSIZE );
  namedWindow("control", CV_WINDOW_AUTOSIZE);
  cvMoveWindow( "Dilation Demo", src.cols, 0 );


  for (auto& n : u){
      createTrackbar( n.first, "control", &n.second, v[n.first] + 1 , cannation);   
  }

  /// Default start
  //Erosion( 0, 0 );
  //Dilation( 0, 0 );
  while (char(waitKey(0)) != 'q') {
    cannation(0, 0);
  }
  waitKey(0);
  return 0;
}

void cannation( int, void* )
{
  char address[40];
  sprintf(address, "/home/behnam/Downloads/Dubaisi/%d.JPG", u["image_num"]);
  Mat src_clone;
  src_clone = imread(address);
  if (src_clone.empty()) src_clone = src.clone();
  for (int i = 0; i < u["pyr_num"]; i++) {
    pyrDown(src_clone, src_clone);
  } 

  GaussianBlur(src_clone, src_clone, Size(u["gaussian_elem"] * 2 + 1, u["gaussian_elem"] * 2 + 1), 3);

  Mat gray_image(src_clone.size(), CV_8UC1), hsv_image, canny_image;
  //cvtColor(src_clone, gray_image, CV_BGR2GRAY);
  cvtColor(src_clone, hsv_image, CV_BGR2HSV);
  int ch[] = {u["channel number"], 0};
  if (u["hsv thresholding?"])  mixChannels(&hsv_image, 1, &gray_image, 1, ch, 1);
  else   mixChannels(&src_clone, 1, &gray_image, 1, ch, 1);
  
  if (u["num interval"] == 0) {
    Canny(gray_image, canny_image, u["canny_first"], u["canny_second"], 2 * u["canny_kernel_size"] + 1);
  }
  else {
    if (u["incrementalThresholding"])
      bitwise_and(gray_image >= (u["num interval"] - 1) * u["interval"],  gray_image <= u["num interval"] * u["interval"], canny_image);
    else
      canny_image = gray_image <= u["num interval"] * u["interval"];
  }


  int erosion_type;
  if( u["erosion_elem"] == 0 ){ erosion_type = MORPH_RECT; }
  else if( u["erosion_elem"] == 1 ){ erosion_type = MORPH_CROSS; }
  else if( u["erosion_elem"] == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*u["erosion_size"] + 1, 2*u["erosion_size"]+1 ),
                                       Point( u["erosion_size"], u["erosion_size"] ) );
  erode( canny_image, canny_image, element );

  int dilation_type;
  if( u["dilation_elem"] == 0 ){ dilation_type = MORPH_RECT; }
  else if( u["dilation_elem"] == 1 ){ dilation_type = MORPH_CROSS; }
  else if( u["dilation_elem"] == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element2 = getStructuringElement( dilation_type,
                                       Size( 2*u["dilation_size"] + 1, 2*u["dilation_size"]+1 ),
                                       Point( u["dilation_size"], u["dilation_size"] ) );
  /// Apply the dilation operation
  dilate( canny_image, canny_image, element2 );

  vector<vector<Point>> contours, good_contours;
  findContours(canny_image.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  for (size_t i = 0; i < contours.size(); i++) {
    if (contourArea(contours[i]) < (src_clone.cols * src_clone.rows) / (u["areaRatio"] * 10.0) ) continue;
    good_contours.push_back(contours[i]);
  }

  Mat contour_image(src_clone.size(), CV_8UC1);
  drawContours(contour_image, good_contours, -1, Scalar(100));
  Mat maskedImage(src_clone.size(), src_clone.type());
  int ch2[] = {0, 0, 0, 1, 0, 2};
  mixChannels(&canny_image, 1, &maskedImage, 1, ch2, 3);
  imshow( "Canny Image", canny_image);
  imshow( " contours", contour_image);
  multiply(src_clone, maskedImage, src_clone);
  imshow( "image", src_clone);
}
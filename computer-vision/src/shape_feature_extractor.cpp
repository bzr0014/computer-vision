#include <../include/shape_feature_extractor.h>
#include <../include/utilities.h>
#include <stdio.h>
#include <sstream>

using namespace std;
using namespace cv;

RNG rng(255);

template<class T> vector<T> array_to_vector(T* src, int src_size) {
	vector<T> dst;
	for (int i = 0; i < src_size; i++) {
		dst.push_back(src[i]);
	}
	return dst;
}
void shape_feature_extractor::get_contours() {
	Mat image_canny;
	Canny(image, image_canny, 50, 5);
	Mat str_el = getStructuringElement(CV_SHAPE_RECT, Size(5, 5));
	morphologyEx(image_canny, image_canny, MORPH_DILATE, str_el);
	morphologyEx(image_canny, image_canny, MORPH_DILATE, str_el);
	morphologyEx(image_canny, image_canny, MORPH_CLOSE, str_el);
	image_canny = Scalar::all(255) - image_canny;
	//imshow("mardas", image_canny);

	findContours(image_canny, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	image_area = image.cols * image.rows;
	approx = vector<vector<Point> >(contours.size());
}
void shape_feature_extractor::get_contours(int t)
{
	printf("\nstart to extract contours ...\n");
	int N[2][3] = { {51, 51, 51}, {51, 51, 51} };
	int thresh = 50;
	Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// down-scale and upscale the image to filter out the noise
	pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
	pyrUp(pyr, timg, image.size());
	vector<vector<Point> > tempContours;

	// find squares in every color plane of the image
	for (int is_incremental = 0; is_incremental < 2; is_incremental++) {
		for( int mode = 0; mode < 2; mode++) {
			for( int c = 0; c < 3; c++ )
			{
			int ch[] = {c, 0};
			mixChannels(&timg, 1, &gray0, 1, ch, 1);
			equalizeHist(gray0, gray0);
			// try several threshold levels
			for( int l = 0; l < N[mode][c]; l++ )
			{
				// hack: use Canny instead of zero threshold level.
				// Canny helps to catch squares with gradient shading
				if( l == 0 )
				{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 0, 150, 5);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1,-1));
				}
				else
				{
					// apply threshold if l!=0:
					//	 tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
					gray = gray0 < (l+1)*255/N[mode][c];
					if (is_incremental) bitwise_and(gray, gray0 >= l * 255 / N[mode][c], gray);
					Mat element = getStructuringElement( MORPH_RECT, Size(3, 3) );
					erode(gray, gray, element );
				
				}

				// find contours and store them all as a list
				findContours(gray, tempContours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
				// test each contour
				for( size_t i = 0; i < tempContours.size(); i++ )
				{
					contours.push_back(tempContours[i]);
				}
			}
			}
		}
	}
	image_area = image.cols * image.rows;
	approx = vector<vector<Point> >(contours.size());
	printf("finished extracting contours!\n\n");
}
void shape_feature_extractor::filter_contours() {
	printf("started filtering contours ... \n");
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), approx[i], arcLength(Mat(contours[i]), true)*0.02, true);
		double contour_area = utilities::calc_area(approx[i]);
		if (contour_area > image_area / 800)// && contourArea(approx[i]) < 10000)
		{
			good_contours.push_back(approx[i]);
			bounding_rects.push_back(boundingRect(Mat(approx[i])));
			min_rects.push_back(minAreaRect(Mat(approx[i])));
			vector<Point> hull;
			convexHull(Mat(contours[i]), hull);
			hulls.push_back(hull);
			Point2f center;
			float radius;
			minEnclosingCircle(Mat(contours[i]), center, radius);
			Point int_center((int)center.x, (int)center.y);
			centers.push_back(int_center);
			radii.push_back((double)radius);
		}
	}
	size = good_contours.size();
	printf("finished filtering contours!\n\n");
}
void shape_feature_extractor::get_shapes() {
	printf("start getting shapes ... \n");	
	Mat contour_image = Mat::zeros(image.size(), CV_8UC1);
	Rect mardas = Rect(5, 10, 4, 3);
	double raghas = utilities::calc_area(utilities::rect_to_contour(mardas));
	for (int i = 0; i < good_contours.size(); i++) {
		double contour_area = utilities::calc_area(good_contours[i]);
		double rect_area = utilities::calc_area(utilities::rect_to_contour(bounding_rects[i]));
		double ratio = contour_area / rect_area;
		vector<Point> min_rect_pts = utilities::rotatedRect_to_contour(min_rects[i]);
		min_rects_contour.push_back(min_rect_pts);
		double min_rect_area = utilities::calc_area(min_rect_pts);
		ratio = min_rect_area / rect_area;
		double angle = min_rects[i].angle;
		double hull_area = utilities::calc_area(hulls[i]);
		ratio = hull_area / min_rect_area;
		//if (ratio > .9) {
			drawContours(contour_image, good_contours, i, Scalar::all(abs(rng.gaussian(255))), 4);
			//drawContours(contour_image, hulls, i, Scalar::all(abs(rng.gaussian(255))), 4);
			drawContours(contour_image, min_rects_contour, i, Scalar::all(abs(rng.gaussian(255))), 4);
			//rectangle(contour_image, bounding_rects[i], Scalar::all(abs(rng.gaussian(255))), 1);
			//circle(contour_image, centers[i], radii[i], Scalar::all(abs(rng.gaussian(255))));
		//}
	}
	imshow("contour_image", contour_image);
	printf("finished getting shapes ...\n\n");
	//waitKey(0);
}
void shape_feature_extractor::get_shape_areas() {
	printf("start getting shape areas ... \n");	
	int size = good_contours.size();
	for (int i = 0; i < size; i++) {
		good_contours_areas.push_back(utilities::calc_area(good_contours[i]));
		bounding_rects_areas.push_back(utilities::calc_area(utilities::rect_to_contour(bounding_rects[i])));
		approx_areas.push_back(utilities::calc_area(approx[i]));
		min_rects_areas.push_back(utilities::calc_area(min_rects_contour[i]));
		hulls_areas.push_back(utilities::calc_area(hulls[i]));
		circles_areas.push_back(3.14 * pow(radii[i], 2));
	}
	printf("finished getting shape areas!\n\n");
}
void shape_feature_extractor::add_to_features(vector<double> feature, string label) {
	int num_features = features.size();
	typedef pair<string, int> String_int_Pair;
	labels.insert(String_int_Pair(label, num_features));
	features.push_back(feature);
	
}

vector<double> shape_feature_extractor::calc_hull_to_min_rect() {
	vector<double> ratios(size);
	//calculate area ratio
	for (int i = 0; i < size; i++) {
		ratios[i] = hulls_areas[i] / min_rects_areas[i];
	}
	add_to_features(ratios, "hull_to_minRect");
	//hull_to_mean_rect_statistics = utilities::calc_statistics(hulls_area);
	//if (size > 1) hull_area = utilities::normalize_values(hull_area);
	return ratios;
}
vector<double> shape_feature_extractor::calc_min_rect_length_to_width() {
	vector<double> min_rect_length_to_width(size);
	for (int i = 0; i < size; i++) {
			min_rect_length_to_width[i] = min_rects[i].size.height / min_rects[i].size.width;
		}
		//min_rect_length_to_width_statistics = utilities::calc_statistics(min_rect_length_to_width);
		//if (size > 1) min_rect_length_to_width = utilities::normalize_values(min_rect_length_to_width);
	add_to_features(min_rect_length_to_width, "minRect_length_to_width");
	return min_rect_length_to_width;
}
vector<double> shape_feature_extractor::calc_rectagularity() {
	vector<double> rectangularity(size);
	for (int i = 0; i < size; i++) {
		rectangularity[i] = good_contours_areas[i] / min_rects_areas[i];
	}
	add_to_features(rectangularity, "rectangularity");
	return rectangularity;
	//rectangularity_statistics = utilities::calc_statistics(rectangualrity);
	//if (size > 1) rectangualrity = utilities::normalize_values(rectangualrity);
}
vector<vector<double> > shape_feature_extractor::calc_hu_moments() {
	Moments contour_moments;
	vector<vector<double> > hu_moments(7);
	for (int i = 0; i < size; i++) {
		contour_moments = moments(good_contours[i]);
		double temp[7];
		HuMoments(contour_moments, temp);
		for (int j = 0; j < 7; j++) {
			hu_moments[j].push_back(temp[j]);
		}
	}
	for (int j = 0; j < 7; j++)  {
		std::ostringstream ostr;
		ostr << j;
		add_to_features(hu_moments[j], "hu moment " + ostr.str());
	//	hu_moments_statistics.push_back(utilities::calc_statistics(hu_moments[j]));
	//	if (size > 1) hu_moments[j] = utilities::normalize_values(hu_moments[j]);
	}
	return hu_moments;
}
vector<double> shape_feature_extractor::calc_elongatedness() {
	Mat str_el = getStructuringElement(CV_SHAPE_RECT, Size(5, 5));
	vector<double> elongatedness(size);
	for (int i = 0; i < size; i++) {
		Mat contour_image = Mat::zeros(image.size(), CV_8UC1);
		drawContours(contour_image, good_contours, i, Scalar(255), CV_FILLED);
		contour_image = Mat(contour_image, bounding_rects[i]);
		int summation;
		int d = 0;
		while ((summation = (int)sum(contour_image).val[0]) != 0) {
			d++;
			erode(contour_image, contour_image, str_el);
		}
		elongatedness[i] = d == 0 ? 0 : good_contours_areas[i] / pow(2 * d, 2);
	}
	//if (size > 1) elongatedness_statistics = utilities::normalize_values(elongatedness);
	add_to_features(elongatedness, "elongatedness");
	return elongatedness;
}
void shape_feature_extractor::extract_features() {
	calc_hull_to_min_rect();
	calc_min_rect_length_to_width();
	calc_rectagularity();
	calc_hu_moments();
	calc_elongatedness();
}
void shape_feature_extractor::match_shapes(shape_feature_extractor that) {
	Mat disp_image = image.clone();
	Mat train_disp_image = that.image.clone();
	for (int j = 0; j < that.size; j++) {
		double distance;
		int min_index = -1;
		double min_distance = INFINITY;
		for (int i = 0; i < size; i++) {
			int feature_index = labels.at("elongatedness");
			distance = abs(features[feature_index][i] - that.features[feature_index][j]) / that.features[10][j];
			if (distance < min_distance) {
				min_distance = distance;
				min_index = i;
			}
		}
		Scalar color = Scalar(abs(rng.gaussian(255)), abs(rng.gaussian(255)), abs(rng.gaussian(255)));
		rectangle(disp_image, bounding_rects[min_index], color, 8);
		rectangle(train_disp_image, that.bounding_rects[j], color, 4);
		//destroyAllWindows();
	}
	pyrDown(disp_image, disp_image); pyrDown(disp_image, disp_image);
	imshow("image", disp_image);
	imshow("Train image", train_disp_image);
	//waitKey(0);
	//destroyAllWindows();
}
void shape_feature_extractor::getImage(Mat _image) {
	printf("start getting image ... \n");	
	if (_image.empty()) return;	
	image = _image.clone();
	get_contours(1);
	filter_contours();
	get_shapes();
	get_shape_areas();
	printf("finished getting image!\n");
}
shape_feature_extractor::shape_feature_extractor(Mat _image) {
	getImage(_image);
}
shape_feature_extractor::shape_feature_extractor() {}

/*
 * test.cu
 *
 *  Created on: Jun 14, 2017
 *      Author: Munesh Singh
 */

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp" // for video capture
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda_types.hpp"
#include <iostream>

void process(const char* img, const char* templ) {
	cv::cuda::setDevice(0);	//initialize CUDA

	cv::Mat image_h = cv::imread(img);
	cv::Mat templ_h = cv::imread(templ);

	cv::cuda::GpuMat templ_d(templ_h); //upload image on gpu
	cv::cuda::GpuMat image_d, result;

	if (image_h.empty())
		exit(1);

	image_d.upload(image_h);
	cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(
			templ_h.type(), CV_TM_CCOEFF_NORMED);

	cv::cuda::GpuMat dst;
	alg->match(image_d, templ_d, result);

	cv::cuda::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1);
	double max_value;

	cv::Point location;

	cv::cuda::minMaxLoc(result, 0, &max_value, 0, &location);


	//copying back to host memory for display
	cv::Mat result_h;
	result.download(result_h);

	//show now the two rectangles, one with the image and the other with matched template

	cv::rectangle(image_h, location,
			cv::Point(location.x + templ_h.cols, location.y + templ_h.rows),
			cv::Scalar::all(0), 2, 8, 0);
	cv::rectangle(result_h, location,
				cv::Point(location.x + templ_h.cols, location.y + templ_h.rows),
				cv::Scalar::all(0), 2, 8, 0);
	cv::imshow("Frame", result_h);
	cv::imshow("Image", image_h);

//	cv::waitKey(0);

}

int main(int argc, char** argv) {
	/*
	cv::Point3f* pixel_ptr = (cv::Point3f*) (d_img.data + d_img.step * y) + x;
	cv::Point3f* pixel_ptr1 = d_img.ptr<cv::Point3f>(y) + x;

	std::cout << pixel_ptr << std::endl << pixel_ptr1 << std::endl;
	 */
	if(argc<3) {
		std::cout << "Usage: 1test ../data/image.jpg ../data/templ.jpg" << std::endl;
		exit(1);
	}

	process(argv[1],argv[2]);
	cv::waitKey(0);
	return 0;

}

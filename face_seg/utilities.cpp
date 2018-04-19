#include "face_seg/utilities.h"
#include <opencv2/imgproc.hpp>

namespace face_seg
{
	void renderSegmentationBlend(cv::Mat& img, const cv::Mat& seg, float alpha,
		const cv::Scalar& color)
	{
		cv::Point3_<uchar> bgr((uchar)color[0], (uchar)color[1], (uchar)color[2]);
		int r, c;
		cv::Point3_<uchar>* img_data = (cv::Point3_<uchar>*)img.data;
		unsigned char* seg_data = seg.data;
		unsigned char s;
		for (r = 0; r < img.rows; ++r)
		{
			for (c = 0; c < img.cols; ++c)
			{
				s = *seg_data++;
				if (s > 128)
				{
					img_data->x = (unsigned char)std::round(bgr.x * alpha + img_data->x*(1 - alpha));
					img_data->y = (unsigned char)std::round(bgr.y * alpha + img_data->y*(1 - alpha));
					img_data->z = (unsigned char)std::round(bgr.z * alpha + img_data->z*(1 - alpha));
				}
				++img_data;
			}
		}
	}

	void removeSmallerComponents(cv::Mat& seg)
	{
		cv::Mat labels;
		cv::Mat stats, centroids;
		cv::connectedComponentsWithStats(seg, labels, stats, centroids);
		if (stats.rows <= 2) return;

		// Find the label of the connected component with maximum area
		cv::Mat areas = stats.colRange(4, 5).clone();
		int* areas_data = (int*)areas.data;
		int max_label = std::distance(areas_data,
			std::max_element(areas_data + 1, areas_data + stats.rows));

		// Clear smaller components
		unsigned char* seg_data = seg.data;
		int* labels_data = (int*)labels.data;
		for (size_t i = 0; i < seg.total(); ++i, ++seg_data)
			if (*labels_data++ != max_label) *seg_data = 0;
	}

	void smoothFlaws(cv::Mat& seg, int smooth_iterations, int smooth_kernel_radius)
	{
		int kernel_size = smooth_kernel_radius * 2 + 1;
		cv::Mat kernel = cv::getStructuringElement(
			cv::MorphShapes::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
		//        for (int i = 0; i < smooth_iterations; ++i)
		{
			cv::morphologyEx(seg, seg, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), smooth_iterations);
			cv::morphologyEx(seg, seg, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), smooth_iterations);
		}
	}

	void fillHoles(cv::Mat& seg)
	{
		double min_val, max_val;
		cv::minMaxLoc(seg, &min_val, &max_val);
		cv::Mat holes = seg.clone();
		cv::floodFill(holes, cv::Point2i(0, 0), cv::Scalar(max_val));
		for (size_t i = 0; i < seg.total(); ++i)
		{
			if (holes.data[i] == 0)
				seg.data[i] = (unsigned char)max_val;
		}
	}

	void postprocessSegmentation(cv::Mat & seg, bool disconnected,
		bool holes, bool smooth, int smooth_iterations, int smooth_kernel_radius)
	{
		if (disconnected) removeSmallerComponents(seg);
		if (holes) fillHoles(seg);
		if (smooth) smoothFlaws(seg, smooth_iterations, smooth_kernel_radius);
		if (disconnected) removeSmallerComponents(seg);
		if (holes) fillHoles(seg);
	}

}   // namespace face_seg


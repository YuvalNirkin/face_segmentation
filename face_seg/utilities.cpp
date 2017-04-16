#include "face_seg/utilities.h"

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
}   // namespace face_seg


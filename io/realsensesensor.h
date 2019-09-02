#pragma once
#include <opencv2/opencv.hpp>
#include <librealsense/rs.hpp>
#include <memory>
class RealsenseSensor
{
public:
    RealsenseSensor();
    ~RealsenseSensor();

    bool getImages(cv::Mat& rgbImage, cv::Mat& rawDepthImage, const float depthScaleFactorVoid);
private:
    rs::context ctx;
    rs::device* sensor;
    float depthScaleFactor;
    rs::intrinsics depth_intrin,color_intrin;
    rs::extrinsics depth_to_color;
    int nFramesRead;
};



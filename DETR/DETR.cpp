#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

#ifdef _WIN32
    const wchar_t* model_path = L"C:/Users/dell/source/repos/DETR/models/DETRmodel.onnx";
#else
    const char* model_path = "C:/Users/dell/source/repos/DETR/models/DETRmodel.onnx";
#endif

    std::cout << "Using Onnxruntime C++ API\n";
    Ort::Session session(env, model_path, session_options);

    std::cout << "Session:" << session << "\n";

    Mat image;
    image = imread("C:/Users/dell/source/repos/DETR/test.jpg", IMREAD_COLOR);
    
}
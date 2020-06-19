#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main()
{
    // Load image to process
    Mat image;
    image = imread("C:/Users/dell/source/repos/DETR/test.jpg", IMREAD_COLOR);
    // If image is an empty matrix
    if (!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    image.convertTo(image, CV_32FC3);
    // Change image format from BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::Mat image_resized;
    // Resize image to (256x256) to fit model input dimensions
    cv::resize(image, image_resized, Size(256, 256));

    cv::Mat image_float;
    //Normalize image (values between 0-1)
    image_resized.convertTo(image_float, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(image_float, channels);

    std::vector<double> mean = { 0.485, 0.456, 0.406 };
    std::vector<double> stddev = { 0.229, 0.224, 0.225 };
    size_t i = 0;
    for (auto& c : channels) {
        c = (c - mean[i]) - stddev[i];
        ++i;
    }

    cv::Mat image_normalized;
    cv::merge(channels, image_normalized);

    // create ONNX env and sessionOptions objects
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

    // path to model
    #ifdef _WIN32
        const wchar_t* model_path = L"C:/Users/dell/source/repos/DETR/models/DETRmodel.onnx";
    #else
        const char* model_path = "C:/Users/dell/source/repos/DETR/models/DETRmodel.onnx";
    #endif

    // create ONNX session
    Ort::Session session(env, model_path, session_options);

    // define model input and output node names
    static const char* input_names[] = { "image" };
    static const char* output_names[] = { "probs", "boxes" };

    // get input node infor
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dims;
    input_node_dims = tensor_info.GetShape();

    // create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    size_t input_tensor_size = 256 * 256 * 3;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(image_normalized.data), input_tensor_size, input_node_dims.data(), 4);

    // pass inputs through model and get output
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 2);

    //TODO: Process outputs
}
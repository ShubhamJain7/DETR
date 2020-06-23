#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main()
{
    // Load image to process
    Mat image;
    image = imread("C:/Users/dell/source/repos/DETR/test.jpg", IMREAD_COLOR);
    // If image is an empty matrix
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Convert image values from int to float
    image.convertTo(image, CV_32FC3);
    // Change image format from BGR to RGB
    cvtColor(image, image, COLOR_BGR2RGB);

    Mat image_resized;
    // Resize image to (256x256) to fit model input dimensions
    resize(image, image_resized, Size(256, 256));
    
    // Normalize image (values between 0-1)
    Mat image_float;
    image_resized.convertTo(image_float, CV_32FC3, 1.0f / 255.0f, 0);

    // Split image channels
    vector<cv::Mat> channels(3);
    split(image_float, channels);

    // Define mean and std-dev for each channel
    vector<double> mean = { 0.485, 0.456, 0.406 };
    vector<double> stddev = { 0.229, 0.224, 0.225 };
    size_t i = 0;
    // Normalize each channel with corresponding mean and std-dev values
    for (auto& c : channels) {
        c = (c - mean[i]) / stddev[i];
        ++i;
    }

    // Concatenate channels to change format from HWC to CHW
    Mat image_normalized;
    vconcat(channels, image_normalized);

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

    // get input node info
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    vector<int64_t> input_node_dims;
    input_node_dims = tensor_info.GetShape();

    // create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    size_t input_tensor_size = 256 * 256 * 3;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(image_normalized.data), input_tensor_size, input_node_dims.data(), 4);

    // pass inputs through model and get output
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 2);

    auto scores = output_tensors[0].GetTensorMutableData<float>();

    // Get length of output
    auto typeInfo = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t len = typeInfo.GetElementCount();

    // store outputs in a 2d array for easier access and processing
    // calculate and store sum of exponents for each row to be used as denominator for apploying the softmax function
    float probs[100][92];
    size_t index = 0;
    float denominator[100];
    while (index < len) {
        for (int i = 0; i < 100; i++) {
            float val = 0;
            for (int j = 0; j < 92; j++) {
                probs[i][j] = scores[index];
                val += exp(probs[i][j]);
                ++index;
            }
            denominator[i] = val;
        }
    }
    
    // Calculate softmax of each item(row-wise) by didving exponent of item by sum of exponents
    // Ignore 92nd column as it isn't required
    // Find the highest probablility and it's index
    float softs[100][91];
    for (size_t i = 0; i < 100; i++)
    {
        float max = 0;
        int index = -1;
        for (size_t j = 0; j < 91; j++)
        {
            softs[i][j] = exp(probs[i][j])/denominator[i];
            if (softs[i][j] >= max) {
                max = softs[i][j];
                index = j;
            }
        }
        if (max > 0.75) {
            cout << i << ":" << index << "(" << max << ")" << endl;
        }
    }
}

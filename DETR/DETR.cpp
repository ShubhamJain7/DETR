#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace cv;

const wchar_t* model_path = L"C:/Users/dell/source/repos/DETR/models/DETRmodel.onnx";
const string image_path = "C:/Users/dell/source/repos/DETR/test.jpg";
const float conf_threshold = 0.75;

Mat preprocess_image(Mat image) {
    // convert image values from int to float
    image.convertTo(image, CV_32FC3);
    // Change image format from BGR to RGB
    cvtColor(image, image, COLOR_BGR2RGB);

    Mat image_resized;
    // resize image to (256x256) to fit model input dimensions
    resize(image, image_resized, Size(256, 256));

    // normalize image (values between 0-1)
    Mat image_float;
    image_resized.convertTo(image_float, CV_32FC3, 1.0f / 255.0f, 0);

    // split image channels
    vector<cv::Mat> channels(3);
    split(image_float, channels);

    // define mean and std-dev for each channel
    vector<double> mean = { 0.485, 0.456, 0.406 };
    vector<double> stddev = { 0.229, 0.224, 0.225 };
    size_t i = 0;
    // normalize each channel with corresponding mean and std-dev values
    for (auto& c : channels) {
        c = (c - mean[i]) / stddev[i];
        ++i;
    }

    // concatenate channels to change format from HWC to CHW
    Mat image_normalized;
    vconcat(channels, image_normalized);

    return image_normalized;
}

int main()
{
    // load image to process
    Mat im;
    im = imread(image_path, IMREAD_COLOR);
    // if image is an empty matrix
    if (!im.data)
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // get image height and width
    int height = im.rows;
    int width = im.cols;

    // get processed image
    Mat image = preprocess_image(im);

    // create ONNX env and sessionOptions objects
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

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
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(image.data), input_tensor_size, input_node_dims.data(), 4);

    // pass inputs through model and get output
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 2);

    // get pointers to outputs
    auto scores = output_tensors[0].GetTensorMutableData<float>();
    auto bboxes = output_tensors[1].GetTensorMutableData<float>();

    // get lengths of outputs
    size_t probs_len = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    size_t boxes_len = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();

    // store outputs in a 2d array for easier access and processing
    // calculate and store sum of exponents for each row to be used as denominator for apploying the softmax function
    float probs[100][92];
    float boxes[100][4];
    size_t probs_index = 0;
    size_t boxes_index = 0;
    float denominators[100];
    while (probs_index < probs_len) {
        for (size_t i = 0; i < 100; i++) {
            float val = 0;
            for (size_t j = 0; j < 92; j++) {
                probs[i][j] = scores[probs_index];
                val += exp(probs[i][j]);
                ++probs_index;

                if (boxes_index < boxes_len && j < 4)  {
                    boxes[i][j] = bboxes[boxes_index];
                    ++boxes_index;
                }
            }
            denominators[i] = val;
        }
    }
    
    // calculate softmax of each item(row-wise) by didving exponent of item by sum of exponents
    // ignore 92nd column as it isn't required
    // find the highest probablility, it's index in the row and the corresponding bounding box
    vector<int> class_ids;
    vector<float> probabilities;
    vector<array<int, 4>> bounding_boxes;
    for (size_t i = 0; i < 100; i++) {
        float max_prob = 0;
        int id = -1;
        for (size_t j = 0; j < 91; j++) {
            float val = exp(probs[i][j])/denominators[i];
            if (val >= max_prob) {
                max_prob = val;
                id = j;
            }
        }
        // filter outputs
        if (max_prob > conf_threshold) {
            class_ids.push_back(id);
            probabilities.push_back(max_prob);

            array<float, 4> box;
            for (size_t k = 0; k < 4; k++) {
                box[k] = boxes[i][k];
            }

            // convert bounding box from cx-w-cy-h format to x1y1x2y2 format and scale to original image dimensions
            array<int, 4> scaled_box = {
                (int)((box[0] - 0.5 * box[2]) * width),
                (int)((box[1] - 0.5 * box[3]) * height),
                (int)((box[0] + 0.5 * box[2]) * width),
                (int)((box[1] + 0.5 * box[3]) * height)
            };
            bounding_boxes.push_back(scaled_box);
        }
    }

    // display final results
    for (size_t i = 0; i < class_ids.size(); i++) {
        cout << class_ids[i] << "(" << probabilities[i] << "): [";
        for (size_t j = 0; j < 4; j++) {
            cout << bounding_boxes[i][j] << ",";
        }
        cout << "]\n";
    }
}

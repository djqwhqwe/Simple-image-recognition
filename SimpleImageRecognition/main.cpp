#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;  //Для упрощения

void get_svm(const Ptr<SVM>& svm, vector<float>& svmDetector) {
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);
    CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
    CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
        (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    CV_Assert(sv.type() == CV_32F);
    svmDetector.clear();
    svmDetector.resize(sv.cols + 1);
    memcpy(&svmDetector[0], sv.ptr(), sv.cols * sizeof(svmDetector[0]));
    svmDetector[sv.cols] = (float)-rho;
}

bool isImage(const filesystem::path& path) {
    string extension = path.extension().string();
    transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return (extension == ".jpg" || extension == ".jpeg" || extension == ".png")
        ? true
        : false;
}

int main() {
    string path = "./testdata";

    if (!filesystem::exists(path)) {
        cerr << "Directory does not exist" << endl;
        return -1;
    }
    if (!filesystem::is_directory(path)) {
        cerr << "Not a directory" << endl;
        return -1;
    }
    if (filesystem::is_empty(path)) {
        cerr << "Directory is empty" << endl;
        return -1;
    }
    if ((filesystem::status(path).permissions() & filesystem::perms::owner_all) ==
        filesystem::perms::none) {
        cerr << "No read or write permissions" << endl;
        return -1;
    }
    for (const auto& entry : filesystem::directory_iterator(path)) {
        Mat img;

        Ptr<SVM> svm = StatModel::load<SVM>("vehicle_detector.yml");

        HOGDescriptor hog;

        hog.winSize = Size(40, 40);

        vector<Rect> detections;

        vector<float> svmDetector;

        string path = entry.path().string();

        if (!isImage(entry)) continue;

        img = imread(path);

        if (img.empty()) continue;

        get_svm(svm, svmDetector);

        hog.setSVMDetector(svmDetector);

        detections.clear();

        hog.detectMultiScale(img, detections);

        for (auto& detection : detections) {
            rectangle(img, detection.tl(), detection.br(), Scalar(0, 0, 255), 1);
            putText(img, "CAR", detection.tl(), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 255), 1);
        }

        hog.winSize = Size(64, 128);

        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

        detections.clear();

        hog.detectMultiScale(img, detections, 0, Size(8, 8), Size(32, 32), 1.2, 2);

        for (auto& detection : detections) {
            cv::rectangle(img, detection.tl(), detection.br(), Scalar(0, 0, 255), 1);
            putText(img, "PERSON", detection.tl(), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 255), 1);
        }

        imwrite(path + "_copy" + entry.path().extension().string(), img);
    }
    return 0;
}
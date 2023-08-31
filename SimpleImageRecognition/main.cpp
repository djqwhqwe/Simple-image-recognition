#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;  //Для упрощения

bool isImage(const filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        ::tolower);
    return (extension == ".jpg" || extension == ".jpeg" || extension == ".png")
        ? true
        : false;
}

int main(int argc, char** argv) {
    string path = "D:/work/opencvtask";

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

    CascadeClassifier car_classifier, person_classifier;

    if (!car_classifier.load(
        "C:/opencv/sources/data/haarcascades/haarcascade_car.xml")) {
        cerr << "OpenCV model for cars recognition is not found.";
        return -1;
    }
    if (!person_classifier.load(
        "C:/opencv/sources/data/haarcascades/haarcascade_body.xml")) {
        cerr << "OpenCV model for a person recognition is not found.";
        return -1;
    }
    for (const auto& entry : filesystem::directory_iterator(path)) {
        vector<Rect> cars_rect;

        vector<Rect> person_rect;

        string path = entry.path().string();

        if (!isImage(entry)) continue;

        Mat img = imread(path);

        if (img.empty()) continue;

        car_classifier.detectMultiScale(img, cars_rect, 1.1, 10);

        person_classifier.detectMultiScale(img, person_rect, 1.1, 10);

        for (int i = 0; i < cars_rect.size(); i++) {
            rectangle(img, cars_rect[i].tl(), cars_rect[i].br(), Scalar(0, 0, 255),
                3);
            putText(img, "CAR", cars_rect[i].tl(), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 255), 2);
        }

        for (int i = 0; i < person_rect.size(); i++) {
            rectangle(img, person_rect[i].tl(), person_rect[i].br(),
                Scalar(0, 0, 255), 3);
            putText(img, "PERSON", cars_rect[i].tl(), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 255), 2);
        }
    }

    return 0;
}
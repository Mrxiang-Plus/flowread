//
// Created by xiang on 2019/12/12.
//

/*
 * 通过光流图像进行单点位置更新
 * 两幅图
 */
#include "flowIOOpenCVWrapper.h"

using namespace std;
float pointx,pointy;
string flow_path,image1_path,image2_path;

void getParameter(){
    cv::FileStorage fs("../config/config.yaml",cv::FileStorage::READ);
    if (!fs.isOpened()){
        cout <<"No file!" << endl;
        return;
    }
    pointx = (double)fs["pointx"];
    pointy = (double)fs["pointy"];
    flow_path = (string)fs["flow_path"];
    image1_path = (string)fs["image1_path"];
    image2_path = (string)fs["image2_path"];

}
int main(int argc, char** argv) {
    if (argc != 1) {
        cout << "wrong input\nusage : cli " << endl;
        return 1;
    }
    getParameter();
    cv::Point2f  point1(pointx,pointy);
    cv::Mat image1 = cv::imread(image1_path);
    cv::Mat image2 = cv::imread(image2_path);

    cv::Mat flow = FlowIOOpenCVWrapper::read(flow_path);
    cv::Point2f point2 = point1 + flow.at<cv::Point2f>(point1.x,point1.y);
    cout<<point1<<" "<<point2<<endl;

    cv::circle(image1,point1,3,cv::Scalar(0, 255, 0), -1, 8);
    cv::circle(image2,point2,3,cv::Scalar(0, 255, 0), -1, 8);

    cv::imshow("image1",image1);
    cv::imshow("image2",image2);

    cv::waitKey(0);
    return 0;
}

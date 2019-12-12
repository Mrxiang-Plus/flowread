//
// Created by xiang on 2019/12/12.
//

/*
 * 通过光流图像进行特征点位置更新
 * 多点
 */
#include "flowIOOpenCVWrapper.h"

using namespace std;

string flow_path,image1_path,image2_path;

void getParameter(){
    cv::FileStorage fs("/home/xiang/Desktop/flow-io-opencv-master/config/config.yaml",cv::FileStorage::READ);
    if (!fs.isOpened()){
        cout <<"No file!" << endl;
        return;
    }

    flow_path = (string)fs["flow_path"];
    image1_path = (string)fs["image1_path"];
    image2_path = (string)fs["image2_path"];

}
bool  compare(cv::KeyPoint a,cv::KeyPoint b){
    return a.response > b.response;
}
void minDistance(cv::Mat image, vector<cv::Point2f> &points, int minDistance,int maxCorners){
    size_t i, j, total = points.size(), ncorners = 0;
    vector<cv::Point2f> corners;
    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( int i = 0; i < total; i++ )
        {
            int y = (int)points[i].y;
            int x = (int)points[i].x;

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <cv::Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            int y = (int)points[i].y;
            int x = (int)points[i].x;

            corners.push_back(cv::Point2f((float)x, (float)y));
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }
    points.clear();
    points = corners;
}
void getPoints(cv::Mat image,std::vector<cv::Point2f> &points,cv::InputArray mask,int maxConer){
    std::vector<cv::KeyPoint> points_Fast ; //暂时存储提取的点
    //Fast检测器
    cv::Ptr<cv::FastFeatureDetector> FastDetector = cv::FastFeatureDetector::create(20, true);
    FastDetector -> detect(image, points_Fast,mask);


    /*决定导入点的数量
     * maxConer > 0,按照maxConer导入
     * maxConer = 0,不导入
     * maxConer < 0,无限制
     */
    vector<cv::Point2f> points_temp;
    for (int i = 0; i < points_Fast.size(); ++i) {
        points_temp.push_back(points_Fast[i].pt);
    }

    minDistance(image, points_temp,30,maxConer);
    if (maxConer > 0){
        for (int i = 0; i < maxConer && i< points_temp.size(); ++i) {
            points.push_back(points_temp[i]);
        }
    }
    if (maxConer < 0){
        for (auto kp:points_temp)
            points.push_back(kp);
    }
}


int main(int argc, char** argv) {
    if (argc != 1) {
        cout << "wrong input\nusage : cli " << endl;
        return 1;
    }
    getParameter();
    vector<cv::Point2f> points;

    cv::Mat image1 = cv::imread(image1_path);
    cv::Mat image2 = cv::imread(image2_path);
    cv::Mat flow = FlowIOOpenCVWrapper::read(flow_path);
    cout << "1" << endl;
    getPoints(image1,points,cv::noArray(),30);
    cout << "2" << endl;

    for (int i = 0; i < points.size(); ++i) {
        cv::circle(image1,points[i],3,cv::Scalar(0, 255, 0), -1, 8);
        cout << "3" << endl;
        cv::Point2f point_up = flow.at<cv::Vec2f>(points[i].x,points[i].y);
        cv::Point2f point2 = points[i] + point_up;
        if(point2.x >= 0 && point2.y >= 0 && point2.x <= 1024 && point2.y <= 436){
            cv::circle(image2,points[i],3,cv::Scalar(0, 255, 0), -1, 8);
            cout << "4" << endl;}

    }


    cv::imshow("image1",image1);
    cv::imshow("image2",image2);

    cv::waitKey(0);
    return 0;
}

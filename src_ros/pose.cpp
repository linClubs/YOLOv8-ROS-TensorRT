//
// Created by ubuntu on 4/7/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "pose/yolov8-pose.hpp"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>


const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6, 12},
                                                         {7, 13},
                                                         {6, 7},
                                                         {6, 8},
                                                         {7, 9},
                                                         {8, 10},
                                                         {9, 11},
                                                         {2, 3},
                                                         {1, 2},
                                                         {1, 3},
                                                         {2, 4},
                                                         {3, 5},
                                                         {4, 6},
                                                         {5, 7}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0}};


class RosNode
{
public:
    RosNode();
    ~RosNode(){};
    void callback(const sensor_msgs::ImageConstPtr &msg);

private:
    std::string pkg_path_, engine_file_path_;
    std::shared_ptr<YOLOv8_pose> yolov8_pose_;
    
    cv::Mat  img_res_, image_;
    cv::Size size_        = cv::Size{640, 640};
    int      topk_        = 100;
    float    score_thres_ = 0.25f;
    float    iou_thres_   = 0.65f;
    std::vector<Object> objs_;

    ros::NodeHandle n_;
    ros::Subscriber sub_img_;
    ros::Publisher pub_img_;
    std::string topic_img_;
    std::string topic_res_img_;
    std::string weight_name_;
};

RosNode::RosNode()
{
    cudaSetDevice(0);
    pkg_path_ = ros::package::getPath("yolov8_trt");
    
    n_.param<std::string>("topic_img", topic_img_, "/camera/color/image_raw");
    n_.param<std::string>("topic_res_img", topic_res_img_, "/pose/image_raw");
    n_.param<std::string>("weight_name",  weight_name_, "yolov8n-pose.engine");
        
    engine_file_path_ = pkg_path_ + "/weights/" + weight_name_;


    std::cout << "\n\033[1;32m--engine_file_path: " << engine_file_path_ << "\033[0m" << std::endl;
    std::cout << "\033[1;32m" << "--topic_img       : " << topic_img_  << "\033[0m" << std::endl;
    std::cout << "\033[1;32m--topic_res_img   : " << topic_res_img_    << "\n\033[0m" << std::endl;

    yolov8_pose_.reset(new YOLOv8_pose(engine_file_path_));
    yolov8_pose_->make_pipe(true);

    pub_img_ = n_.advertise<sensor_msgs::Image>(topic_res_img_, 10);
    sub_img_ = n_.subscribe(topic_img_, 10, &RosNode::callback, this);
}

void RosNode::callback(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
    objs_.clear();
    yolov8_pose_->copy_from_Mat(image, size_);
    auto start = std::chrono::system_clock::now();
    yolov8_pose_->infer();
    auto end = std::chrono::system_clock::now();
    yolov8_pose_->postprocess(objs_, score_thres_, iou_thres_, topk_);
    yolov8_pose_->draw_objects(image, img_res_, objs_, SKELETON, KPS_COLORS, LIMB_COLORS);
    
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    cv::putText(img_res_, "fps: " + std::to_string(int(1000/tc)) , cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, 8);
    ROS_INFO("pose cost %2.4lf ms", tc);



    sensor_msgs::ImagePtr msg_img_new;
    msg_img_new = cv_bridge::CvImage(std_msgs::Header(),"bgr8",img_res_).toImageMsg();
	pub_img_.publish(msg_img_new);


}
int main(int argc, char** argv)
{
    
    ros::init(argc, argv, "seg_node");
    ros::NodeHandle n;
    auto pose_node = std::make_shared<RosNode>();
    ros::spin();
    return 0;
    return 0;
}

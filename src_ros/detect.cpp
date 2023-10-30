//
// Created by ubuntu on 1/20/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "detect/yolov8.hpp"

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};


class RosNode
{
public:
    RosNode();
    ~RosNode(){};
    
    void callback(const sensor_msgs::ImageConstPtr &msg);

private:
    std::string pkg_path_, engine_file_path_;
    std::shared_ptr<YOLOv8> yolov8_;
    cv::Mat img_res_, image_;
    cv::Size size_ = cv::Size{640, 640};
    std::vector<Object> objs_;

  
    int      num_labels_  = 80;
    int      topk_        = 100;
    float    score_thres_ = 0.25f;
    float    iou_thres_   = 0.65f;

    ros::NodeHandle n_;
    ros::Subscriber sub_img_;
    ros::Publisher pub_img_;
    std::string topic_img_;
    std::string topic_res_img_, weight_name_;
};

RosNode::RosNode()
{   
    cudaSetDevice(0);
    pkg_path_ = ros::package::getPath("yolov8_trt");
    
    
    n_.param<std::string>("topic_img", topic_img_, "/camera/color/image_raw");
    n_.param<std::string>("topic_res_img", topic_res_img_, "/detect/image_raw");
    n_.param<std::string>("weight_name", weight_name_, "yolov8n.engine");
    engine_file_path_ = pkg_path_ + "/weights/" + weight_name_;

    std::cout << "\n\033[1;32m--engine_file_path: " << engine_file_path_ << "\033[0m" << std::endl;
    std::cout << "\033[1;32m" << "--topic_img       : " << topic_img_  << "\033[0m" << std::endl;
    std::cout << "\033[1;32m--topic_res_img   : " << topic_res_img_    << "\n\033[0m" << std::endl;

    
    yolov8_.reset(new YOLOv8(engine_file_path_));
    yolov8_->make_pipe(true);

    pub_img_ = n_.advertise<sensor_msgs::Image>(topic_res_img_, 10);
    sub_img_ = n_.subscribe(topic_img_, 10, &RosNode::callback, this);

}

void RosNode::callback(const sensor_msgs::ImageConstPtr &msg)
{   
    objs_.clear();
    cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
    objs_.clear();
    yolov8_->copy_from_Mat(image, size_);
    auto start = std::chrono::system_clock::now();
    yolov8_->infer();
    auto end = std::chrono::system_clock::now();
    
    // 默认版本
    yolov8_->postprocess(objs_);
    yolov8_->draw_objects(image, img_res_, objs_, CLASS_NAMES, COLORS);
    

    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;    
    cv::putText(img_res_, "fps: " + std::to_string(int(1000/tc)) , cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8);
    ROS_INFO("detect cost %2.4f ms", tc);

    sensor_msgs::ImagePtr msg_img_new;
    msg_img_new = cv_bridge::CvImage(std_msgs::Header(),"bgr8",img_res_).toImageMsg();
	pub_img_.publish(msg_img_new);


}


int main(int argc, char** argv)
{   
    ros::init(argc, argv, "det_node");
    ros::NodeHandle n;
    auto det_node = std::make_shared<RosNode>();
    ros::spin();
    return 0;
}

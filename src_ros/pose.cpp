//
// Created by ubuntu on 4/7/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "pose/yolov8-pose.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <ament_index_cpp/get_package_prefix.hpp>
#include <cv_bridge/cv_bridge.h>



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


class RosNode : public rclcpp::Node
{
public:
    RosNode();
    ~RosNode(){};
    void callback(const sensor_msgs::msg::Image::SharedPtr msg);

private:
    std::string pkg_path_, engine_file_path_;
    std::shared_ptr<YOLOv8_pose> yolov8_pose_;
    
    cv::Mat  img_res_, image_;
    cv::Size size_        = cv::Size{640, 640};
    int      topk_        = 100;
    float    score_thres_ = 0.25f;
    float    iou_thres_   = 0.65f;
    std::vector<Object> objs_;

    std::string topic_img_;
    std::string topic_res_img_;
    std::string weight_name_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_;
};

RosNode::RosNode() : Node("pose_node")
{
    cudaSetDevice(0);
    // weight_name_ = "yolov8n-pose.engine";
    // topic_img_ = "/camera/color/image_raw";
    // topic_res_img_ = "/pose/image_raw";

    this->declare_parameter("topic_img", "/camera/color/image_raw");
    this->get_parameter("topic_img", topic_img_);

    this->declare_parameter("topic_res_img", "/pose/image_raw");
    this->get_parameter("topic_res_img", topic_res_img_);

    this->declare_parameter("weight_name", "yolov8n-pose.engine");
    this->get_parameter("weight_name", weight_name_);


    auto package_prefix = ament_index_cpp::get_package_prefix("yolov8_trt");
    engine_file_path_ = package_prefix + "/../../src/YOLOv8-ROS-TensorRT/weights/" + weight_name_;

    std::cout << "\n\033[1;32m--engine_file_path: " << engine_file_path_ << "\033[0m" << std::endl;
    std::cout << "\033[1;32m" << "--topic_img       : " << topic_img_  << "\033[0m" << std::endl;
    std::cout << "\033[1;32m--topic_res_img   : " << topic_res_img_    << "\n\033[0m" << std::endl;

    yolov8_pose_.reset(new YOLOv8_pose(engine_file_path_));
    yolov8_pose_->make_pipe(true);

   pub_img_ = this->create_publisher<sensor_msgs::msg::Image>(topic_res_img_, 10);
   sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(topic_img_, 10, std::bind(&RosNode::callback, this, std::placeholders::_1));
}

void RosNode::callback(const sensor_msgs::msg::Image::SharedPtr msg)
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
    RCLCPP_INFO(this->get_logger(), "pose cost %.3lf ms", tc);



    sensor_msgs::msg::Image::SharedPtr msg_img_new;
    msg_img_new = cv_bridge::CvImage(std_msgs::msg::Header(),"bgr8",img_res_).toImageMsg();
	pub_img_->publish(*msg_img_new);


}

int main(int argc, char** argv)
{
    
    rclcpp::init(argc, argv);
    auto pose_node = std::make_shared<RosNode>();
    rclcpp::spin(pose_node);
    rclcpp::shutdown();
    return 0;
}

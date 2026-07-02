#ifndef INCLUDED_crop_recognition_node_h_
#define INCLUDED_crop_recognition_node_h_

//各インクルード(折りたたんでいるので注意)
#include <ros/ros.h>
#include <ros/package.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <numeric>
#include <image_transport/image_transport.h>


/******************[マクロ変数的な定義]*****************/
#define MIN_HSV_MASK cv::Scalar(35, 100, 50)  // 緑色の範囲
#define MAX_HSV_MASK cv::Scalar(85, 255, 255) // 緑色の範囲
#define KERNEL_SIZE 5 // モルフォロジー処理のカーネルサイズ

//以下杉山コード
#define MIN_HSV_MASK cv::Scalar(0, 0, 143)
#define MAX_HSV_MASK cv::Scalar(245, 43, 255)
//１回目の検出用パラメータ
#define MIN_AREA_1 1150
#define MAX_AREA_1 4000
#define ROUNDNESS_1 0.2
//2回目の検出用パラメータ
#define MIN_AREA_2 750
#define MAX_AREA_2 1700
#define ROUNDNESS_2 0.55

//検出範囲の限定
#define y_min 350//350
#define y_max 450//600
#define x_min 700//500//500
#define x_max 800//600//800
/*******************************************************/

//CropRecognitionクラスの定義
class CropRecognition{
   private:
    //おまじない
    ros::NodeHandle nh;

    /***各パブリッシャー***/
   // 茎の座標を送信するパブリッシャー
    ros::Publisher stalk_position_pub;

    //画像の送信（デバッグ用）
    ros::Publisher rs_color_pub;

    /***各サブスクライバー***/
    // 画像の受信
    ros::Subscriber rs_color_sub;

    /***各変数***/
    // 画像の変数
    cv_bridge::CvImagePtr cv_color_img;

   
   public:
    //デフォルトコンストラクタの宣言
    CropRecognition();

    //各処理の宣言
    //imageCallback関数の宣言
    void imageCallback(const sensor_msgs::ImageConstPtr& image_color);
};

/*********************************************************************************
 * 関数名: CropRecognitionNode::CropRecognitionNode
 * 引数: なし
 * 返り値: なし
 * 処理内容: CropRecognitionNodeクラスのコンストラクタ
 * ******************************************************************************/
CropRecognition::CropRecognition() {
    // パブリッシャーの宣言
    stalk_position_pub = nh.advertise<geometry_msgs::Point>("/stalk_position", 10);
    rs_color_pub = nh.advertise<sensor_msgs::Image>("/crop_detect_image", 10);

    // カラー画像を受け取るためのサブスクライバー
    rs_color_sub = nh.subscribe("/camera/color/image_raw", 10, &CropRecognition::imageCallback, this);
    //カラー画像を受け取るためのサブスクライバー(gazebo)
   //  rs_color_sub=nh.subscribe("/camera_tool/color/image_raw", 10, &SeedDetector::imageCallback, this);
}

/*********************************************************************************
 * 関数名: CropRecognitionNode::imageCallback
 * 引数: const sensor_msgs::ImageConstPtr& image_color
 * 返り値: なし
 * 処理内容: 画像をサブスクライブしてから、茎の位置を推定し、座標をパブリッシュする
 * ******************************************************************************/
void CropRecognitionNode::imageCallback(const sensor_msgs::ImageConstPtr& image_color) {
    ROS_INFO("Detecting Crop and Stalk !!");

    // カラー画像を変数に格納
    try {
        cv_color_img = cv_bridge::toCvCopy(image_color, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& ex) {
        ROS_ERROR("cv_bridge exception: %s", ex.what());
        return;
    }

    // 画像をHSVに変換
    cv::Mat img_hsv;
    cv::cvtColor(cv_color_img->image, img_hsv, cv::COLOR_BGR2HSV);

    // 緑色のマスクを作成
    cv::Mat green_mask;
    cv::inRange(img_hsv, MIN_HSV_MASK, MAX_HSV_MASK, green_mask);

    // モルフォロジー処理用のカーネルを作成
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(KERNEL_SIZE, KERNEL_SIZE));

    // クロージング処理
    cv::Mat closed_mask;
    cv::morphologyEx(green_mask, cv::MORPH_CLOSE, closed_mask, kernel);

    // オープニング処理
    cv::Mat opened_mask;
    cv::morphologyEx(closed_mask, cv::MORPH_OPEN, opened_mask, kernel);

    // マスクを緑色で塗りつぶす（デバッグ用）
    cv::Mat mask_colored = cv::Mat::zeros(opened_mask.size(), CV_8UC3);
    mask_colored.setTo(cv::Scalar(0, 255, 0), opened_mask);

    // 輪郭の検出
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 輪郭ごとに重心を計算し、茎の位置を推定
    for (const auto& contour : contours) {
        cv::Moments M = cv::moments(contour);

        if (M.m00 != 0) {  // 面積が0でない場合
            int cX = static_cast<int>(M.m10 / M.m00);  // 重心のx座標
            int cY = static_cast<int>(M.m01 / M.m00);  // 重心のy座標

            // パブリッシャー用の座標データを作成
            geometry_msgs::Point stalk_position;
            stalk_position.x = cX;
            stalk_position.y = cY;
            stalk_position.z = 0;

            // 茎の座標をパブリッシュ
            stalk_position_pub.publish(stalk_position);

            // デバッグ用に画像に重心を表示
            cv::circle(cv_color_img->image, cv::Point(cX, cY), 5, cv::Scalar(0, 255, 0), -1);
        }
    }

    // デバッグ用の画像をパブリッシュ
    rs_color_pub.publish(cv_color_img->toImageMsg());
}

#endif // INCLUDED_crop_recognition_node_h_

//以降杉山プログラム

/*********************************************************************************
 * 関数名:SeedDetector::iImageCallback
 * 引数: const sensor_msgs::ImageConstPtr& image_color
 * 返り値: なし
 * 処理内容: 画像をサブスクライブしてから、種子団子の有無をパブリッシュする
 * ******************************************************************************/
void SeedDetector::imageCallback(const sensor_msgs::ImageConstPtr& image_color){
   ROS_INFO("Detecting Seed !!");
   //カラー画像の変数
   cv_bridge::CvImagePtr cv_color_img;

   //ROSメッセージをcv_bridgeを使ってopencvの画像に変換
   try{
      cv_color_img = cv_bridge::toCvCopy(image_color, sensor_msgs::image_encodings::BGR8);
   }
   catch(cv_bridge::Exception& ex){
      ROS_ERROR("cv_bridge exception: %s", ex.what());
   }

   // cv_color_imgを表示
   // cv::imshow("color_image", cv_color_img->image);

   // //背景差分する範囲を指定 0807in
   cv::Mat roi_image = cv::Mat(cv_color_img->image, cv::Rect(x_min,y_min,x_max-x_min,y_max-y_min));
   //上記の範囲に赤枠0807in
   cv::rectangle(cv_color_img->image, cv::Point(x_min,y_min), cv::Point(x_max,y_max), cv::Scalar(0,0,255), 5, 8, 0);

   //HSV色空間への変換
   cv::Mat img_hsv;
   cv::cvtColor(roi_image,img_hsv,cv::COLOR_BGR2HSV_FULL);

   //マスクを作成
   cv::Mat img_mask;
   cv::inRange(img_hsv, MIN_HSV_MASK,MAX_HSV_MASK,img_mask);
   //輪郭の検出
   std::vector<std::vector<cv::Point>> contours;//検出下輪郭が入る変数
   std::vector<cv::Vec4i> hierachy; //検出した輪郭の階層構造の変数

   //-----------------------------------1回目の検出-------------------------------------------------
   cv::findContours(img_mask, contours, hierachy, cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
   //サイズの絞り込み
   std::vector<std::vector<cv::Point>> new_contours;
   for(auto c: contours){
      double area = cv::contourArea(c);
      if( area > MIN_AREA_1 && area < MAX_AREA_1){
         new_contours.push_back(c);
      }
   }
   //円形度による絞り込み
   std::vector<cv::RotatedRect> circles;
   for(auto c: new_contours){
      double area = cv::contourArea(c);
      double length = cv::arcLength(c,true);
      if(length != 0){
         double roundness = 4*M_PI*area/length/length;
         if(roundness > ROUNDNESS_1){
            cv::RotatedRect ellipse = cv::fitEllipse(c);
            circles.push_back(ellipse);
         }
      }
   }
   //-----------------------------------------------------------------------------------------------

   //-----------------------------------2回目の検出-------------------------------------------------
   if( circles.size() == 0 ){
      cv::Mat kernel=cv::Mat::ones(cv::Size(5, 5), CV_8U); //走査カーネルのサイズを規定
      //膨張・収縮の画像変数
      cv::Mat erosion;
      cv::Mat dilation;
      //モルフォロジー演算処理もどき
      cv::erode(img_mask, erosion, kernel, cv::Point(-1,-1), 1);
      cv::erode(erosion, erosion, kernel, cv::Point(-1,-1), 1);
      cv::dilate(erosion, dilation, kernel, cv::Point(-1,-1), 1);
      cv::dilate(dilation, dilation, kernel, cv::Point(-1,-1), 1);
      cv::dilate(dilation, dilation, kernel, cv::Point(-1,-1), 1);
      cv::erode(dilation, erosion, kernel, cv::Point(-1,-1), 1);

      //検出
      cv::findContours(img_mask, contours, hierachy, cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
      //サイズの絞り込み
      std::vector<std::vector<cv::Point>> new_contours;
      for(auto c: contours){
         double area = cv::contourArea(c);
         if( area > MIN_AREA_2 && area < MAX_AREA_2){
            new_contours.push_back(c);
         }
      }
      //円形度による絞り込み
      std::vector<cv::RotatedRect> circles;
      for(auto c: new_contours){
         double area = cv::contourArea(c);
         double length = cv::arcLength(c,true);
         if(length != 0){
            double roundness = 4*M_PI*area/length/length;
            if(roundness > ROUNDNESS_2){
               cv::RotatedRect ellipse = cv::fitEllipse(c);
               circles.push_back(ellipse);
            }
         }
      }
   }
   //---------------------------------------------------------------------------------------------------
   
   //パブリッシュするデータの作成
    if(circles.size() >= 1 ){
         result.data = true; //種子団子あり
         ROS_INFO("SUCCESS in Detection !!");
         ROS_INFO("\n");
    }
    else{
        result.data = false; //種子団子なし
        ROS_INFO("FAILED in Detection !!\n");
        ROS_INFO("\n");
    }
    //検出した楕円を描画
   for(auto c: circles){
      cv::ellipse(roi_image, c, cv::Scalar(0, 0, 255), 4, 8);
   }
   
    //パブリッシュ
    rs_color_pub.publish(cv_color_img->toImageMsg());

    //result_pub.publish(result);

};

/*********************************************************************************
 * 関数名:SeedDetector::detect_flagCallback
 * 引数: const std_msgs::BoolConstPtr& detect_flag
 * 返り値: なし
 * 処理内容: SeedDetectorクラスのコンストラクタ
 * ******************************************************************************/
void SeedDetector::detect_flagCallback(const std_msgs::BoolConstPtr& detect_flag){
   //もしdetect_flagがtrueなら
   if(detect_flag->data == true){
      result_pub.publish(result);
   }
}


#endif // INCLUDED_seed_test_node_h_

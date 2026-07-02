#include "crop_recognition_node.hpp"

// メイン関数
int main(int argc, char** argv) {
   // おまじない
   ros::init(argc, argv, "crop_recognition_node");
   ros::NodeHandle nh;    
   // 種子団子検出クラスの宣言
   CropRecognition c_r; 

   // ループの周期(Hz)
   ros::Rate loop_rate(100);
   
   while(ros::ok()){
   // メッセージコールバックの呼び出し
   ros::spinOnce();
   // ループのスリープ
   loop_rate.sleep();
   }
    
}
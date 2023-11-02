import matplotlib.pyplot as plt
import numpy as np
import rospy
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
import sys
import os
import cv2


       
      
def rosimg2cv(image):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

    return cv_image


      
def callback(rosimage: Image):
    
    #Original Image
    cv_image = rosimg2cv(rosimage)
    # window_name = 'Original image'
    # cv2.imshow(window_name, cv_image)
    
    # Grayscale
    cv_image_GS = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
    # window_name_GS = 'Greyscale image'
    # cv2.imshow(window_name_GS,cv_image_GS)
    
    # Binary
    ret, cv_image_b = cv2.threshold(cv_image_GS,127,255,cv2.THRESH_BINARY)
    
    # Plot
    titles = ['Original Image','Greyscale image','Binary image']
    images = [cv_image, cv_image_GS, cv_image_b]
    for i in range(3):
        plt.subplot(1,3,i+1),plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.imshow
    
    # plt.imshow(cv_image)  
    # plt.title('Original image')
    # plt.show()
    # rospy.loginfo(print('shape of cv_image is \n', cv_image.shape))


        
        
if __name__ == '__main__':
    rospy.init_node('tutorial_2', anonymous=True)
    # pub = rospy.Publisher('/pub_mask',Int16MultiArray, queue_size=1000)
    sub = rospy.Subscriber('/RAW IMAGE NAO',Image,callback) # TODO: find image topic from NAO!
    rospy.loginfo('Node has been started.')

    rospy.spin()
    
    
    
'''
rostopic info /xtion/rgb/image_raw
Type: sensor_msgs/Image

Publishers: 
 * /gazebo (http://desktop:37407/)

Subscribers: 
 * /image_raw_to_rect_color_relay (http://desktop:45563/)
 * /darknet_ros (http://desktop:40941/)
 * /rviz (http://desktop:43473/)
 * /xtion/rgb/image_proc (http://desktop:37891/)

'''
#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
import cv2 as cv
from cv_bridge import CvBridge


class LocalizeUAV():
    """Use thermal markers to self-localize a UAV."""
    def __init__(self):
        rospy.init_node("localize_uav", anonymous=True)
        self.thermal_img_msg = Image()
        self.bridge = CvBridge()
        self.uav_position_msg = PoseStamped()
        self.img_width = 160
        self.img_height = 120
        self.uav_height = 1300.75
        self.HFOV = np.pi*57/180
        self.DFOV = np.pi*71/180
        self.VFOV = 2*np.arctan2(np.sqrt(np.square(np.tan(self.DFOV/2)) - np.square(np.tan(self.HFOV/2))),1)
        self.initial_x = None
        self.initial_y = None

        rospy.Subscriber("/kevin/camera/rgb/image_raw", Image, callback=self.thermal_img)
        self.thermal_img_pub = rospy.Publisher("/sentinel/thermal/rgb/image_raw", Image, queue_size=1)
        self.uav_position_pub = rospy.Publisher("/sentinel/position/global", PoseStamped, queue_size=1)
        
        rospy.spin()

    def thermal_img(self, msg):
        thermal_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv.cvtColor(thermal_img_cv, cv.COLOR_BGR2HSV)
        lower_thermal_limit = np.array([16, 24, 246])
        upper_thermal_limit = np.array([36, 255, 255])
        mask = cv.inRange(hsv, lower_thermal_limit, upper_thermal_limit)
        contours = cv.findContours(mask.copy(),
                                    cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) > 0:
            contours = sorted(contours, key=cv.contourArea, reverse=True)
            contour = contours[0]
            (xg, yg, wg, hg) = cv.boundingRect(contour)
            cv.rectangle(thermal_img_cv, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 1)
            center = [xg + wg / 2, yg + hg / 2]
            cv.circle(thermal_img_cv,(int(np.round(center[0])), int(np.round(center[1]))),1,(255,0,0),1)

            global_x, global_y = self.calulate_position(center)

            self.uav_position_msg.header.stamp = rospy.Time.now()
            self.uav_position_msg.header.frame_id = "map"
            self.uav_position_msg.pose.position.x = global_x
            self.uav_position_msg.pose.position.y = global_y
            self.uav_position_pub.publish(self.uav_position_msg)
            
        else:
            print("No marker.")

        self.thermal_img_msg.header.stamp = rospy.Time.now()
        self.thermal_img_msg.header.frame_id = "map"
        self.thermal_img_msg = self.bridge.cv2_to_imgmsg(thermal_img_cv, "bgr8")
        self.thermal_img_pub.publish(self.thermal_img_msg)

        print(f"X Position: {global_x}, Y Position: {global_y}")

    def calulate_position(self, center):
        apx = self.HFOV/self.img_width
        apy = self.VFOV/self.img_height
        marker_alt = self.uav_height

        delta_pixel_x = center[0] - self.img_width/2
        delta_pixel_y = center[1] - self.img_height/2
        # print(f"Delta pixel_x: {delta_pixel_x}, Delta pixel_y: {delta_pixel_y}")

        alpha = np.abs(delta_pixel_x)*apx
        beta = np.abs(delta_pixel_y)*apy
        deltax_img = np.sign(delta_pixel_x)*np.tan(alpha)*marker_alt
        deltay_img = np.sign(delta_pixel_y)*np.tan(beta)*marker_alt
        deltaS_img = np.sqrt(np.square(deltax_img) + np.square(deltay_img))
        theta_img = np.arctan2(deltay_img, deltax_img)

        # print(f"Theta: {theta_img}")

        self.current_deltaS = deltaS_img

        # Calculate the angle between the estimated position and the target position
        theta_horizontal = theta_img-np.pi/2 #Angle to the target in x-y plane
        theta_vertical = np.arctan2(self.current_deltaS, marker_alt) #Angle to the target in relative to straight down plane

        if self.initial_x is None and self.initial_y is None:
            self.initial_x = self.current_deltaS*np.cos(theta_horizontal)
            self.initial_y = self.current_deltaS*np.sin(theta_horizontal)

        global_x = self.initial_x - self.current_deltaS*np.cos(theta_horizontal)
        global_y = self.initial_y - self.current_deltaS*np.sin(theta_horizontal)

        return global_x, global_y


if __name__ == "__main__":
    try:
        LocalizeUAV()
    except rospy.ROSInterruptException:
        pass
    
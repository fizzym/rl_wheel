#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('rl_wheel')
import numpy as np
import rospy
import cv2
import sys

from operator import add
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    '''
    Constructor
    '''
    self.joint_angle_pub = rospy.Publisher(
       "/trough_bot/arm_trough_back_joint_position_controller/command",
        Float64, queue_size=1)
    self.ball_control_period_pub = rospy.Publisher("/ball_controller_period_sec",
                                                   Float64, queue_size=1)
    self.ball_P_term_pub = rospy.Publisher("/ball_P_term", Float64, queue_size=1)
    self.ball_D_term_pub = rospy.Publisher("/ball_D_term", Float64, queue_size=1)
    self.ball_error_pub = rospy.Publisher("/ball_error", Float64, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/trough_bot/camera1/image_raw",
                                      Image, self.camera_callback, queue_size=1)
    
    self.position_integral = 0
    self.ball_velocity = 0
    self.prev_ball_position = 0
    self.prev_time = 0.0
    self.callback_call = 0


  def camera_callback(self, data):
    '''
    '''
    # Calculate loop time
    cur_time = rospy.get_time()
    loop_time = cur_time - self.prev_time
    self.prev_time = cur_time
    
    if loop_time == 0:
       # Most likely Gazebo is paused and the plugin is still publishing data
       return

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    
    cv_image_blue = cv_image[:,:,2]
    ret, cv_image_masked = cv2.threshold(cv_image_blue, 10, 255, cv2.THRESH_BINARY_INV)

    # Calculate position of balls on x (weighted average position of mask)
    # Collapse mask on X
    vertical_sum = np.sum(cv_image_masked, axis=0)
    # Weighted average of collapsed mask on X
    ball_x_position = self.prev_ball_position
    if (sum(vertical_sum) > 0):
        ball_x_position = np.average(np.arange(cols), weights=vertical_sum)

    # PID on the ball position
    ball_setpoint = 400
    ball_P = 0.0005
    ball_I = 0.0
    ball_D = -0.0001

    position_error = ball_setpoint - ball_x_position
    self.position_integral = self.position_integral + position_error

    # Calculate ball velocity for the D term
    # To mitigate loop_time jitter low pass filter it by running an exponential decay
    # window on velocity.
    velocity_factor = 0.9 # 1 is only current velocity
    self.ball_velocity = ((1-velocity_factor) * self.ball_velocity + 
                          velocity_factor * (ball_x_position - self.prev_ball_position) / loop_time)
    self.prev_ball_position = ball_x_position

    P_term = ball_P * position_error
    I_term = ball_I * self.position_integral
    D_term = ball_D * self.ball_velocity

    output = P_term + I_term + D_term

    MAX_ANGLE = 0.15 # 0.05rad ~ 3deg
    # Clip the output to MAX_ANGLE
    angle = MAX_ANGLE if (output > MAX_ANGLE) else output
    angle = -MAX_ANGLE if (output < -MAX_ANGLE) else angle

    try:
        self.joint_angle_pub.publish(angle)
        self.ball_P_term_pub.publish(P_term)
        self.ball_D_term_pub.publish(D_term)
        self.ball_error_pub.publish(position_error)
        self.ball_control_period_pub.publish(loop_time)
    except CvBridgeError as e:
      print(e)

    print("Callback: {:03d} | Sim time (sec): {:.3f} | Loop period: {:.3f} | Error: {:.2f} | Angle: {:.2f}".
          format(self.callback_call, cur_time, loop_time, position_error, angle))

    if (True):
        # Display telemetry

        # Add position of ball as a circle
        circle_center = (int(ball_x_position), int(rows/2))
        circle_radius = 4
        circle_color = (0, 255, 0)
        circle_width = 2
        cv2.circle(cv_image, circle_center, circle_radius, circle_color, circle_width)
        # Add error, P, I, D terms along with commanded angle text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        font_start_pos = (10, 250)
        font_color = (0, 0, 255)
        font_line_type = cv2.LINE_AA
        text_array = [
           "Callback call: {:03d}".format(self.callback_call),
           "Error: {:.3f}".format(position_error),
            "P term: {:.3f}".format(P_term),
            "I term: {:.3f}".format(I_term),
            "D term: {:.3f}".format(D_term),
            "Output: {:.3f}".format(angle),
            "Loop period (s): {:.3f}".format(loop_time)]
        i = 0
        for text_value in text_array:
            # increment text font by 20 pixels for every entry (each entry one row)
            font_start_pos = list(map(add, font_start_pos, (0, 20)))
            # display row of text
            cv2.putText(cv_image, text_value, font_start_pos, font, 
                        font_scale, font_color, font_thickness, font_line_type)

        cv2.imshow("Raw image", cv_image)
        #cv2.imshow("Blue channel", cv_image_blue)
        #cv2.imshow("Masked", cv_image_masked)
        cv2.waitKey(3)
    
    self.callback_call += 1


def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
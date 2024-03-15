import rospy
import time
import sys
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from move_base_msgs.msg import MoveBaseActionResult

# 设置多个目标点
goal_list = [
    [3.079, -0.709, 0.000, 0.000, 0.000, 0.443, 0.897],
    [2.703, -5.321, 0.000, 0.000, 0.000, -0.405, 0.914],
    [4.689, -0.725, 0.000, 0.000, 0.000, 0.951, 0.309],
    [4.448, -5.476, 0.000, 0.000, 0.000, -0.403, 0.915],
    [0.945, -0.376, 0.000, 0.000, 0.000, -0.718, 0.696],
    [0.558, -5.256, 0.000, 0.000, 0.000, 0.977, -0.212],
    [0.756, -2.413, 0.000, 0.000, 0.000, 0.345, 0.939],
    [0.860, -0.577, 0.000, 0.000, 0.000, 0.943, 0.333],
    [-0.080, 0.080, 0.000, 0.000, 0.000,  1.000, 0.000]
]
rospy.init_node('pubposase')
turtle_vel_pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=1)

mypose=PoseStamped()
turtle_vel_pub.publish(mypose) #先发送一个空位置，试探一下，否则第一个包容易丢
time.sleep(1)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python3 one_goal.py <goal_index>")
    sys.exit(1)

  goal_index = int(sys.argv[1])
    
  mypose = PoseStamped()
  # 设置下一个目标点
  mypose.header.frame_id = 'map'
  mypose.pose.position.x = goal_list[goal_index][0]
  mypose.pose.position.y = goal_list[goal_index][1]
  mypose.pose.position.z = goal_list[goal_index][2]
  mypose.pose.orientation.x = goal_list[goal_index][3]
  mypose.pose.orientation.y = goal_list[goal_index][4]
  mypose.pose.orientation.z = goal_list[goal_index][5]
  mypose.pose.orientation.w = goal_list[goal_index][6]
	
  turtle_vel_pub.publish(mypose)

	
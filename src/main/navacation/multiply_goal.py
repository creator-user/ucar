# ps -ef | grep vino
# 
# 为了让小车经过多个导航点时不停留花时间来原地调整方向，可以在设置目标点时，将朝向角度设置为与前一个目标点相同。
# 这样小车就可以直接从前一个目标点到达下一个目标点，而不需要在目标点处调整方向。
# 具体来说，可以在每个目标点的朝向角度中设置与前一个目标点相同的值，例如在代码中的第1个目标点后，可以将第2个目标点的朝向角度设置为-0.009
# 与第1个目标点相同。同样的，可以在后续的目标点中重复这个步骤，以便小车可以顺利地到达每个目标点。

import rospy
import time
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from move_base_msgs.msg import MoveBaseActionResult

# 设置多个目标点
goal_list = [
    [2.567, -0.444, 0.000, 0.000, 0.000, -0.479, 0.878],
    [3.145, -0.543, 0.000, 0.000, 0.000, -0.942, 0.335],

    [2.410, -5.495, 0.000, 0.000, 0.000,  0.442, 0.897],
    [3.000, -5.561, 0.000, 0.000, 0.000,  0.887, 0.461],
    
    [5.376, -0.627, 0.000, 0.000, 0.000, -0.920, 0.393],
    [4.051, -0.463, 0.000, 0.000, 0.000, -0.418, 0.909],

    [4.778, -5.506, 0.000, 0.000, 0.000,  0.881, 0.474],
    [4.071, -5.665, 0.000, 0.000, 0.000,  0.421, 0.907],
    
    [0.880, -0.358, 0.000, 0.000, 0.000, -0.711, 0.703],

    [0.868, -2.262, 0.000, 0.000, 0.000, -0.716, 0.698],
    [0.756, -5.297, 0.000, 0.000, 0.000, -0.987, 0.159],
    [0.797, -2.474, 0.000, 0.000, 0.000,  0.910, 0.415],

    [0.878, -0.287, 0.000, 0.000, 0.000,  0.726, 0.688],
    [-0.080, 0.000, 0.000, 0.000, 0.000,  1.000, 0.000]
]

goal_index = 0

def callback(data):
    global goal_index
    if data.status.status == 3:
        # 到达E点，发布D点
        if goal_index < 14:

            # 设置下一个目标点D
            mypose = PoseStamped()
            mypose.header.frame_id = 'map'
            mypose.pose.position.x = goal_list[goal_index][0]
            mypose.pose.position.y = goal_list[goal_index][1]
            mypose.pose.position.z = goal_list[goal_index][2]
            mypose.pose.orientation.x = goal_list[goal_index][3]
            mypose.pose.orientation.y = goal_list[goal_index][4]
            mypose.pose.orientation.z = goal_list[goal_index][5]
            mypose.pose.orientation.w = goal_list[goal_index][6]
            turtle_vel_pub.publish(mypose)
            goal_index += 1

        else:
            rospy.loginfo("所有的目标点都到了!")

if __name__ == '__main__':
    # 创建一个发布器，用于发布机器人的移动命令
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    # 创建一个Twist消息实例
    twist = Twist()

    rospy.init_node('pubpose')
    turtle_vel_pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)
    rospy.Subscriber("/move_base/result", MoveBaseActionResult, callback)
    rospy.wait_for_service('/move_base/make_plan')

    mypose=PoseStamped()
    turtle_vel_pub.publish(mypose) #先发送一个空位置，试探一下，否则第一个包容易丢
    time.sleep(1)

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
    goal_index += 1

    rospy.spin()    

















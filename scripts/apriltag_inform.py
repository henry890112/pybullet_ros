#!/usr/bin/env python3
import rospy
import tf
from geometry_msgs.msg import TransformStamped
import numpy as np

def get_transform_matrix(trans, rot):
    # Create a 4x4 identity matrix
    transform_matrix = np.identity(4)
    
    # Create a 3x3 rotation matrix from the quaternion
    rotation_matrix = tf.transformations.quaternion_matrix(rot)[:3, :3]
    
    # Set the rotation part of the transformation matrix
    transform_matrix[:3, :3] = rotation_matrix
    
    # Set the translation part of the transformation matrix
    transform_matrix[:3, 3] = trans
    
    return transform_matrix

def get_transform(listener, target_frame, source_frame):
    try:
        # Wait for the transform to become available
        listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
        (trans, rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        return trans, rot
    except (tf.Exception, tf.ConnectivityException, tf.LookupException, tf.ExtrapolationException) as e:
        rospy.logerr("Error occurred while fetching transform: %s", e)
        return None, None

def main():
    rospy.init_node('tf_listener_node')

    listener = tf.TransformListener()

    target_frame = 'world'  # Replace with your target frame
    source_frame = 'D435_tag_0'

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        trans, rot = get_transform(listener, target_frame, source_frame)
        if trans and rot:
            rospy.loginfo("Position: x=%f, y=%f, z=%f", trans[0], trans[1], trans[2])
            rospy.loginfo("Orientation: x=%f, y=%f, z=%f, w=%f", rot[0], rot[1], rot[2], rot[3])

            transform_matrix = get_transform_matrix(trans, rot)
            rospy.loginfo("Transform matrix:\n%s", transform_matrix)

        rate.sleep()

if __name__ == '__main__':
    main()

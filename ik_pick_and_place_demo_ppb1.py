#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Inverse Kinematics Pick and Place Demo
"""
import argparse
import struct
import sys
import copy
from copy import deepcopy
import numpy as np
import numpy.matlib
from numpy import *
import math


import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface

class PickAndPlace(object):
    def __init__(self, limb, hover_distance = 0.15, verbose=True):
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, end_pose,starting_joint_angles,obstacle_pose=mat([[0.58875],[0.48375],[0.03]]),obstacle_radius=0.03):
        calibration=mat([[0],[0.01],[1.6]])
        obstacle_pose=obstacle_pose+calibration
        approach = deepcopy(end_pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        qmilestones=self.RRT(starting_joint_angles, joint_angles, obstacle_pose, obstacle_radius)
        keys = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
        for i in range(len(qmilestones)):
            qiDict=dict(zip(keys,np.asarray(qmilestones[i]).reshape(-1)))
            
            self._limb.move_to_joint_positions(qiDict)
            

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)

    def _servo_to_pose(self, pose):
        # servo down to release
        pose=deepcopy(pose)
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def pick(self, pose):
        # open the gripper
        starting_joint_angles=self._limb.joint_angles()
        self.gripper_open()
        # servo above pose
        self._approach(pose,starting_joint_angles)
        # servo to pose
        self._servo_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, container_pose):
        # servo above pose
        starting_joint_angles=self._limb.joint_angles()
        self._approach(container_pose,starting_joint_angles)
        # servo to pose
        self._servo_to_pose(container_pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()

    def RRT(self,starting_joint_angles, pose_angles, obstacle_pose, obstacle_radius):
        overhead_orientation = Quaternion(
                             x=-0.0249590815779,
                             y=0.999649402929,
                             z=0.00737916180073,
                             w=0.00486450832011)
        P1 = mat([[starting_joint_angles.get('left_s0'),
                   starting_joint_angles.get('left_s1'),
                   starting_joint_angles.get('left_e0'),
                   starting_joint_angles.get('left_e1'),
                   starting_joint_angles.get('left_w0'),
                   starting_joint_angles.get('left_w1'),
                   starting_joint_angles.get('left_w2')]])

        Pf = mat([[pose_angles.get('left_s0'),
                   pose_angles.get('left_s1'),
                   pose_angles.get('left_e0'),
                   pose_angles.get('left_e1'),
                   pose_angles.get('left_w0'),
                   pose_angles.get('left_w1'),
                   pose_angles.get('left_w2')]])
        if self.robotcollision(P1, obstacle_pose, obstacle_radius) == 1 or self.robotcollision(Pf, obstacle_pose,
                                                                                     obstacle_radius) == 1:
            print("starting point or end point in collision")
            exit(1)

        qMilestones = vstack((P1, Pf))

        iteration = 300
        qMax = mat([[0.5, -0.5, -0.5, 2.3, 1, 1.5, 0]])
        qMin = mat([[-0.5, -1.5, -1.5, 1, 0, 0.5, -1]])
        while self.checkedge(qMilestones[-1, :], qMilestones[-2, :], obstacle_pose, obstacle_radius) == 1:

            Pt_set = np.matlib.repmat(qMin, iteration, 1) + multiply(np.matlib.repmat((qMax - qMin), iteration, 1),np.random.uniform(0, 1, [iteration, 7]))        
            useless=[]
            for i in range(len(Pt_set)):
                posei=self.fk_request(Pt_set[i,:],4)[0:3,3]
                Bool=self.ik_request(Pose(position=Point(x=posei[0,0],
                                                     y=posei[1,0]-0.01,
                                                     z=posei[2,0]-1.6),
                                      orientation=overhead_orientation))
                if Bool == False :
                    useless.append(i)
                    
            Pt_set=delete(Pt_set,useless,axis=0)
            Pt_sum = vstack((P1, Pt_set, Pf))
            n = len(Pt_sum)
            dismat = np.zeros([n, n])
            for c in range(n):
                pointc = Pt_sum[c]

                for d in range(n):
                    pointd = Pt_sum[d]
                    D = pointc - pointd
                    distance = sqrt(np.dot(D, D.T))
                    dismat[c, d] = distance
                    dismat[d, c] = distance

            dismat2 = np.zeros([n, n])

            for i in range(1, n - 1):
                qnear = np.argmin(dismat[0:i, i])
                print(qnear)

                while self.checkedge(Pt_sum[i, :], Pt_sum[qnear, :], obstacle_pose, obstacle_radius) == 1:

                    D = Pt_sum[i, :] - Pt_sum[qnear, :]
                    Pt_sum[i, :] = Pt_sum[qnear, :] + 0.5 * D

                    for a in range(n - 1):
                        pointa = Pt_sum[a]

                        for b in range(n - 1):
                            pointb = Pt_sum[b]
                            D = pointa - pointb
                            distance = sqrt(np.dot(D, D.T))
                            dismat[a, b] = distance
                            dismat[b, a] = distance

                dismat2[i, qnear] = 1
                dismat2[qnear, i] = 1

            qnear = np.argmin(dismat[0:n - 1, n - 1])
            dismat2[n - 1, qnear] = 1
            dismat2[qnear, n - 1] = 1

            qMilestones = Pt_sum[-1, :]
            j = len(Pt_sum) - 1

            while j > 0:
                nodepos = argmax(dismat[0:j + 1, j])
                qMilestones = vstack((Pt_sum[nodepos], qMilestones))
                j = nodepos

            iteration = iteration + 50

        return qMilestones

    def fk_request(self,joint_angles, jointnum):
        Tworld = mat([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tbl = mat([[sqrt(2) / 2, sqrt(2) / 2, 0, 278],
               [-sqrt(2) / 2, sqrt(2) / 2, 0, -64],
               [0, 0, 1, 1104],
               [0, 0, 0, 1]])
        T0 = mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 270.35], [0, 0, 0, 1]])
        T1 = mat([[math.cos(joint_angles[0, 0]), -math.sin(joint_angles[0, 0]), 0, 0],
              [math.sin(joint_angles[0, 0]), math.cos(joint_angles[0, 0]), 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
        T2 = mat([[-math.sin(joint_angles[0, 1]), -math.cos(joint_angles[0, 1]), 0, 69],
              [0, 0, 1, 0],
              [-math.cos(joint_angles[0, 1]), math.sin(joint_angles[0, 1]), 0, 0],
              [0, 0, 0, 1]])
        T3 = mat([[math.cos(joint_angles[0, 2]), -math.sin(joint_angles[0, 2]), 0, 0],
              [0, 0, -1, -364.35],
              [math.sin(joint_angles[0, 2]), math.cos(joint_angles[0, 2]), 0, 0],
              [0, 0, 0, 1]])
        T4 = mat([[math.cos(joint_angles[0, 3]), -math.sin(joint_angles[0, 3]), 0, 69],
              [0, 0, 1, 0],
              [-math.sin(joint_angles[0, 3]), -math.cos(joint_angles[0, 3]), 0, 0],
              [0, 0, 0, 1]])
        T5 = mat([[math.cos(joint_angles[0, 4]), -math.sin(joint_angles[0, 4]), 0, 0],
              [0, 0, -1, -374.29],
              [math.sin(joint_angles[0, 4]), math.cos(joint_angles[0, 4]), 0, 0],
              [0, 0, 0, 1]])
        T6 = mat([[math.cos(joint_angles[0, 5]), -math.sin(joint_angles[0, 5]), 0, 0.1],
              [0, 0, 1, 0],
              [-math.sin(joint_angles[0, 5]), -math.cos(joint_angles[0, 5]), 0, 0],
              [0, 0, 0, 1]])
        T7 = mat([[math.cos(joint_angles[0, 6]), -math.sin(joint_angles[0, 6]), 0, 0],
              [0, 0, -1, 0],
              [math.sin(joint_angles[0, 6]), math.cos(joint_angles[0, 6]), 0, 0],
              [0, 0, 0, 1]])
        Tgl = mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 229.525], [0, 0, 0, 1]])

        Tshoulder = Tworld * Tbl* T0* T1* T2
        Telbow = T3* T4
        Twrist = T5* T6* T7
        Tgripper = Tgl
        T = [Tshoulder, Telbow, Twrist, Tgripper]
        Tfk = mat(eye(4))
        i = 0
        while i < jointnum:
            Tfk = np.dot(Tfk, T[i-1])
            i = i + 1

        Tfk=Tfk/1000

        return Tfk

    def robotcollision(self,joint_angles, obstacle_pose, obstacle_radius):
        x1 = self.fk_request(joint_angles, 1)[0:3, 3]
        x2 = self.fk_request(joint_angles, 2)[0:3, 3]
        x3 = self.fk_request(joint_angles, 3)[0:3, 3]
        x4 = self.fk_request(joint_angles, 4)[0:3, 3]

        vec = mat(np.arange(0, 1.1, 0.1))
        m = 11

        x12 = multiply(np.matlib.repmat(x2 - x1, 1, m), np.matlib.repmat(vec, 3, 1)) + np.matlib.repmat(x1, 1, m)
        x23 = multiply(np.matlib.repmat(x3 - x2, 1, m), np.matlib.repmat(vec, 3, 1)) + np.matlib.repmat(x2, 1, m)
        x34 = multiply(np.matlib.repmat(x4 - x3, 1, m), np.matlib.repmat(vec, 3, 1)) + np.matlib.repmat(x3, 1, m)

        x = hstack((x12, x23, x34))

        if sum(sum(multiply((x - np.matlib.repmat(obstacle_pose, 1, 33)),
                    (x - np.matlib.repmat(obstacle_pose, 1, 33))),0) < obstacle_radius **2) > 0:
            collision = 1
        else:
            collision = 0
        return collision

    def checkedge(self,joint_angles1, joint_angles2, obstacle_pose, obstacle_radius):
        n = 11
        m = 7
        vec = mat(np.arange(0, 1.1, 0.1))
        viapts = multiply(np.matlib.repmat((joint_angles2 - joint_angles1).T, 1, n),
                          np.matlib.repmat(vec, m, 1)) + np.matlib.repmat(joint_angles1.T, 1, n)
        collision = 0

        for i in range(n):
            collision = self.robotcollision(viapts[:, i].T, obstacle_pose, obstacle_radius)
            if collision:
                break
        return collision
    

def load_gazebo_models(table_pose=Pose(position=Point(x=1.0, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)),
                       block_reference_frame="world",
                       block1_pose=Pose(position=Point(x=0.7225, y=-0.0235, z=0.7825)),
                       block1_reference_frame="world",
                       block2_pose=Pose(position=Point(x=0.6225, y=0.2765, z=0.7825)),
                       block2_reference_frame="world",
                       trash_can_pose=Pose(position=Point(x=0.5, y=0.644, z=0)),
                       trash_can_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')

    block1_xml = ''
    with open (model_path + "block1/model.urdf", "r") as block1_file:
        block1_xml=block1_file.read().replace('\n', '')

    block2_xml = ''
    with open (model_path + "block2/model.urdf", "r") as block2_file:
        block2_xml=block2_file.read().replace('\n', '')

    trash_can_xml=''
    with open (model_path + "trash_can/model.sdf", "r") as trash_can_file:
        trash_can_xml=trash_can_file.read().replace('\n', '')



    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))
    
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block1", block1_xml, "/",
                               block1_pose, block1_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block2", block2_xml, "/",
                               block2_pose, block2_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))
    
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_urdf = spawn_sdf("trash_can", trash_can_xml, "/",
                               trash_can_pose, trash_can_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("block")
        resp_delete = delete_model("trash_can")
        resp_delete = delete_model("block1")
        resp_delete = delete_model("block2")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

def main():
    """RSDK Inverse Kinematics Pick and Place Example

    A Pick and Place example using the Rethink Inverse Kinematics
    Service which returns the joint angles a requested Cartesian Pose.
    This ROS Service client is used to request both pick and place
    poses in the /base frame of the robot.

    Note: This is a highly scripted and tuned demo. The object location
    is "known" and movement is done completely open loop. It is expected
    behavior that Baxter will eventually mis-pick or drop the block. You
    can improve on this demo by adding perception and feedback to close
    the loop.
    """
    rospy.init_node("ik_pick_and_place_demo")
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    load_gazebo_models()
    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    # Wait for the All Clear from emulator startup
    rospy.wait_for_message("/robot/sim/started", Empty)

    limb = 'left'
    hover_distance = 0.15 # meters
    # Starting Joint angles for left arm
    starting_joint_angles = {'left_w0': 0.6699952259595108,
                             'left_w1': 1.030009435085784,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.189968899785275,
                             'left_e1': 1.9400238130755056,
                             'left_s0': -0.08000397926829805,
                             'left_s1': -0.9999781166910306}
    overhead_orientation = Quaternion(
                             x=-0.0249590815779,
                             y=0.999649402929,
                             z=0.00737916180073,
                             w=0.00486450832011)
    pnp = PickAndPlace(limb, hover_distance)
    # An orientation for gripper fingers to be overhead and parallel to the obj
    
    ball_poses = list()
    
    container_pose=Pose(position=Point(x=0.5275, y=0.6675, z=-0.14),orientation=overhead_orientation)
    # The Pose of the block in its initial location.
    # You may wish to replace these poses with estimates
    # from a perception node.
    ball_poses.append(Pose(
        position=Point(x=0.7, y=0.15, z=-0.14),
        orientation=overhead_orientation))
    ball_poses.append(Pose(
        position=Point(x=0.75, y=0, z=-0.14),
        orientation=overhead_orientation))
    ball_poses.append(Pose(
        position=Point(x=0.65, y=0.3, z=-0.14),
        orientation=overhead_orientation))
    # Feel free to add additional desired poses for the object.
    # Each additional pose will get its own pick and place.

    # Move to the desired starting angles
    pnp._guarded_move_to_joint_position(starting_joint_angles)
    
    for i in range(len(ball_poses)):
        print("\nPicking...")
        pnp.pick(ball_poses[i])
        print("\nPlacing...")
        pnp.place(container_pose)

if __name__ == '__main__':
    sys.exit(main())

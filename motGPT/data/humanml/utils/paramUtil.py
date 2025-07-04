import numpy as np

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

# offsets is the global offset between children and parent joint.
# 22
t2m_raw_offsets = np.array([[0, 0, 0],  # pelvis
                                 [1, 0, 0],  # left_hip
                                 [-1, 0, 0],  # right_hip
                                 [0, 1, 0],  # spine1
                                 [0, -1, 0],  # left_knee
                                 [0, -1, 0],  # right_knee
                                 [0, 1, 0],  # spine2
                                 [0, -1, 0],  # left_ankle
                                 [0, -1, 0],  # right_ankle
                                 [0, 1, 0],  # spine3
                                 [0, 0, 1],  # left_foot
                                 [0, 0, 1],  # right_foot
                                 [0, 1, 0],  # neck
                                 [1, 0, 0],  # left_collar
                                 [-1, 0, 0],  # right_collar
                                 [0, 0, 1],  # head
                                 [0, -1, 0],  # left_shoulder
                                 [0, -1, 0],  # right_shoulder
                                 [0, -1, 0],  # left_elbow
                                 [0, -1, 0],  # right_elbow
                                 [0, -1, 0],  # left_wrist
                                 [0, -1, 0]])  # right_wrist

# 30
t2m_hand_raw_offsets = np.array([[1, 0, 0],  # left_index1
                                [1, 0, 0],  # left_index2
                                [1, 0, 0],  # left_index3
                                [1, 0, 0],  # left_middle1
                                [1, 0, 0],  # left_middle2
                                [1, 0, 0],  # left_middle3
                                [1, 0, 0],  # left_pinky1
                                [1, 0, 0],  # left_pinky2
                                [1, 0, 0],  # left_pinky3
                                [1, 0, 0],  # left_ring1
                                [1, 0, 0],  # left_ring2
                                [1, 0, 0],  # left_ring3
                                [1, 0, 0],  # left_thumb1
                                [1, 0, 0],  # left_thumb2
                                [1, 0, 0],  # left_thumb3
                                [-1, 0, 0],  # right_index1
                                [-1, 0, 0],  # right_index2
                                [-1, 0, 0],  # right_index3
                                [-1, 0, 0],  # right_middle1
                                [-1, 0, 0],  # right_middle2
                                [-1, 0, 0],  # right_middle3
                                [-1, 0, 0],  # right_pinky1
                                [-1, 0, 0],  # right_pinky2
                                [-1, 0, 0],  # right_pinky3
                                [-1, 0, 0],  # right_ring1
                                [-1, 0, 0],  # right_ring2
                                [-1, 0, 0],  # right_ring3
                                [-1, 0, 0],  # right_thumb1
                                [-1, 0, 0],  # right_thumb2
                                [-1, 0, 0],])  # right_thumb3

t2m_all_raw_offsets = np.concatenate(
    (t2m_raw_offsets, t2m_hand_raw_offsets), axis=0)


t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]
# right_wrist ---> right_pinky1 ---> right_pinky2 --->right_pinky3
# right_wrist ---> right_ring1 ---> right_ring2 --->right_ring3
# right_wrist ---> right_middle1 ---> right_middle2 --->right_middle3
# right_wrist ---> right_index1 ---> right_index2 --->right_index3
# right_wrist ---> right_thumb1 ---> right_thumb2 --->right_thumb3

t2m_body_hand_kinematic_chain = t2m_kinematic_chain + \
    t2m_left_hand_chain + t2m_right_hand_chain
    
kit_tgt_skel_id = '03950'

t2m_tgt_skel_id = '000021'


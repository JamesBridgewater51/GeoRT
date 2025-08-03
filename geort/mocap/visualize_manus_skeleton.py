import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Skeleton Connection Definition ---
# Defines the "bones" by connecting the joint IDs.
MANUS_SKELETON_BONES = [
    # Palm
    [0, 1], [0, 5], [0, 9], [0, 13], [0, 17], # Wrist to finger bases
    [5, 9], [9, 13], [13, 17],             # Connections between finger bases

    # Thumb
    [1, 2], [2, 3], [3, 4],

    # Index Finger
    [5, 6], [6, 7], [7, 8],

    # Middle Finger
    [9, 10], [10, 11], [11, 12],

    # Ring Finger
    [13, 14], [14, 15], [15, 16],

    # Pinky Finger
    [17, 18], [18, 19], [19, 20]
]

def calculate_absolute_positions(bone_offsets):
    """
    Calculates the absolute 3D positions of joints from their parent-relative offsets.
    This performs a basic forward kinematics pass.
    """
    # The full skeleton has 21 joints (1 wrist + 5 fingers * 4 joints)
    absolute_positions = np.zeros((21, 3))
    
    # Define which joints belong to which finger chain, starting from the wrist (ID 0)
    # Each list contains the sequential IDs for that finger
    chains = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20]
    }
    
    # The C++ code structure implies a hierarchy where each finger chain is relative to the wrist (ID 0)
    # And each joint in a finger is relative to the previous joint in that same finger.
    for finger, joint_ids in chains.items():
        # The first joint of each finger is relative to the wrist (ID 0)
        parent_position = absolute_positions[0] 
        # The first joint in each finger chain from the C++ code is a direct offset from the root
        absolute_positions[joint_ids[0]] = parent_position + bone_offsets[joint_ids[0]]
        parent_position = absolute_positions[joint_ids[0]]
        
        # Subsequent joints are relative to the previous one in the chain
        for i in range(1, len(joint_ids)):
            joint_id = joint_ids[i]
            # The bone_offsets are defined relative to the previous joint in the chain
            # E.g., The offset for joint 6 is relative to joint 5's position
            current_pos = parent_position + bone_offsets[joint_id]
            absolute_positions[joint_id] = current_pos
            parent_position = current_pos
            
    return absolute_positions

def plot_hand(ax, joint_positions, title):
    """
    Plots a single hand skeleton on a given 3D axis.
    """
    # Plot the joints as scatter points
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], c='red', marker='o', s=50, depthshade=True)

    # Plot the bones as lines connecting the joints
    for bone in MANUS_SKELETON_BONES:
        start_joint = joint_positions[bone[0]]
        end_joint = joint_positions[bone[1]]
        ax.plot([start_joint[0], end_joint[0]], 
                [start_joint[1], end_joint[1]], 
                [start_joint[2], end_joint[2]], 'b-')

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(title)
    ax.view_init(elev=90, azim=-90) # Top-down view for clarity
    ax.set_aspect('equal') # Ensure proportions are correct

def main():
    """
    Main function to define data and create the plots.
    """
    # Raw bone offset vectors extracted from the C++ code
    # We add the wrist at [0,0,0] as the first entry (ID 0)
    left_hand_offsets = np.array([
        [0.0, 0.0, 0.0], # ID 0: Wrist
        [0.024950, 0.000000, 0.025320], [0.000000, 0.000000, 0.032742], [0.000000, 0.000000, 0.028739], [0.000000, 0.000000, 0.028739],
        [0.011181, 0.000000, 0.052904], [0.000000, 0.000000, 0.038257], [0.000000, 0.000000, 0.020884], [0.000000, 0.000000, 0.018759],
        [0.000000, 0.000000, 0.051287], [0.000000, 0.000000, 0.041861], [0.000000, 0.000000, 0.024766], [0.000000, 0.000000, 0.019683],
        [-0.011274, 0.000000, 0.049802], [0.000000, 0.000000, 0.039736], [0.000000, 0.000000, 0.023564], [0.000000, 0.000000, 0.019868],
        [-0.020145, 0.000000, 0.047309], [0.000000, 0.000000, 0.033175], [0.000000, 0.000000, 0.018020], [0.000000, 0.000000, 0.019129],
    ])

    right_hand_offsets = np.array([
        [0.0, 0.0, 0.0], # ID 0: Wrist
        [0.0250, 0.0000, 0.0050], [0.0000, 0.0000, 0.0390], [0.0000, 0.0000, 0.0330], [0.0000, 0.0000, 0.0210],
        [0.0170, 0.0000, 0.0870], [0.0000, 0.0000, 0.0260], [0.0000, 0.0000, 0.0220], [0.0000, 0.0000, 0.0200],
        [0.0000, 0.0000, 0.0920], [0.0000, 0.0000, 0.0260], [0.0000, 0.0000, 0.0260], [0.0000, 0.0000, 0.0220],
        [-0.0170, 0.0000, 0.0840], [0.0000, 0.0000, 0.0210], [0.0000, 0.0000, 0.0210], [0.0000, 0.0000, 0.0200],
        [-0.0340, 0.0000, 0.0720], [0.0000, 0.0000, 0.0210], [0.0000, 0.0000, 0.0210], [0.0000, 0.0000, 0.0200],
    ])

    # Calculate absolute positions for both hands
    left_hand_joints = calculate_absolute_positions(left_hand_offsets)
    right_hand_joints = calculate_absolute_positions(right_hand_offsets)
    
    # Set up the figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Plot the left hand
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_hand(ax1, left_hand_joints, 'Left Hand Skeleton (Initial Pose)')

    # Plot the right hand
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_hand(ax2, right_hand_joints, 'Right Hand Skeleton (Initial Pose)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os

# --- Configuration ---
# TODO: Change this to the path of your skeleton data file.
SKELETON_DATA_FILE = 'data/human_alex.npy'
SKELETON_DATA_FILE = 'data/Jul_31_2025/manus_mocap_test_long_fps_10.npy'
FPS = 1000.0

# --- Skeleton Definition ---
# This defines the "bones" of the hand by connecting the joint IDs.
# Based on the Manus joint ID map you provided.
MANUS_SKELETON_BONES = [
    # Palm
    [0, 1], [0, 5], [0, 9], [0, 13], [0, 17], # Wrist to finger bases
    [5, 9], [9, 13], [13, 17], # Connections between finger bases

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

def main():
    """
    Main function to load data and run the visualization.
    """
    # Load the skeleton animation data from the .npy file
    try:
        skeleton_data = np.load(SKELETON_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{SKELETON_DATA_FILE}'.")
        return
        
    if skeleton_data.ndim != 3 or skeleton_data.shape[2] != 3:
        print(f"Error: Data has incorrect shape {skeleton_data.shape}. Expected (T, J, 3).")
        return
        
    num_frames, num_joints, _ = skeleton_data.shape
    print(f"Loaded {num_frames} frames with {num_joints} joints each.")

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Determine axis limits to keep the view stable
    all_x = skeleton_data[:, :, 0].flatten()
    all_y = skeleton_data[:, :, 1].flatten()
    all_z = skeleton_data[:, :, 2].flatten()

    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    min_z, max_z = np.min(all_z), np.max(all_z)

    # Make the plot cubic
    max_range = np.array([max_x-min_x, max_y-min_y, max_z-min_z]).max() / 2.0
    mid_x = (max_x+min_x) * 0.5
    mid_y = (max_y+min_y) * 0.5
    mid_z = (max_z+min_z) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Manus Hand Skeleton Animation')

    # This function is called for each frame of the animation
    def update_plot(frame_index):
        ax.cla() # Clear the previous frame
        
        # Get the joint positions for the current frame
        joints = skeleton_data[frame_index]
        
        # Plot the joints as scatter points
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', marker='o')
        
        # Plot the bones as lines connecting the joints
        for bone in MANUS_SKELETON_BONES:
            start_joint = joints[bone[0]]
            end_joint = joints[bone[1]]
            ax.plot([start_joint[0], end_joint[0]], 
                    [start_joint[1], end_joint[1]], 
                    [start_joint[2], end_joint[2]], 'b-')

        # Re-apply the stable axis limits and labels
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Manus Hand Skeleton Animation (Frame {frame_index}/{num_frames})')

    # Create the animation object
    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        frames=num_frames, 
        interval=1000/FPS, # Interval in milliseconds
        repeat=True
    )

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()

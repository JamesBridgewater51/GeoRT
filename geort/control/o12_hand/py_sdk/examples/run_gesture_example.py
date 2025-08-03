import numpy as np
from ..src import O12HandCtrl, GestureID

def print_vector(label: str, vec: np.ndarray):
    """Helper function to print a labeled vector."""
    vec_str = " ".join([f"{x:8.10f}" for x in vec])
    print(f"{label:<28}: [ {vec_str} ]")

def main():
    """
    Example usage of the O12HandCtrl SDK.
    This script demonstrates setting a gesture and converting between
    motor inputs and joint positions.
    """
    print("--- O12Hand Python SDK Example ---")
    
    # Initialize the controller for a right hand
    # For a left hand, use: O12HandCtrl(is_right_hand=False)
    hand_ctrl = O12HandCtrl(is_right_hand=True)
    
    print("\nAvailable Gestures:")
    for gesture in GestureID:
        if gesture.name != "MAXGESTUREID":
            print(f"  {gesture.value:2d}: {gesture.name}")

    try:
        gesture_id_str = input("\nEnter the integer ID for the desired gesture: ")
        gesture_id = int(gesture_id_str)
        gesture = GestureID(gesture_id)
    except (ValueError, KeyError):
        print(f"Invalid ID. Defaulting to Gesture 0 (HOME).")
        gesture = GestureID.HOME

    print(f"\nSelected Gesture: {gesture.name} ({gesture.value})")
    print("-" * 35)

    # 1. Set a hand gesture to get the required motor inputs
    print("1. Calculating motor inputs for the gesture...")
    motor_input = hand_ctrl.set_hand_gesture(gesture)
    print_vector("Calculated Motor Inputs", motor_input)

    # 2. Convert the motor inputs back to active joint positions
    print("\n2. Converting motor inputs back to joint positions...")
    active_joints = hand_ctrl.convert_actuator_to_joint(motor_input)
    print_vector("Resulting Active Joints (rad)", active_joints)

    # 3. Compute all 19 joint positions (active + passive)
    print("\n3. Calculating all 19 joint positions...")
    all_joints = hand_ctrl.get_all_joint_pos(active_joints)
    print_vector("All Joint Positions (rad)", all_joints)
    
    print("\n--- Example Finished ---")

if __name__ == "__main__":
    main()
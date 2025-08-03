# Omnihand Python SDK (omnihand-py)

This project is a Python port of the original C++ SDK for controlling the O12 dexterous hand. It provides a faithful implementation of the conversion logic between motor inputs and joint positions, wrapped in a user-friendly and Pythonic package.

## Features

The SDK provides a set of functionalities for controlling the O12 hand:

1.  **Gesture-to-Motor Input Conversion:** Generate motor input values based on predefined gestures.
2.  **Motor-to-Joint Conversion:** Convert motor feedback values (or target inputs) into active joint angles (radians).
3.  **Joint-to-Motor Conversion:** Calculate motor input values from target active joint angles (radians).
4.  **Full Joint Angle Computation:** Compute all 19 joint angles (both active and passive) from the 12 active joint angles.

## Project Structure
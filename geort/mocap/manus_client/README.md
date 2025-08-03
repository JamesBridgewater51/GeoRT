Purpose: Interface directly with the Manus SDK and broadcast the raw data over ROS2. This is a standard practice to bridge proprietary C++ SDKs into the ROS ecosystem.

Code: right_hand_ros.cpp (and other .cpp files) uses the provided ManusSDK.h to connect to the Manus Core software.

Functionality: It receives raw joint data (positions and quaternions) for the 21 hand joints. It then publishes this raw data to several ROS2 topics (e.g., /manus_quats).

Build: It's built as a standard ROS2 package using colcon, as shown in CMakeLists.txt and colcon_build.sh.

Note:

The Joint-to-ID correspondences map：

### Step 1: Trace the Skeleton Creation in right_hand_ros.cpp

The process of defining the skeleton starts in your main function and flows through several methods before the nodes and their IDs are actually created.

    main() -> Run(): The main function calls t_Client.Run().

    Run() -> LoadTestSkeleton(): Inside the Run() method, after a connection is established, it immediately calls LoadTestSkeleton() to configure the hand model that Manus Core will use for tracking.

    LoadTestSkeleton() -> SetupHandNodes(): This is the key step. LoadTestSkeleton() creates a basic skeleton configuration and then calls SetupHandNodes(t_SklIndex) to populate it with the actual joints (nodes).

### Step 2: Analyze the Logic of SetupHandNodes()

This function is the source of truth for your joint IDs. It programmatically creates each joint in a specific order and assigns it a unique integer ID.

The core logic is a nested loop:

    An outer loop iterates 5 times, once for each finger (for (uint32_t i = 0; i < t_NumFingers; i++)).

    An inner loop iterates 4 times for each finger, creating the 4 joints that make up that finger chain (for (uint32_t j = 0; j < t_NumJoints; j++)).

The ID for each joint is assigned using this formula inside the loop: 1 + t_FingerId + j.

    t_FingerId is a running counter that increases by 4 after each finger is complete.

    j is the inner loop counter from 0 to 3.

A special case is the Hand Root (Wrist), which is created before the loops with a hardcoded ID of 0.

### Step 3: The Joint-to-ID Correspondence Map

By following the logic of the loops, we can build a definitive map of which joint corresponds to which ID. The anatomical names (MCP, PIP, etc.) are based on the standard Manus skeleton hierarchy.

    ID 0: Hand Root (Wrist)

    Thumb (Finger i = 0)

        t_FingerId starts at 0.

        ID 1: Thumb Metacarpal (CMC)

        ID 2: Thumb Proximal Phalanx (MCP)

        ID 3: Thumb Distal Phalanx (IP)

        ID 4: Thumb Tip

    Index Finger (Finger i = 1)

        t_FingerId is now 4.

        ID 5: Index Metacarpal (MCP)

        ID 6: Index Proximal Phalanx (PIP)

        ID 7: Index Middle Phalanx (DIP)

        ID 8: Index Tip

    Middle Finger (Finger i = 2)

        t_FingerId is now 8.

        ID 9: Middle Metacarpal (MCP)

        ID 10: Middle Proximal Phalanx (PIP)

        ID 11: Middle Middle Phalanx (DIP)

        ID 12: Middle Tip

    Ring Finger (Finger i = 3)

        t_FingerId is now 12.

        ID 13: Ring Metacarpal (MCP)

        ID 14: Ring Proximal Phalanx (PIP)

        ID 15: Ring Middle Phalanx (DIP)

        ID 16: Ring Tip

    Pinky Finger (Finger i = 4)

        t_FingerId is now 16.

        ID 17: Pinky Metacarpal (MCP)

        ID 18: Pinky Proximal Phalanx (PIP)

        ID 19: Pinky Middle Phalanx (DIP)

        ID 20: Pinky Tip
import numpy as np
import warnings
from .constants import *
from cprint import cprint

DBG_O12_CONTROL_LIMITS = False
if DBG_O12_CONTROL_LIMITS:
    ACTIVE_ACTUATOR_MAX, ACTIVE_ACTUATOR_MIN = np.full((MAX_ACTIVE_JOINT,), -np.inf, dtype=np.int32), np.full((MAX_ACTIVE_JOINT,), np.inf, dtype=np.int32)
    ACTIVE_MOTOR_LENGTH_MAX, ACTIVE_MOTOR_LENGTH_MIN = np.full((MAX_ACTIVE_JOINT,), -np.inf, dtype=np.float64), np.full((MAX_ACTIVE_JOINT,), np.inf, dtype=np.float64)


class O12HandCtrl:
    """
    Provides control functionalities for the Omnihand O12 robotic hand,
    including gesture setting and joint position conversions.
    This class is a Python port of the original C++ SDK.
    """

    def __init__(self, is_right_hand: bool = True):
        """
        Initializes the O12HandCtrl.

        Args:
            is_right_hand (bool): Set to True for a right-hand and False for a left-hand.
        """
        self._is_right_hand = is_right_hand

        # --- Actuator and Motor Limits (from C++ source) ---
        # Note: In the C++ source, left and right limits are identical.
        # The logic is preserved here for faithfulness to the original.
        _left_actuator_max = np.array([1825, 1825, 1825, 1825, 0, 2000, 2000, 2000, 2000, 2000, 2000, 2000], dtype=np.int32)
        _left_actuator_min = np.array([0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        _right_actuator_max = np.array([1825, 1825, 1825, 1825, 0, 2000, 2000, 2000, 2000, 2000, 2000, 2000], dtype=np.int32)
        _right_actuator_min = np.array([0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

        if self._is_right_hand:
            self._motor_input_max = _right_actuator_max
            self._motor_input_min = _right_actuator_min
            self._active_joint_max = O12RightHandUpperLimits
            self._active_joint_min = O12RightHandLowerLimits
            self._motor_length_max = np.array([9.18700000e-05, 1.46509266e-03, 1.46428506e-03, 9.20900000e-05,
 9.25745281e-03, 0.00000000e+00, 2.94377735e-04, 3.84199423e-06,
 0.00000000e+00, 0.00000000e+00, 1.00182249e-02, 9.75592691e-03], dtype=np.float64)
            self._motor_length_min = np.array([-2.50680494e-03, -2.50658494e-03, -1.16448106e-02, -1.53941008e-02,
  0.00000000e+00, -7.80532153e-03, -1.61518820e-02, -1.67476429e-03,
 -1.52515617e-02, -1.59909343e-02, 3.06606299e-10, 0.00000000e+00], dtype=np.float64)
        else: # Left Hand
            self._motor_input_max = _left_actuator_max
            self._motor_input_min = _left_actuator_min
            self._active_joint_max = O12LeftHandUpperLimits
            self._active_joint_min = O12LeftHandLowerLimits
            self._motor_length_max = np.array([1.46428506e-03, 9.20900000e-05, 9.18700000e-05, 1.46509266e-03,
 9.25745281e-03, 0.00000000e+00, 2.94377735e-04, 1.88846122e-04,
 0.00000000e+00, 0.00000000e+00, 1.00182249e-02, 9.75592691e-03], dtype=np.float64)
            self._motor_length_min = np.array([-2.50680494e-03, -4.34761299e-03, -1.53949094e-02, -1.37679835e-02,
  0.00000000e+00, -7.80532153e-03, -1.74770236e-02, -2.25166323e-03,
 -1.52515617e-02, -1.59909343e-02, 3.06606299e-10, 0.00000000e+00], dtype=np.float64)

        # Note: np.polyval expects coeffs from highest power to lowest, so we reverse the C++ arrays.
        self._coeffs_thumb_roll = np.array([0.1882, 0.4586, -2.492, -4.414, 0], dtype=np.float64)
        self._coeffs_thumb_abad = np.array([-0.003763, -1.543, 5.34, 2.258, 0.0], dtype=np.float64)
        self._coeffs_thumb_mcp = np.array([-0.3282, 4.583, -5.908, 13.67, 0.0], dtype=np.float64)
        self._coeffs_thumb_pip = np.array([0.03171, -1.426, 2.782, 6.43, 0.0], dtype=np.float64)
        self._coeffs_ring_pinky = np.array([0.3069, 0.5897, -4.315, -6.131, 0.0], dtype=np.float64)

        self._coeffs_thumb_roll_motor2joint = np.array([-0.0002056, -0.003241, -0.02276, -0.2234, 0.0003082], dtype=np.float64)
        self._coeffs_thumb_abad_motor2joint = np.array([-0.0002297, 0.005332, -0.04555, 0.2933, 0.0151], dtype=np.float64)
        self._coeffs_thumb_mcp_motor2joint = np.array([-4.305e-06, -0.0001356, 0.003089, 0.07182, -0.0009974], dtype=np.float64)
        self._coeffs_thumb_pip_motor2joint = np.array([-7.521e-06, 0.0005015, -0.006454, 0.1512, 0.001086], dtype=np.float64)
        self._coeffs_ring_pinky_motor2joint = np.array([-1.273e-05, -0.0005519, -0.009015, -0.1511, 0.003838], dtype=np.float64)

        self._coeffs_index1 = np.array([9.187e-05, 0.006308, -0.009054, -0.003653, 0.001198, -0.004188, -0.001179, 0.005724, -0.0003357, 0.002189], dtype=np.float64)
        self._coeffs_index2 = np.array([9.209e-05, -0.00631, -0.009054, -0.003652, -0.001195, -0.004188, 0.001179, 0.005724, 0.0003358, 0.002189], dtype=np.float64)
        self._coeffs_index1_motor2joint = np.array([1.808e-05, -79.06, 79.06, -653.8, -1.453, 655.2, -5.315e+04, 1.008e+05, -1.008e+05, 5.318e+04], dtype=np.float64)
        self._coeffs_index2_motor2joint = np.array([0.005958, -52.95, -52.97, -2402.0, 2243.0, -2404.0, -1.13e+05, 5.735e+04, 5.744e+04, -1.13e+05], dtype=np.float64)

        self._index_pip_coeffs = np.array([0.000121510769570933,  -2.50581605596806e-17, -0.00103193271538432,
      -0.00427430539256996,  0.00258150633056518,   0.00106565649135874,
      0.000445355187310781,  -1.15126760683569e-16, -0.000315293352620664,
      -0.00176175763410993,  -0.000359504650317669, 0.000164618553099208,
      -0.000971467023043282, 4.26342912043187e-17,  -3.74107772687308e-17,
      -0.00160481465128942,  -0.000497492920305404, 0.00112894377816561,
      -3.52449771274991e-17, 0.000156796765047573,  -1.44301859784978e-17,
      -0.000878819839291868, 1.81155546699953e-17], dtype=np.float64)
        self._middle_pip_coeffs = np.array([      -0.000109932620914924, -3.18785239704263e-17, 0.00113033958734184,
      -0.00559155815327005,  -0.00259866102086915,  -0.00118840846223080,
      0.00119767174835502,   -1.31570246962506e-16, 0.000305003046707372,
      -0.00255295056336236,  0.000313535197591134,  -0.000154760553058508,
      0.00109876667802121,   4.71747610106523e-17,  5.46867960305804e-17,
      -0.00165731420103010,  0.000483193043350727,  0.00125723116061942,
      -3.65349518429928e-17, 0.000333771002971031,  5.43196285667778e-17,
      0.000770994250512108,  -1.96478838587447e-17], dtype=np.float64)
        self._index_pip_coeffs_motor2joint = np.array([    0.0173596453435215, 1.97171321894013e-13, -0.120343097864840,
    -245.325895605700, 0.462734728662583, 0.151700922498064,
    18838.0466623167, 1.78625701663702e-14, -0.0529754300109596,
    -1276113.14949799, -0.0391058192681997, 0.0274437579747416,
    0.0,      -1.63250107955574e-13,   -9.03734415058791e-13,
    26.5563336955081, -0.0772201461144102,-37.0697860474473,
    3.49020768234486e-15, 1.99948844467367,5.34506047834448e-12,
    -2939.71010872299, -4.68988809496012e-14], dtype=np.float64)
        self._middle_pip_coeffs_motor2joint = np.array([      -0.0289128957320330, -2.27065602706093e-13, 0.147346884558557,
      -214.057422041002, -0.431818226482737, -0.171044209910922,
      -11920.7177704693, -2.00247621030886e-14, 0.0588543967614107,
      -814147.575218152, 0.0219579712374130, -0.0286496084627873,
      0.0,             1.65078067894637e-13, -6.67524556160442e-12,
      33.1263043072275, 0.0686439152657324, -35.1766050055121,
     -7.77876247691696e-15, -2.28131328730178, 1.12876765393007e-10,
      2845.43805588110, 1.60350358888510e-13], dtype=np.float64)

        # Passive joint coefficients (DIP and PIP)
        self._thumb_dip_coeff = np.array([-0.11, -0.2753, -0.3319, 0.6411, -0.0003096], dtype=np.float64)
        self._index_dip_coeff = np.array([-0.2818, 0.3563, -0.05396, 1.103, -0.00213], dtype=np.float64)
        self._middle_dip_coeff = np.array([-0.2258, 0.3413, -0.0627, 1.1, -0.001314], dtype=np.float64)
        self._ring_pip_coeff = np.array([0.168, -0.4807, 0.4097, 0.7812, 9.541e-05], dtype=np.float64)
        self._ring_dip_coeff = np.array([0.021, -0.3236, 0.4381, 0.8647, -0.002773], dtype=np.float64)

        # Gesture definitions
        self._gesture_map = {gestureID: active_qpos.tolist() for gestureID, active_qpos in enumerate(ActiveJointPosGestureMap)}

        # Warnings and logging related.
        self.warn_msg_actuator_joint_mapping_issued = False
        

    def set_hand_gesture(self, gesture: GestureID) -> np.ndarray:
        """
        Sets a predefined hand gesture and returns the motor input values.

        Args:
            gesture (GestureID): The desired gesture from the GestureID enum.

        Returns:
            np.ndarray: A 12-element array of motor input values.
        """
        if gesture not in self._gesture_map:
            cprint.warn(f"Gesture {gesture.name} not implemented, returning HOME position.")
            active_joint_pos = np.array(self._gesture_map[GestureID.HOME], dtype=np.float64)
        else:
            active_joint_pos = np.array(self._gesture_map[gesture], dtype=np.float64)
        
        # Clamp to valid joint limits before conversion
        self._clamp(self._active_joint_max, self._active_joint_min, active_joint_pos)
        return self.convert_joint_to_actuator(active_joint_pos, is_active_only=True)

    def convert_joint_to_actuator(self, joint_pos: np.ndarray, is_active_only: bool = True) -> np.ndarray:
        """
        Converts joint positions (in radians) into actuator input values.

        Args:
            joint_pos (np.ndarray): Joint positions array. Can be either:
                - 12-element array of active joint positions (if is_active_only=True)
                - 19-element array of all joint positions (if is_active_only=False)
            is_active_only (bool): True if joint_pos contains only active joints, 
                                 False if it contains all 19 joints.

        Returns:
            np.ndarray: 12-element array of actuator/motor input values.
        """
        if is_active_only:
            if joint_pos.shape != (MAX_ACTIVE_JOINT,):
                raise ValueError(f"Input joint_pos must have {MAX_ACTIVE_JOINT} elements when is_active_only=True.")
            active_joint_pos = joint_pos.copy()
        else:
            if joint_pos.shape != (MAX_JOINT,):
                raise ValueError(f"Input joint_pos must have {MAX_JOINT} elements when is_active_only=False.")
            
            # Extract active joints from all joints using the mapping from get_all_joint_pos method
            # Based on the mapping in get_all_joint_pos, create the reverse mapping
            active_joint_pos = np.zeros(MAX_ACTIVE_JOINT, dtype=np.float64)
            
            # Map from all joints to active joints (reverse of get_all_joint_pos mapping)
            active_joint_pos[ActiveJointID.ActiveJointThumbRoll] = joint_pos[JointID.JThumbRoll]
            active_joint_pos[ActiveJointID.ActiveJointThumbAbAd] = joint_pos[JointID.JThumbAbad]
            active_joint_pos[ActiveJointID.ActiveJointThumbMCP] = joint_pos[JointID.JThumbMCP]
            active_joint_pos[ActiveJointID.ActiveJointThumbPIP] = joint_pos[JointID.JThumbPIP]
            active_joint_pos[ActiveJointID.ActiveJointIndexAbAd] = joint_pos[JointID.JIndexAbad]
            active_joint_pos[ActiveJointID.ActiveJointIndexMCP] = joint_pos[JointID.JIndexMCP]
            active_joint_pos[ActiveJointID.ActiveJointIndexPIP] = joint_pos[JointID.JIndexPIP]
            active_joint_pos[ActiveJointID.ActiveJointMiddleABAD] = joint_pos[JointID.JMiddleAbad]
            active_joint_pos[ActiveJointID.ActiveJointMiddleMCP] = joint_pos[JointID.JMiddleMCP]
            active_joint_pos[ActiveJointID.ActiveJointMiddlePIP] = joint_pos[JointID.JMiddlePIP]
            active_joint_pos[ActiveJointID.ActiveJointRingMCP] = joint_pos[JointID.JRingMCP]
            active_joint_pos[ActiveJointID.ActiveJointPinkyMCP] = joint_pos[JointID.JPinkyMCP]
        
        pos = active_joint_pos.copy()
        self._check_and_clamp_finger_workspace(pos)
        
        motor_length = self._active_joint_to_motor_length(pos)
        motor_input = self._motor_length_to_motor_input(motor_length)
        return motor_input

    def convert_actuator_to_joint(self, motor_input: np.ndarray) -> np.ndarray:
        """
        Converts actuator input values back into active joint positions (in radians).

        Args:
            motor_input (np.ndarray): 12-element array of actuator/motor input values.

        Returns:
            np.ndarray: 12-element array of active joint positions.
        """
        if not self.warn_msg_actuator_joint_mapping_issued:
            cprint.err("Actuator to joint mapping coefficients and Joint to Actuator mapping coefficients are polynomially fitted based on large samples. And experiments have shown the coefficients are not consistent (e.g, the Actuator->Joint followed by Joint->Actuator cannot recover original input, and vice versa.). Please be wary of this inconsistency.")
            self.warn_msg_actuator_joint_mapping_issued = True
        if motor_input.shape != (ACTUATOR_COUNT,):
            raise ValueError(f"Input motor_input must have {ACTUATOR_COUNT} elements, got {motor_input.shape[0]}.")

        motor_length = self._motor_input_to_motor_length(motor_input)
        active_joint_pos = self._motor_length_to_active_joint(motor_length)
        return active_joint_pos

    def get_all_joint_pos(self, active_joint_pos: np.ndarray) -> np.ndarray:
        """
        Computes all 19 joint angles (active and passive) from the 12 active joint angles.

        Args:
            active_joint_pos (np.ndarray): 12-element array of active joint positions.

        Returns:
            np.ndarray: 19-element array of all joint positions.
        """
        if active_joint_pos.shape != (MAX_ACTIVE_JOINT,):
            raise ValueError(f"Input active_joint_pos must have {MAX_ACTIVE_JOINT} elements.")

        all_joints = np.zeros(MAX_JOINT, dtype=np.float64)

        # Map active joints to their positions in the full joint array
        all_joints[JointID.JThumbRoll] = active_joint_pos[ActiveJointID.ActiveJointThumbRoll]
        all_joints[JointID.JThumbAbad] = active_joint_pos[ActiveJointID.ActiveJointThumbAbAd]
        all_joints[JointID.JThumbMCP] = active_joint_pos[ActiveJointID.ActiveJointThumbMCP]
        all_joints[JointID.JThumbPIP] = active_joint_pos[ActiveJointID.ActiveJointThumbPIP]
        all_joints[JointID.JIndexAbad] = active_joint_pos[ActiveJointID.ActiveJointIndexAbAd]
        all_joints[JointID.JIndexMCP] = active_joint_pos[ActiveJointID.ActiveJointIndexMCP]
        all_joints[JointID.JIndexPIP] = active_joint_pos[ActiveJointID.ActiveJointIndexPIP]
        all_joints[JointID.JMiddleAbad] = active_joint_pos[ActiveJointID.ActiveJointMiddleABAD]
        all_joints[JointID.JMiddleMCP] = active_joint_pos[ActiveJointID.ActiveJointMiddleMCP]
        all_joints[JointID.JMiddlePIP] = active_joint_pos[ActiveJointID.ActiveJointMiddlePIP]
        all_joints[JointID.JRingMCP] = active_joint_pos[ActiveJointID.ActiveJointRingMCP]
        all_joints[JointID.JPinkyMCP] = active_joint_pos[ActiveJointID.ActiveJointPinkyMCP]

        # Calculate passive joints using polynomial models
        all_joints[JointID.JThumbDIP] = self._predict_poly(all_joints[JointID.JThumbPIP], self._thumb_dip_coeff)
        all_joints[JointID.JIndexDIP] = self._predict_poly(all_joints[JointID.JIndexPIP], self._index_dip_coeff)
        all_joints[JointID.JMiddleDIP] = self._predict_poly(all_joints[JointID.JMiddlePIP], self._middle_dip_coeff)
        
        # Ring and Pinky PIP/DIP are coupled to their MCP joints
        ring_mcp = all_joints[JointID.JRingMCP]
        pinky_mcp = all_joints[JointID.JPinkyMCP]
        all_joints[JointID.JRingPIP] = self._predict_poly(ring_mcp, self._ring_pip_coeff)
        all_joints[JointID.JRingDIP] = self._predict_poly(ring_mcp, self._ring_dip_coeff)
        all_joints[JointID.JPinkyPIP] = self._predict_poly(pinky_mcp, self._ring_pip_coeff) # Uses same coeff as ring
        all_joints[JointID.JPinkyDIP] = self._predict_poly(pinky_mcp, self._ring_dip_coeff) # Uses same coeff as ring

        return all_joints

    # --- Internal "Private" Helper Methods ---

    def _active_joint_to_motor_length(self, active_joint_pos: np.ndarray) -> np.ndarray:
        """
        Converts active joint positions to motor lengths for the O12 hand prosthetic.
        """
        joint_pos = active_joint_pos.copy()
        self._clamp(self._active_joint_max, self._active_joint_min, joint_pos)

        # Apply hand-specific sign conventions
        if self._is_right_hand:
            # FIXME: this is a hack to match the potential joint mismatch.
            joint_pos[ActiveJointID.ActiveJointThumbMCP] = -0.8312 - joint_pos[ActiveJointID.ActiveJointThumbMCP]
            joint_pos[ActiveJointID.ActiveJointThumbPIP] = -1.3 - joint_pos[ActiveJointID.ActiveJointThumbPIP]

            joint_pos[ActiveJointID.ActiveJointThumbAbAd] *= -1
            joint_pos[ActiveJointID.ActiveJointThumbMCP] *= -1
            joint_pos[ActiveJointID.ActiveJointThumbPIP] *= -1
        else: # Left Hand
            # FIXME: this is a hack to match the potential joint mismatch.
            joint_pos[ActiveJointID.ActiveJointThumbMCP] = -0.8312 - joint_pos[ActiveJointID.ActiveJointThumbMCP]
            joint_pos[ActiveJointID.ActiveJointThumbPIP] = -1.3 - joint_pos[ActiveJointID.ActiveJointThumbPIP]

            joint_pos[ActiveJointID.ActiveJointThumbRoll] *= -1
            joint_pos[ActiveJointID.ActiveJointIndexAbAd] *= -1
            joint_pos[ActiveJointID.ActiveJointMiddleABAD] *= -1
            joint_pos[ActiveJointID.ActiveJointThumbMCP] *= -1
            joint_pos[ActiveJointID.ActiveJointThumbPIP] *= -1

        motor_length = np.zeros(ACTUATOR_COUNT, dtype=np.float64)
        
        # Thumb (units: input rad, output m)
        motor_length[O12handProActuator.ActuatorThumbRoll] = self._predict_poly(joint_pos[ActiveJointID.ActiveJointThumbRoll], self._coeffs_thumb_roll) * 1e-3
        motor_length[O12handProActuator.ActuatorThumbABAD] = self._predict_poly(joint_pos[ActiveJointID.ActiveJointThumbAbAd], self._coeffs_thumb_abad) * 1e-3
        motor_length[O12handProActuator.ActuatorThumbMCP] = self._predict_poly(joint_pos[ActiveJointID.ActiveJointThumbMCP], self._coeffs_thumb_mcp) * 1e-3
        motor_length[O12handProActuator.ActuatorThumbPIP] = self._predict_poly(joint_pos[ActiveJointID.ActiveJointThumbPIP], self._coeffs_thumb_pip) * 1e-3

        # Index Finger
        idx_abad, idx_mcp, idx_pip = joint_pos[ActiveJointID.ActiveJointIndexAbAd], joint_pos[ActiveJointID.ActiveJointIndexMCP], joint_pos[ActiveJointID.ActiveJointIndexPIP]
        motor_length[O12handProActuator.ActuatorIndex1] = self._predict_poly33(idx_abad, idx_mcp, self._coeffs_index1)
        motor_length[O12handProActuator.ActuatorIndex2] = self._predict_poly33(idx_abad, idx_mcp, self._coeffs_index2)
        motor_length[O12handProActuator.ActuatorIndex3] = self._finger_pip_fit_predict(idx_abad, idx_mcp, idx_pip, self._index_pip_coeffs)

        # Middle Finger
        mid_abad, mid_mcp, mid_pip = joint_pos[ActiveJointID.ActiveJointMiddleABAD], joint_pos[ActiveJointID.ActiveJointMiddleMCP], joint_pos[ActiveJointID.ActiveJointMiddlePIP]
        motor_length[O12handProActuator.ActuatorMiddle1] = self._predict_poly33(mid_abad, mid_mcp, self._coeffs_index1) # Reuses index coeffs
        motor_length[O12handProActuator.ActuatorMiddle2] = self._predict_poly33(mid_abad, mid_mcp, self._coeffs_index2) # Reuses index coeffs
        motor_length[O12handProActuator.ActuatorMiddle3] = self._finger_pip_fit_predict(mid_abad, mid_mcp, mid_pip, self._middle_pip_coeffs)

        # Ring and Pinky
        motor_length[O12handProActuator.ActuatorRing] = self._predict_poly(joint_pos[ActiveJointID.ActiveJointRingMCP], self._coeffs_ring_pinky) * 1e-3
        motor_length[O12handProActuator.ActuatorPinky] = self._predict_poly(joint_pos[ActiveJointID.ActiveJointPinkyMCP], self._coeffs_ring_pinky) * 1e-3

        if DBG_O12_CONTROL_LIMITS:
            global ACTIVE_MOTOR_LENGTH_MAX, ACTIVE_MOTOR_LENGTH_MIN
            ACTIVE_MOTOR_LENGTH_MAX = np.maximum(ACTIVE_MOTOR_LENGTH_MAX, motor_length)
            ACTIVE_MOTOR_LENGTH_MIN = np.minimum(ACTIVE_MOTOR_LENGTH_MIN, motor_length)
            cprint.ok(f"Motor Length Max: {ACTIVE_MOTOR_LENGTH_MAX}, Min: {ACTIVE_MOTOR_LENGTH_MIN}")
        self._clamp(self._motor_length_max, self._motor_length_min, motor_length)
        return motor_length

    def _motor_length_to_active_joint(self, motor_length: np.ndarray) -> np.ndarray:
        """
        Converts motor lengths to active joint positions for the O12 hand prosthetic.
        """
        ml = motor_length.copy()
        self._clamp(self._motor_length_max, self._motor_length_min, ml)
        
        active_joint_pos = np.zeros(MAX_ACTIVE_JOINT, dtype=np.float64)

        # Thumb (units: input m, output rad)
        active_joint_pos[ActiveJointID.ActiveJointThumbRoll] = self._predict_poly(ml[O12handProActuator.ActuatorThumbRoll] * 1e3, self._coeffs_thumb_roll_motor2joint)
        active_joint_pos[ActiveJointID.ActiveJointThumbAbAd] = self._predict_poly(ml[O12handProActuator.ActuatorThumbABAD] * 1e3, self._coeffs_thumb_abad_motor2joint)
        active_joint_pos[ActiveJointID.ActiveJointThumbMCP] = self._predict_poly(ml[O12handProActuator.ActuatorThumbMCP] * 1e3, self._coeffs_thumb_mcp_motor2joint)
        active_joint_pos[ActiveJointID.ActiveJointThumbPIP] = self._predict_poly(ml[O12handProActuator.ActuatorThumbPIP] * 1e3, self._coeffs_thumb_pip_motor2joint)

        # Index Finger
        idx_abad = self._predict_poly33(ml[O12handProActuator.ActuatorIndex1], ml[O12handProActuator.ActuatorIndex2], self._coeffs_index1_motor2joint)
        idx_mcp = self._predict_poly33(ml[O12handProActuator.ActuatorIndex1], ml[O12handProActuator.ActuatorIndex2], self._coeffs_index2_motor2joint)
        idx_pip = self._finger_pip_fit_predict(idx_abad, idx_mcp, ml[O12handProActuator.ActuatorIndex3], self._index_pip_coeffs_motor2joint)
        active_joint_pos[ActiveJointID.ActiveJointIndexAbAd], active_joint_pos[ActiveJointID.ActiveJointIndexMCP], active_joint_pos[ActiveJointID.ActiveJointIndexPIP] = idx_abad, idx_mcp, idx_pip

        # Middle Finger
        mid_abad = self._predict_poly33(ml[O12handProActuator.ActuatorMiddle1], ml[O12handProActuator.ActuatorMiddle2], self._coeffs_index1_motor2joint) # Reuses index coeffs
        mid_mcp = self._predict_poly33(ml[O12handProActuator.ActuatorMiddle1], ml[O12handProActuator.ActuatorMiddle2], self._coeffs_index2_motor2joint) # Reuses index coeffs
        mid_pip = self._finger_pip_fit_predict(mid_abad, mid_mcp, ml[O12handProActuator.ActuatorMiddle3], self._middle_pip_coeffs_motor2joint)
        active_joint_pos[ActiveJointID.ActiveJointMiddleABAD], active_joint_pos[ActiveJointID.ActiveJointMiddleMCP], active_joint_pos[ActiveJointID.ActiveJointMiddlePIP] = mid_abad, mid_mcp, mid_pip

        # Ring and Pinky
        active_joint_pos[ActiveJointID.ActiveJointRingMCP] = self._predict_poly(ml[O12handProActuator.ActuatorRing] * 1e3, self._coeffs_ring_pinky_motor2joint)
        active_joint_pos[ActiveJointID.ActiveJointPinkyMCP] = self._predict_poly(ml[O12handProActuator.ActuatorPinky] * 1e3, self._coeffs_ring_pinky_motor2joint)

        # Apply hand-specific sign conventions
        if self._is_right_hand:
            active_joint_pos[ActiveJointID.ActiveJointThumbAbAd] *= -1
            active_joint_pos[ActiveJointID.ActiveJointThumbMCP] *= -1
            active_joint_pos[ActiveJointID.ActiveJointThumbPIP] *= -1

            active_joint_pos[ActiveJointID.ActiveJointThumbMCP] = -0.8312 - active_joint_pos[ActiveJointID.ActiveJointThumbMCP]
            active_joint_pos[ActiveJointID.ActiveJointThumbPIP] = -1.3 - active_joint_pos[ActiveJointID.ActiveJointThumbPIP]
        else: # Left Hand
            active_joint_pos[ActiveJointID.ActiveJointIndexAbAd] *= -1
            active_joint_pos[ActiveJointID.ActiveJointMiddleABAD] *= -1
            active_joint_pos[ActiveJointID.ActiveJointThumbMCP] *= -1
            active_joint_pos[ActiveJointID.ActiveJointThumbPIP] *= -1
        
        self._clamp(self._active_joint_max, self._active_joint_min, active_joint_pos)
        return active_joint_pos

    def _motor_length_to_motor_input(self, motor_length: np.ndarray) -> np.ndarray:
        return self._scale(self._motor_length_max, self._motor_length_min, motor_length, self._motor_input_max, self._motor_input_min)

    def _motor_input_to_motor_length(self, motor_input: np.ndarray) -> np.ndarray:
        return self._scale(self._motor_input_max, self._motor_input_min, motor_input, self._motor_length_max, self._motor_length_min)

    def _clamp(self, max_vals: np.ndarray, min_vals: np.ndarray, values: np.ndarray):
        """Clamps the `values` array in-place, handling inverted min/max ranges."""
        for i in range(values.size):
            low, high = min_vals[i], max_vals[i]
            if low > high:
                low, high = high, low
            values[i] = np.clip(values[i], low, high)

    def _scale(self, max_in, min_in, value_in, max_out, min_out):
        """Linearly scales an input value from one range to another."""
        # Clamp the input value to its valid range first
        clamped_value = value_in.copy()
        self._clamp(max_in, min_in, clamped_value)
        
        # Perform scaling
        range_in = max_in - min_in
        range_out = max_out - min_out
        
        # Avoid division by zero
        # Create a mask for non-zero range_in to prevent warnings/errors
        non_zero_mask = range_in != 0
        
        result = np.zeros_like(value_in, dtype=np.float64)
        
        # Calculate ratio only for elements with a valid input range
        ratio = np.zeros_like(value_in, dtype=np.float64)
        ratio[non_zero_mask] = (clamped_value[non_zero_mask] - min_in[non_zero_mask]) / range_in[non_zero_mask]
        
        result = min_out + ratio * range_out
        
        # For integer output types, round and cast
        if np.issubdtype(max_out.dtype, np.integer):
            return np.round(result).astype(max_out.dtype)
        return result.astype(max_out.dtype)

    def _predict_poly(self, x: float, coeffs: np.ndarray) -> float:
        """Predicts value using a 1D polynomial. Coeffs are highest power first."""
        return np.polyval(coeffs, x)

    def _predict_poly33(self, x: float, y: float, coeffs: np.ndarray) -> float:
        """Predicts value using a 3rd order bivariate polynomial."""
        p = coeffs
        return (p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*x*y + p[5]*y**2 +
                p[6]*x**3 + p[7]*x**2*y + p[8]*x*y**2 + p[9]*y**3)

    def _finger_pip_fit_predict(self, abad: float, mcp: float, pip: float, coeffs: np.ndarray) -> float:
        """Predicts value using a complex multivariate polynomial fit."""
        X = np.array([
            1.0, abad, mcp, pip,
            abad**2, mcp**2, pip**2,
            abad**3, mcp**3, pip**3,
            abad**4, mcp**4, pip**4,
            abad*mcp, abad*pip, mcp*pip,
            abad**2*mcp, abad**2*pip,
            mcp**2*abad, mcp**2*pip,
            pip**2*abad, pip**2*mcp,
            abad*mcp*pip
        ])
        return np.dot(X, coeffs)

    def _check_and_clamp_finger_workspace(self, joint_pos: np.ndarray):
        """Checks if index/middle finger joints (ABAD and MCP) remain within a defined triangular workspace and clamps them if they go outside.."""
        p0 = (0, 0)
        p1 = (0.94, 1.5)
        p2 = (-0.94, 1.5)

        # Check Index Finger
        p_index = [joint_pos[ActiveJointID.ActiveJointIndexAbAd], joint_pos[ActiveJointID.ActiveJointIndexMCP]]
        if not self._is_in_parallel(p_index, p0, p1, p2):
            # cprint.warn(f"Index finger Abad/MCP outside valid workspace. Clamping index finger point {p_index} to parallel workspace dfefined by {p0}, {p1}, {p2}.")
            p_clamped = self._clamp_point_to_parallel(p_index, p0, p1, p2)
            joint_pos[ActiveJointID.ActiveJointIndexAbAd], joint_pos[ActiveJointID.ActiveJointIndexMCP] = p_clamped

        # Check Middle Finger
        p_middle = [joint_pos[ActiveJointID.ActiveJointMiddleABAD], joint_pos[ActiveJointID.ActiveJointMiddleMCP]]
        if not self._is_in_parallel(p_middle, p0, p1, p2):
            # cprint.warn(f"Middle finger Abad/MCP outside valid workspace. Clamping middle finger point {p_middle} to parallel workspace defined by {p0}, {p1}, {p2}.")
            p_clamped = self._clamp_point_to_parallel(p_middle, p0, p1, p2)
            joint_pos[ActiveJointID.ActiveJointMiddleABAD], joint_pos[ActiveJointID.ActiveJointMiddleMCP] = p_clamped

    def _is_in_parallel(self, p, p0, p1, p2):
        """
        1. Creates vectors from p0 to p1 and p0 to p2.
        2. Calculates the determinant to check if triangle is degenerate.
        3. Uses Barycentric coordinates (a,b) to check if point p is within the triangle formed by p0, p1, p2.
        4. Point is inside if 0 <= a <= 1 and 0 <= b <= 1.
        5. Returns True if point is inside, False otherwise.
        """
        v0 = (p1[0] - p0[0], p1[1] - p0[1])
        v1 = (p2[0] - p0[0], p2[1] - p0[1])
        vp = (p[0] - p0[0], p[1] - p0[1])
        det = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(det) < 1e-9: return False
        a = (vp[0] * v1[1] - vp[1] * v1[0]) / det
        b = (v0[0] * vp[1] - v0[1] * vp[0]) / det
        return 0 <= a <= 1 and 0 <= b <= 1

    def _clamp_point_to_parallel(self, p, p0, p1, p2):
        # This is a simplified clamping logic. A full geometric projection would be more complex.
        # For this port, we will just clamp the components independently as a robust fallback.
        x_min = min(p0[0], p1[0], p2[0])
        x_max = max(p0[0], p1[0], p2[0])
        y_min = min(p0[1], p1[1], p2[1])
        y_max = max(p0[1], p1[1], p2[1])
        p[0] = np.clip(p[0], x_min, x_max)
        p[1] = np.clip(p[1], y_min, y_max)
        return p
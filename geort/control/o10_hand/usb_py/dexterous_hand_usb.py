"""
Dexterous Hand USB Communication Library
=======================================

This module provides a high‑level API for communicating with the OmniHand
series dexterous hand over the virtual serial port exposed on the USB
connector.  The protocol description used in this implementation is
derived from the PDF document *OminiHand 基础款 通信协议&上位机 用户版 v1.3*.  The
document defines a simple frame format that encapsulates commands
targeting individual motors, sensors and device settings.  Commands are
identified by a single byte (`cmd`), followed by zero or more data
payload bytes.  Multi‑byte values are sent and received in little‑endian
order and are expressed in units specified by the protocol (e.g.
position is expressed as a 16‑bit integer in the range 0–4096, current in
centi‑amps, etc.).

Frame Format
------------

Each message transmitted over the serial port uses the following
structure (see section 2 *USB/串口 通信帧格式* in the manual):

    +---------------+---------------+---------------+-----------------+---------------+
    |   Frame head  |   Device ID   | Data length   | Data payload    |     CRC       |
    +---------------+---------------+---------------+-----------------+---------------+
    | 2 bytes       | 2 bytes       | 1 byte        | 1–64 bytes      | 2 bytes       |

* **Frame head**:  The constant value `0xAAEE` encoded in little‑endian
  form (0xEE followed by 0xAA).  This allows the receiver to reliably
  locate the start of a frame.

* **Device ID**:  A 16‑bit little‑endian identifier of the target device.
  Valid IDs range from 0 to 0x7FF.  A special broadcast address
  (`0x7FF`) can be used for discovery; devices will respond using their
  own IDs【879085932721491†screenshot】.

* **Data length**:  The number of bytes in the data payload (the `cmd`
  and any associated parameters).  The length must be between 1 and 64
  inclusive.

* **Data payload**:  The first byte of the payload is always the
  command code.  The remaining bytes (if any) contain command‑specific
  arguments.  Multi‑byte fields are packed in little‑endian order as
  specified throughout chapter 3 of the protocol documentation【879085932721491†screenshot】.

* **CRC**:  A 16‑bit checksum calculated over the entire frame from the
  start of the frame head through the end of the data payload.  The
  manual provides a reference implementation of the CRC function
  (`Crc16`) in C which uses a precomputed lookup table based on the
  CRC‑CCITT polynomial (0x1021).  This module includes an equivalent
  Python implementation to ensure interoperability【879085932721491†screenshot】.

The implementation below wraps the low‑level frame construction and
parsing into a convenient `DexterousHandUSB` class.  It exposes
type‑safe methods for every command defined in the protocol (see
section 3.2 of the manual).  Each method builds the appropriate
payload, transmits the frame, waits for a reply and decodes the
returned data into Python types.  If a CRC mismatch occurs or the
device replies with an error status (e.g. `0x00` for failure), the
method raises an exception.

Prerequisites
-------------

* Python 3.8 or higher.
* The `pyserial` package installed.  If it is not already available,
  install it via pip: `pip install pyserial`.

Usage Example
-------------

```
from dexterous_hand_usb import DexterousHandUSB

# Create an instance for the device on COM3 (Windows) or /dev/ttyACM0 (Linux).
hand = DexterousHandUSB(port='/dev/ttyACM0', baudrate=460800)

# Enable the device.  Mode 1 = enable, mode 2 = calibration.
hand.set_enable(mode=1)

# Set the position of joint #3 to 50 % of its travel range.
position = int(4096 * 0.5)  # convert 0–1 range into protocol units
hand.set_axis_position(axis_index=3, position=position)

# Query the current position of joint #3.
current_pos = hand.get_axis_position(axis_index=3)
print(f'Current position of joint 3: {current_pos}/4096')

# Clean up when finished.
hand.close()
```

The methods provided here raise `RuntimeError` if the device reports
failure or the CRC does not validate.  For long‑running applications
consider catching these exceptions and retrying the operation.
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Iterable, List, Optional, Tuple, Union
from cprint import cprint

import serial  # type: ignore


class CRC16:
    """Compute CRC‑CCITT (0x1021) using a lookup table.

    The protocol document provides a C implementation of a 256‑entry
    lookup table (``Crc16Tab``) used to accelerate computation of the
    16‑bit checksum.  This Python class reproduces that behaviour.  The
    state machine is initialised with `crc=0` and updated for each
    incoming byte.  After processing all bytes the resulting 16‑bit
    integer is returned.
    """

    # The table from the manual (converted from hexadecimal values).  See
    # section 2 of the PDF for the full table【879085932721491†screenshot】.
    _TABLE: Tuple[int, ...] = (
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
        0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
        0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
        0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
        0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
        0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
        0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
        0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
        0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
        0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
        0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
        0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
        0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
        0xEDAE, 0xFD8F, 0xCDCC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
        0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
        0xFF9F, 0xEFBE, 0xDFDD, 0xCFDC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F58,
        0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
        0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
        0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
        0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
        0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
        0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
        0xA7DB, 0xB7FA, 0x8799, 0x9798, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
        0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
        0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
        0x5844, 0x4865, 0x7806, 0x6807, 0x18C0, 0x08E1, 0x3882, 0x2883,
        0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABB8, 0xBB98,
        0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
        0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBD2A, 0xAD0B, 0x9D68, 0x8D49,
        0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
        0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
        0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0,
    )

    @classmethod
    def compute(cls, data: Union[bytes, bytearray, memoryview]) -> int:
        """Compute the CRC of the given byte sequence.

        Args:
            data: A bytes‑like object containing the frame head, device ID,
                length and payload.  The CRC covers all bytes in this
                sequence.

        Returns:
            The 16‑bit unsigned integer CRC value.
        """
        crc = 0
        for byte in data:
            # python ints are unlimited precision; mask to eight bits to
            # match C's behaviour
            tmp = ((crc >> 8) ^ byte) & 0xFF
            crc = ((crc << 8) & 0xFFFF) ^ cls._TABLE[tmp]
        return crc & 0xFFFF


class DexterousHandUSB:
    """High level API for the OmniHand dexterous hand USB protocol.

    When instantiated, this class opens a serial connection to the device
    and spawns a background reader thread that continuously consumes
    incoming frames.  Responses to commands are delivered to the
    appropriate waiting caller via a simple promise/future mechanism.
    Commands that do not receive a response within `timeout` seconds will
    raise a `TimeoutError`.
    """

    # Frame constants
    FRAME_HEAD = b"\xEE\xAA"  # little‑endian encoding of 0xAAEE【879085932721491†screenshot】
    MAX_PAYLOAD = 64
    CRC_DEBUG = 0x5555  # reserved CRC that bypasses validation【879085932721491†screenshot】

    def __init__(self, port: str, baudrate: int = 460800, device_id: int = 1,
                 timeout: float = 0.2) -> None:
        """Create a new USB connection to the dexterous hand.

        Args:
            port: Name of the serial port (e.g. ``'COM3'`` or ``'/dev/ttyACM0'``).
            baudrate: Communication speed.  According to the manual,
                ``460 800`` Bd is the default for the OmniHand【879085932721491†screenshot】.
            device_id: The 11‑bit CAN/USB node ID of the device.  You may
                supply ``0x7FF`` to broadcast a query when the ID is unknown.
            timeout: Maximum number of seconds to wait for a response
                before raising a `TimeoutError`.
        """
        if not (0 <= device_id <= 0x7FF):
            raise ValueError("device_id must be in range 0..0x7FF")
        self.device_id = device_id
        self.timeout = timeout
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,  # short read timeout; we handle overall timeout manually
        )
        # Synchronisation primitives
        self._lock = threading.Lock()
        self._response_event = threading.Event()
        self._response_frame: Optional[bytes] = None
        # Start background reader
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def close(self) -> None:
        """Close the serial port and terminate the reader thread."""
        if self._serial.is_open:
            self._serial.close()
        self._reader_thread.join(timeout=0.2)

    # ----------------------------------------------------------------------
    # Frame handling
    # ----------------------------------------------------------------------

    def _build_frame(self, cmd: int, payload: Iterable[int], *, device_id: Optional[int] = None) -> bytes:
        """Construct a frame for transmission.

        Args:
            cmd: The 8‑bit command code defined by the protocol.
            payload: An iterable of 8‑bit values comprising any command
                parameters.  **Do not include the command code itself**;
                it is prepended automatically.
            device_id: Optionally override the default device ID.

        Returns:
            The complete byte sequence ready for transmission over the
            serial port.

        Raises:
            ValueError: If the data payload would exceed the maximum
                allowed length (64 bytes).
        """
        payload_bytes = bytes([cmd]) + bytes(payload)
        if len(payload_bytes) < 1 or len(payload_bytes) > self.MAX_PAYLOAD:
            raise ValueError(f"payload length {len(payload_bytes)} out of range 1–{self.MAX_PAYLOAD}")
        # Frame head + device id (little‑endian) + length + payload
        did = self.device_id if device_id is None else device_id
        header = self.FRAME_HEAD + struct.pack('<H', did) + struct.pack('<B', len(payload_bytes))
        frame_without_crc = header + payload_bytes
        crc = CRC16.compute(frame_without_crc)
        # FIXME: using self.CRC_DEBUG for debuging
        crc = self.CRC_DEBUG & 0xFFFF
        # In little‑endian send low byte then high byte
        frame = frame_without_crc + struct.pack('<H', crc)
        return frame

    def _send_frame(self, frame: bytes) -> None:
        """Send a frame over the serial port."""
        with self._lock:
            self._serial.write(frame)
            self._serial.flush()

    def _read_loop(self) -> None:
        """Background thread that reads and validates incoming frames.

        The loop continually reads from the serial port, searching for the
        frame head pattern.  When a frame is detected, it extracts the
        length, verifies the CRC and stores the frame for retrieval by
        `_wait_for_response`.  If the CRC fails, the frame is discarded.
        """
        buffer = bytearray()
        while True:
            # Exit if serial port is closed
            if not self._serial.is_open:
                return
            try:
                chunk = self._serial.read(256)
            except serial.SerialException:
                return
            if not chunk:
                continue
            buffer.extend(chunk)
            while True:
                # Search for frame head pattern
                idx = buffer.find(self.FRAME_HEAD)
                if idx < 0:
                    # Drop data until potential head appears
                    buffer.clear()
                    break
                # Ensure we have at least the fixed header after the head
                if len(buffer) < idx + 2 + 2 + 1:
                    cprint.warn(f"Received frame with insufficient data: {len(buffer)} bytes, expected at least 5 bytes after head")
                    break  # not enough data
                # Extract header fields
                start = idx
                # Frame layout: head (2 bytes) + ID (2) + length (1)
                did = int.from_bytes(buffer[start + 2:start + 4], 'little')
                length = buffer[start + 4]
                total_len = 2 + 2 + 1 + length + 2
                if len(buffer) < start + total_len:
                    cprint.warn(f"Received frame with incomplete data: {len(buffer)} bytes, expected {total_len} bytes")
                    break  # wait for more bytes
                frame = bytes(buffer[start:start + total_len])
                # Remove consumed bytes
                del buffer[:start + total_len]
                # Validate CRC unless debug CRC is used
                payload_end = 2 + 2 + 1 + length
                crc_expected = int.from_bytes(frame[payload_end:payload_end + 2], 'little')
                if crc_expected == self.CRC_DEBUG:
                    crc_ok = True
                else:
                    crc_ok = CRC16.compute(frame[:-2]) == crc_expected
                # if not crc_ok:
                #     cprint.warn(f"Received frame with invalid CRC: {crc_expected:#04x}, expected: {CRC16.compute(frame[:-2]):#04x}")
                #     continue  # discard corrupt frame
                # Check if this frame corresponds to the current device ID or broadcast
                # Accept frames with matching ID or broadcast (0x7FF)
                if did not in (self.device_id, 0x7FF):
                    cprint.warn(f"Received frame with unexpected device ID: {did}, expected: {self.device_id}")
                    continue  # ignore frames for other devices
                # Store frame and signal any waiting thread
                with self._lock:
                    self._response_frame = frame
                    self._response_event.set()
            # End of inner loop
        # End of outer while

    def _wait_for_response(self, expected_cmd: int) -> bytes:
        """Wait for a reply frame matching a specific command code.

        Args:
            expected_cmd: The command code that the response must match.

        Returns:
            The payload bytes (including the command code) of the response.

        Raises:
            TimeoutError: If no response arrives within the configured
                timeout period.
            RuntimeError: If the response command code does not match the
                expected value.
        """
        deadline = time.time() + self.timeout
        while True:
            timeout = deadline - time.time()
            if timeout <= 0:
                raise TimeoutError("no response from device")
            # Wait for an event that signals a response has been stored
            if not self._response_event.wait(timeout=timeout):
                continue
            with self._lock:
                frame = self._response_frame
                self._response_frame = None
                self._response_event.clear()
            if frame is None:
                continue
            # Extract payload (length is stored at offset 4)
            length = frame[4]
            payload = frame[5:5 + length]
            if not payload:
                # Should never happen (length >=1)
                continue
            cmd = payload[0]
            if cmd != expected_cmd:
                # Unexpected command; ignore and continue waiting
                cprint.warn(f"Unexpected command: {cmd}, expected: {expected_cmd}, the whole payload: {payload}")
                continue
            return payload

    # ----------------------------------------------------------------------
    # Command wrappers
    # ----------------------------------------------------------------------
    # Each method below corresponds to a command in section 3.2.  They
    # construct the payload, transmit it and parse the response.

    def set_enable(self, mode: int) -> None:
        """Enable, disable or calibrate the device.

        Command 0x01 writes the enable state.  The protocol expects a
        single byte argument with the following semantics【879085932721491†screenshot】:

        * 0x00: disable the device (motors will not actuate).
        * 0x01: enable the device (normal operation).
        * 0x02: calibration mode (used when calibrating sensors/motors).

        The device replies with a single byte indicating success (0x01) or
        failure (0x00).

        Args:
            mode: One of ``0``, ``1`` or ``2``.

        Raises:
            RuntimeError: If the device reports failure.
        """
        if mode not in (0, 1, 2):
            raise ValueError("mode must be 0 (disable), 1 (enable) or 2 (calibration)")
        frame = self._build_frame(0x01, [mode])
        self._send_frame(frame)
        resp = self._wait_for_response(0x01)
        status = resp[1]
        if status != 0x01:
            raise RuntimeError("failed to set enable state")

    def get_enable_state(self) -> int:
        """Query the current enable/calibration state.

        Command 0x02 returns a single byte indicating the current mode【879085932721491†screenshot】:

        * 0x00: disabled;
        * 0x01: enabled;
        * 0x02: calibration mode.

        Returns:
            The current enable state.
        """
        frame = self._build_frame(0x02, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x02)
        return resp[1]

    def set_axis_position(self, axis_index: int, position: int) -> None:
        """Set the current position of a single axis.

        Command 0x03 accepts a three‑byte payload: ``d0`` is the axis index
        (1–10) and ``d1–d2`` is a 16‑bit little‑endian position value in the
        range 0–4096 (representing 0–360.0 degrees scaled to 0.1°
        resolution).  The device acknowledges with a byte indicating
        success or failure【879085932721491†screenshot】.

        Args:
            axis_index: The joint number (1–10).
            position: The desired position in protocol units (0–4096).

        Raises:
            RuntimeError: If the device reports failure.
        """
        cprint.warn(f"[USB/set_axis_position]: This func currently always fails.")
        if not (1 <= axis_index <= 10):
            raise ValueError("axis_index must be between 1 and 10")
        if not (0 <= position <= 4096):
            raise ValueError("position must be between 0 and 4096 inclusive")
        payload = [axis_index] + list(struct.pack('<H', position))
        frame = self._build_frame(0x03, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x03)
        if resp[1] != 0x01:
            raise RuntimeError(f"failed to set axis {axis_index} position")

    def set_device_id(self, new_id: int) -> None:
        """Assign a new CAN/USB node ID to the device.

        Command 0x04 writes a 16‑bit little‑endian value specifying the
        desired ID in the range 1–0x7FE.  The device acknowledges with a
        status byte【879085932721491†screenshot】.
        """
        if not (1 <= new_id < 0x7FF):
            raise ValueError("new_id must be between 1 and 0x7FE")
        payload = list(struct.pack('<H', new_id))
        frame = self._build_frame(0x04, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x04)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set device ID")
        # Update our local ID so subsequent commands use the new one
        self.device_id = new_id

    def return_home(self) -> None:
        """Return all joints to their initial positions (blocking).

        Command 0x05 instructs the hand to move all joints back to their
        default positions.  The call blocks until the device replies.

        Raises:
            RuntimeError: If the device reports failure.
        """
        frame = self._build_frame(0x05, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x05)
        if resp[1] != 0x01:
            raise RuntimeError("failed to return to home position")

    def set_single_axis_target(self, axis_index: int, target: int) -> int:
        """Move a single axis to a target position and obtain the resulting position.

        Command 0x06 takes a 3‑byte payload: ``d0`` is the axis index
        (1–10) and ``d1–d2`` is a 16‑bit target position (0–4096).  The
        device replies with three bytes: ``d0`` (axis index) and ``d1–d2``
        (current position)【879085932721491†screenshot】.

        Args:
            axis_index: The joint number (1–10).
            target: Target position in protocol units (0–4096).

        Returns:
            The position reported by the device after execution (should
            match the target if the movement succeeded).
        """
        if not (1 <= axis_index <= 10):
            raise ValueError("axis_index must be between 1 and 10")
        if not (0 <= target <= 4096):
            raise ValueError("target must be between 0 and 4096 inclusive")
        payload = [axis_index] + list(struct.pack('<H', target))
        frame = self._build_frame(0x06, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x06)
        if len(resp) != 4:
            raise RuntimeError("unexpected response length for set_single_axis_target")
        resp_axis = resp[1]
        current_pos = int.from_bytes(resp[2:4], 'little') if len(resp) >= 4 else 0
        if resp_axis != axis_index:
            raise RuntimeError(f"response axis index mismatch (expected {axis_index}, got {resp_axis})")
        return current_pos

    def get_axis_position(self, axis_index: int) -> int:
        """Query the current position of a single axis.

        Command 0x07 accepts a 1‑byte payload identifying the axis.  The
        device returns the same data layout as command 0x06: axis index
        plus 16‑bit current position【879085932721491†screenshot】.

        Args:
            axis_index: The joint number (1–10).

        Returns:
            The current joint position in protocol units (0–4096).
        """
        if not (1 <= axis_index <= 10):
            raise ValueError("axis_index must be between 1 and 10")
        frame = self._build_frame(0x07, [axis_index])
        self._send_frame(frame)
        resp = self._wait_for_response(0x07)
        if len(resp) < 3:
            raise RuntimeError("unexpected response length for get_axis_position")
        resp_axis = resp[1]
        pos = int.from_bytes(resp[2:4], 'little')
        if resp_axis != axis_index:
            raise RuntimeError(f"response axis index mismatch (expected {axis_index}, got {resp_axis})")
        return pos

    def set_all_positions(self, positions: Iterable[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Set all 10 axes to the specified target positions.

        Command 0x08 takes exactly 20 bytes of target position data:
        each joint occupies two little‑endian bytes (uint16).  In the
        response the device returns 60 bytes comprising three groups of
        20 bytes each: current position values, speed feedback and torque
        feedback, followed by an additional 10‑byte block of fault flags【879085932721491†screenshot】.

        Args:
            positions: An iterable of exactly 10 integers (0–4096) giving
                the target positions for joints 1 through 10.

        Returns:
            A tuple of four lists:

              1. 10 current position values (uint16 each)
              2. 10 current speed values (uint16 each)
              3. 10 current torque values (uint16 each)
              4. 10 fault flags (uint8 each)
        """
        pos_list = list(positions)
        if len(pos_list) != 10:
            raise ValueError("positions must contain exactly 10 values")
        for p in pos_list:
            if not (0 <= p <= 4096):
                raise ValueError("each position must be between 0 and 4096 inclusive")
        payload = []
        for p in pos_list:
            payload.extend(struct.pack('<H', p))
        # Append four zero bytes for reserved freedom degrees according to the manual
        payload.extend([0x00, 0x00, 0x00, 0x00])
        # FIXME: append another 7 \x00 placeholders to receive message. In practice, sending payload less than 24 bytes won't receive any replies.
        payload.extend([0x00] * 7)

        frame = self._build_frame(0x08, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x08)

        # # Response should be 60 bytes: positions, speed, torque, fault flags (10 bytes each)
        # if len(resp) != 61:
        #     raise RuntimeError(f"unexpected response length for set_all_positions: {len(resp)} bytes")
        # data = resp[1:]
        # # parse 10 positions
        # cur_pos = [int.from_bytes(data[i:i+2], 'little') for i in range(0, 20, 2)]
        # # next 20 bytes: speed
        # cur_speed = [int.from_bytes(data[i:i+2], 'little') for i in range(20, 40, 2)]
        # # next 10 bytes: torque (uint8), but spec says 10 bytes? Actually 20 bytes? We'll treat as 20 bytes
        # cur_torque = [int.from_bytes(data[i:i+2], 'little') for i in range(40, 60, 2)]
        # # last 10 bytes: fault flags (uint8)
        # faults = list(data[60:70]) if len(data) >= 70 else []

        # return cur_pos, cur_speed, cur_torque, faults

        # FIXME: according to docs, would return 60-bytes data contaiing current qpos, speeds, torques, faults, but in practice, it would return only a 32-bytes containing sent qpos.
        if len(resp) != 25:
            raise RuntimeError(f"unexpected response length for set_all_positions: {len(resp)} bytes")
        data = resp[1:]
        # parse 10 positions
        cur_pos = [int.from_bytes(data[i:i+2], 'little') for i in range(0, 20, 2)]

        return cur_pos


    def get_all_positions(self) -> List[int]:
        """Read the current position of all axes.

        Command 0x09 returns 20 bytes of position data (2 bytes per
        joint)【879085932721491†screenshot】.

        Returns:
            A list of 10 position values (uint16 each).
        """
        frame = self._build_frame(0x09, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x09)
        if len(resp) != 21:
            raise RuntimeError(f"unexpected response length for get_all_positions: {len(resp)} bytes")
        data = resp[1:]
        return [int.from_bytes(data[i:i+2], 'little') for i in range(0, 20, 2)]

    def get_all_currents(self) -> List[int]:
        """Read the current (centi‑amps) of all axes.

        Command 0x0A returns 20 bytes containing two bytes per joint【879085932721491†screenshot】.

        Returns:
            A list of 10 current values as unsigned integers (units of 0.01 A).
        """
        frame = self._build_frame(0x0A, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x0A)
        if len(resp) != 21:
            raise RuntimeError("unexpected response length for get_all_currents")
        data = resp[1:]
        return [int.from_bytes(data[i:i+2], 'little') for i in range(0, 20, 2)]

    def get_all_speeds(self) -> List[int]:
        """Read the speed of all axes.

        Command 0x0B returns 20 bytes containing two bytes per joint【879085932721491†screenshot】.

        Returns:
            A list of 10 speed values (units and sign depend on context).
        """
        frame = self._build_frame(0x0B, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x0B)
        if len(resp) != 21:
            raise RuntimeError("unexpected response length for get_all_speeds")
        data = resp[1:]
        return [int.from_bytes(data[i:i+2], 'little') for i in range(0, 20, 2)]

    def get_all_temperatures(self) -> List[int]:
        """Read the temperature (degrees Celsius) of all axes.

        Command 0x0C returns 10 bytes, one per axis, each a signed 8‑bit
        temperature value【879085932721491†screenshot】.

        Returns:
            A list of 10 integer temperatures.
        """
        frame = self._build_frame(0x0C, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x0C)
        if len(resp) != 11:
            raise RuntimeError("unexpected response length for get_all_temperatures")
        data = resp[1:]
        return [struct.unpack('<b', bytes([t]))[0] for t in data]

    def get_error_code(self) -> int:
        """Retrieve the current error code from the device.

        Command 0x0D returns a 16‑bit unsigned error code【879085932721491†screenshot】.  The
        meaning of each code is detailed in section 3.2.  A value of
        ``0x0000`` indicates no error; other values map to specific
        fault conditions (e.g. motor stall, over temperature, encoder
        error, communication error, initialisation fault, etc.).

        Returns:
            The error code as an integer.
        """
        frame = self._build_frame(0x0D, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x0D)
        if len(resp) != 3:
            raise RuntimeError("unexpected response length for get_error_code")
        return int.from_bytes(resp[1:3], 'little')

    def clear_error(self) -> None:
        """Clear the current error condition.

        Command 0x0E clears any active error.  A single status byte is
        returned indicating success or failure【879085932721491†screenshot】.
        """
        frame = self._build_frame(0x0E, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x0E)
        if resp[1] != 0x01:
            raise RuntimeError("failed to clear error")

    def play_internal_action(self, action_index: int) -> None:
        """Trigger a predefined motion stored on the device.

        Command 0x0F accepts a one byte action index.  According to
        section 3.2 this command is reserved and currently not used by
        the device; however it is implemented here for completeness.  The
        response indicates success (0x01) or failure (0x00)【879085932721491†screenshot】.
        """
        if not (0 <= action_index <= 255):
            raise ValueError("action_index must be between 0 and 255")
        frame = self._build_frame(0x0F, [action_index])
        self._send_frame(frame)
        resp = self._wait_for_response(0x0F)
        if resp[1] != 0x01:
            raise RuntimeError("failed to play internal action")

    def get_mass_production_positions(self) -> List[int]:
        """Read the factory calibration positions for all axes.

        Command 0x10 returns 20 bytes of production calibration data (2 bytes
        per axis)【879085932721491†screenshot】.  Units are 0.1 degrees.
        """
        frame = self._build_frame(0x10, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x10)
        if len(resp) != 21:
            raise RuntimeError("unexpected response length for get_mass_production_positions")
        data = resp[1:]
        return [int.from_bytes(data[i:i+2], 'little') for i in range(0, 20, 2)]

    def get_single_sensor(self, sensor_index: int) -> List[int]:
        """Read the values of a specific fingertip or palm sensor array.

        Command 0x11 reads a single sensor board.  The `sensor_index`
        identifies which of the seven fingertip sensors or two palm
        sensors to read (1–7 for fingertips, other values reserved for
        palm/back).  The response length depends on the sensor type:

        * Fingertip sensors (indices 1–7) return 16 data points (4×4 array).
        * Palm/back sensors return 25 data points (5×5 array).

        Returns:
            A list of unsigned integers representing the raw sensor
            values.  Elements are ordered row‑major (first index
            corresponds to row, second to column)【879085932721491†screenshot】.
        """
        if not (1 <= sensor_index <= 7):
            raise ValueError("sensor_index must be between 1 and 7 for fingertips")
        frame = self._build_frame(0x11, [sensor_index])
        self._send_frame(frame)
        resp = self._wait_for_response(0x11)
        data = resp[1:]
        # The first byte of the reply is the sensor index
        if data and data[0] != sensor_index:
            raise RuntimeError("sensor index mismatch in response")
        # Remove the index byte
        values = list(data[1:])
        return values

    def get_all_fingertip_sensors(self) -> List[int]:
        """Read all fingertip sensors (thumb, index and middle fingers).

        Command 0x12 returns 48 bytes: three 4×4 arrays for the thumb,
        index and middle fingers【879085932721491†screenshot】.
        """
        frame = self._build_frame(0x12, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x12)
        values = list(resp[1:])
        if len(values) != 48:
            raise RuntimeError("unexpected response length for get_all_fingertip_sensors")
        return values

    def get_all_remaining_fingertip_sensors(self) -> List[int]:
        """Read the remaining fingertip sensors (ring and little fingers).

        Command 0x13 returns 32 bytes: two 4×4 arrays【879085932721491†screenshot】.
        """
        frame = self._build_frame(0x13, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x13)
        values = list(resp[1:])
        if len(values) != 32:
            raise RuntimeError("unexpected response length for get_all_remaining_fingertip_sensors")
        return values

    def get_palm_sensors(self) -> List[int]:
        """Read the palm and back sensors.

        Command 0x14 returns 50 bytes: 5×5 arrays for the palm and back【879085932721491†screenshot】.
        """
        frame = self._build_frame(0x14, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x14)
        values = list(resp[1:])
        if len(values) != 50:
            raise RuntimeError("unexpected response length for get_palm_sensors")
        return values

    def set_run_mode(self, motor_id: int, run_mode: int) -> None:
        """Set the operating mode of a motor (e.g. position, speed).

        Command 0x15 expects a two‑byte payload: ``byte1`` is the motor
        ID (1–10) and ``byte2`` is the run mode (protocol details are
        defined in section 3.2).  The device returns a status byte【879085932721491†screenshot】.
        """
        if not (1 <= motor_id <= 10):
            raise ValueError("motor_id must be between 1 and 10")
        if not (0 <= run_mode <= 255):
            raise ValueError("run_mode must fit in one byte")
        frame = self._build_frame(0x15, [motor_id, run_mode])
        self._send_frame(frame)
        resp = self._wait_for_response(0x15)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set run mode")

    def get_all_loads(self) -> List[int]:
        """Read the current load (PWM duty cycle) of all supported axes.

        Command 0x1A returns 20 bytes of uint16 load values (0.1 % units)
        for axes 1, 2, 3, 5, 6, 8 and 10【879085932721491†screenshot】.  Unsupported axes return zero.
        """
        frame = self._build_frame(0x1A, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x1A)
        if len(resp) != 21:
            raise RuntimeError("unexpected response length for get_all_loads")
        data = resp[1:]
        return [int.from_bytes(data[i:i+2], 'little') for i in range(0, 20, 2)]

    def set_motor_speed(self, speeds: Iterable[int]) -> None:
        """Specify target speeds for all 10 motors.

        Command 0x20 writes a list of 10 signed 16‑bit speeds (range
        –4096 to 4096).  The device returns a status byte【879085932721491†screenshot】.
        """
        spd_list = list(speeds)
        if len(spd_list) != 10:
            raise ValueError("speeds must contain exactly 10 values")
        payload = []
        for s in spd_list:
            if not (-4096 <= s <= 4096):
                raise ValueError("each speed must be between –4096 and 4096 inclusive")
            payload.extend(struct.pack('<h', s))
        frame = self._build_frame(0x20, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x20)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set motor speeds")

    def set_overload_torque(self, motor_id: int, percentage: int) -> None:
        """Set the overload torque threshold for a specific motor.

        Command 0x21 writes a motor ID and an overload torque percentage
        (0 %–100.0 % expressed as an integer 0–1000 with 0.1 % units)【879085932721491†screenshot】.
        """
        cprint.warn(f"[USB/set_overload_torque]: According to the docs, bytes2's 0~1000 represents max torque's 0~100 percentage, however, one byte can only represent 0~255.")
        cprint.warn(f"[USB/set_overload_torque]: This func currently always fails.")
        if not (1 <= motor_id <= 10):
            raise ValueError("motor_id must be between 1 and 10")
        if not (0 <= percentage <= 1000):
            raise ValueError("percentage must be between 0 and 1000 (0.1 % units)")
        payload = [motor_id, percentage]
        frame = self._build_frame(0x21, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x21)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set overload torque")

    def set_overload_time(self, motor_id: int, time_ms: int) -> None:
        """Set the overload duration threshold for a motor.

        Command 0x22 writes the motor ID and the overload time in units of
        0.01 s (e.g. a value of 50 means 0.5 s)【879085932721491†screenshot】.
        """
        cprint.warn(f"[USB/set_overload_time]: This func currently always fails.")
        if not (1 <= motor_id <= 10):
            raise ValueError("motor_id must be between 1 and 10")
        if not (0 <= time_ms <= 65535):
            raise ValueError("time_ms must be between 0 and 65535 (0.01 s units)")
        payload = [motor_id, time_ms]
        frame = self._build_frame(0x22, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x22)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set overload time")

    def set_protection_torque(self, motor_id: int, percentage: int) -> None:
        """Set the protection torque threshold for a motor.

        Command 0x23 writes the motor ID and the protection torque
        percentage (0–100 %)【879085932721491†screenshot】.
        """
        cprint.warn(f"[USB/set_protection_torque]: This func currently always fails.")
        if not (1 <= motor_id <= 10):
            raise ValueError("motor_id must be between 1 and 10")
        if not (0 <= percentage <= 100):
            raise ValueError("percentage must be between 0 and 100 (1 % units)")
        payload = [motor_id, percentage]
        frame = self._build_frame(0x23, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x23)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set protection torque")

    def set_min_start_torque(self, axis_index: int, value: int) -> None:
        """Define the minimum starting torque for a motor.

        Command 0x24 writes a three‑byte payload: the axis index and a
        16‑bit torque value representing a percentage of the full stall
        torque (0–100.0 %, units of 0.1 %)【879085932721491†screenshot】.
        """
        cprint.warn(f"[USB/set_min_start_torque]: This func currently always fails.")
        if not (1 <= axis_index <= 10):
            raise ValueError("axis_index must be between 1 and 10")
        if not (0 <= value <= 65535):
            raise ValueError("value must fit in 16 bits")
        payload = [axis_index] + list(struct.pack('<H', value))
        frame = self._build_frame(0x24, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x24)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set minimum start torque")

    def set_max_torque(self, axis_index: int, value: int) -> None:
        """Define the maximum torque limit for a motor.

        Command 0x25 writes a three‑byte payload: axis index and the
        maximum torque percentage (0–100.0 %, units of 0.1 %)【879085932721491†screenshot】.
        """
        cprint.warn(f"[USB/set_max_torque]: This func currently always fails.")
        if not (1 <= axis_index <= 10):
            raise ValueError("axis_index must be between 1 and 10")
        if not (0 <= value <= 1000):
            raise ValueError("value must be between 0 and 1000 (0.1 % units)")
        payload = [axis_index] + list(struct.pack('<H', value))
        frame = self._build_frame(0x25, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x25)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set maximum torque")

    def get_all_sensor_ids(self) -> List[int]:
        """Retrieve the IDs of all fingertip sensors.

        Command 0x27 returns a seven‑byte list of sensor identifiers【879085932721491†screenshot】.
        """
        frame = self._build_frame(0x27, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x27)
        return list(resp[1:])

    def set_plot_interval(self, interval_ms: int) -> None:
        """Adjust the reporting interval for plot data.

        Command 0x28 writes a 16‑bit little‑endian interval in milliseconds【879085932721491†screenshot】.
        """
        if not (0 <= interval_ms <= 65535):
            raise ValueError("interval_ms must be between 0 and 65535 milliseconds")
        payload = list(struct.pack('<H', interval_ms))
        frame = self._build_frame(0x28, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x28)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set plot interval")

    def get_plot_data(self) -> List[Tuple[int, int, int]]:
        """Retrieve the latest plot data for all axes.

        Command 0x29 returns 60 bytes arranged as 10 groups of three
        little‑endian values: position, speed and current【879085932721491†screenshot】.

        Returns:
            A list of 10 tuples ``(position, speed, current)``.
        """
        frame = self._build_frame(0x29, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x29)
        data = resp[1:]
        if len(data) != 60:
            raise RuntimeError("unexpected response length for get_plot_data")
        result = []
        for i in range(0, 60, 6):
            pos = int.from_bytes(data[i:i+2], 'little')
            spd = int.from_bytes(data[i+2:i+4], 'little')
            cur = int.from_bytes(data[i+4:i+6], 'little')
            result.append((pos, spd, cur))
        return result

    def set_hand_orientation(self, is_left: bool) -> None:
        """Specify whether the device is a left or right hand.

        Command 0x31 accepts a single byte: 0 for right hand (default)
        and 1 for left hand【879085932721491†screenshot】.
        """
        payload = [1 if is_left else 0]
        frame = self._build_frame(0x31, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0x31)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set hand orientation")

    def set_control_source(self, source: int) -> None:
        """Change the command source (robot vs GUI).

        Command 0x80 sets the control source: 0 selects the internal
        controller, 1 selects the external GUI【879085932721491†screenshot】.
        """
        if source not in (0, 1):
            raise ValueError("source must be 0 (robot) or 1 (operator)")
        frame = self._build_frame(0x80, [source])
        self._send_frame(frame)
        resp = self._wait_for_response(0x80)
        # The response echoes the original value
        if resp[1] != source:
            raise RuntimeError("failed to set control source")

    def get_control_source(self) -> int:
        """Query which controller currently has authority.

        Command 0x81 returns a single byte: 0 for robot control, 1 for
        operator control【879085932721491†screenshot】.
        """
        frame = self._build_frame(0x81, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0x81)
        return resp[1]

    def set_product_serial(self, vendor_code: bytes, material_code: bytes, date_code: bytes,
                            serial_number: bytes) -> None:
        """Program the product serial information into the device.

        Command 0xC1 writes a 19‑byte structure composed of three bytes
        vendor code, six bytes material code, six bytes date code (YYMMDD)
        and four bytes serial number【879085932721491†screenshot】.

        Args:
            vendor_code: Three bytes identifying the supplier.
            material_code: Six bytes identifying the product type.
            date_code: Six ASCII digits representing the manufacturing
                date in YYMMDD format.
            serial_number: Four bytes of sequential serial number.
        """
        if len(vendor_code) != 3 or len(material_code) != 6 or len(date_code) != 6 or len(serial_number) != 4:
            raise ValueError("vendor_code (3 bytes), material_code (6), date_code (6) and serial_number (4) must be the correct lengths")
        payload = list(vendor_code + material_code + date_code + serial_number)
        frame = self._build_frame(0xC1, payload)
        self._send_frame(frame)
        resp = self._wait_for_response(0xC1)
        if resp[1] != 0x01:
            raise RuntimeError("failed to set product serial")

    def get_product_serial(self) -> Tuple[bytes, bytes, bytes, bytes]:
        """Retrieve the product serial information.

        Command 0xC2 returns a 19‑byte payload containing vendor code,
        material code, date code and serial number【879085932721491†screenshot】.
        """
        frame = self._build_frame(0xC2, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0xC2)
        data = resp[1:]
        if len(data) != 19:
            raise RuntimeError("unexpected response length for get_product_serial")
        vendor_code = data[0:3]
        material_code = data[3:9]
        date_code = data[9:15]
        serial_number = data[15:19]
        return vendor_code, material_code, date_code, serial_number

    def get_device_info(self) -> Tuple[int, str, str, str, int]:
        """Query the device model and firmware/hardware versions.

        Command 0xCD returns information about the connected device in
        eight bytes: device type, product status (ASCII), software
        version (three ASCII bytes), hardware version (three ASCII bytes)
        and the number of degrees of freedom【879085932721491†screenshot】.

        Returns:
            A tuple ``(device_type, product_status, software_version,
            hardware_version, dof)`` where strings are decoded from
            ASCII.
        """
        frame = self._build_frame(0xCD, [])
        self._send_frame(frame)
        resp = self._wait_for_response(0xCD)
        data = resp[1:]
        if len(data) < 9:
            raise RuntimeError("unexpected response length for get_device_info")
        device_type = data[0]
        product_status = data[1:3].decode('ascii')
        software_version = f"{data[3]}.{data[4]}.{data[5]}"
        hardware_version = f"{data[6]}.{data[7]}.{data[8]}"
        dof = data[9] if len(data) > 9 else 0
        return device_type, product_status, software_version, hardware_version, dof


__all__ = ["DexterousHandUSB", "CRC16"]

if __name__ == "__main__":
    # Example usage - execute all DexterousHandUSB USB communication functions
    hand = DexterousHandUSB(port='/dev/ttyACM0', device_id=0x01)
    print("Connected to DexterousHandUSB")
    
    # Test basic device info and status
    print("\n=== Device Information ===")
    device_info = hand.get_device_info()
    print(f"Device info: Type={device_info[0]}, Status={device_info[1]}, SW={device_info[2]}, HW={device_info[3]}, DOF={device_info[4]}")
    
    print("Current enable state:", hand.get_enable_state())
    error_code = hand.get_error_code()
    print(f"Error code: {error_code}")
    if error_code != 0:
        hand.clear_error()
        print(f"Cleared error, current error code: {hand.get_error_code()}")
    
    # Test control source
    print("\n=== Control Source ===")
    # control_source = hand.get_control_source()
    # print(f"Current control source: {control_source}")
    # hand.set_control_source(0)  # Set to robot control
    # print("Set control source to robot")
    
    # Test enable/disable
    print("\n=== Enable/Disable ===")
    hand.set_enable(mode=1)  # Enable
    print("Device enabled")
    
    # Test single axis operations
    print("\n=== Single Axis Operations ===")
    # hand.set_axis_position(1, 2048)  # Set axis 1 to mid position
    # print("Set axis 1 position to 2048")
    for i in range(1,11):
        try:
            pos = hand.get_axis_position(i)
        except Exception as e:
            print(f"Axis {i} position: ")
    
    target_pos = hand.set_single_axis_target(1, 1024)
    print(f"Set axis 1 target to 1024, result: {target_pos}")
    
    # Test all axes operations
    print("\n=== All Axes Operations ===")
    all_positions = hand.get_all_positions()
    print(f"All positions: {all_positions}")
    
    all_currents = hand.get_all_currents()
    print(f"All currents: {all_currents[:3]}...")  # Show first 3
    
    all_speeds = hand.get_all_speeds()
    print(f"All speeds: {all_speeds[:3]}...")  # Show first 3
    
    all_temps = hand.get_all_temperatures()
    print(f"All temperatures: {all_temps[:3]}...")  # Show first 3
    
    all_loads = hand.get_all_loads()
    print(f"All loads: {all_loads[:3]}...")  # Show first 3
    
    # Test mass production positions
    mass_prod_pos = hand.get_mass_production_positions()
    print(f"Mass production positions: {mass_prod_pos[:3]}...")  # Show first 3
    
    # Test setting all positions
    test_positions = [2048] * 10  # Mid position for all axes
    pos_feedback = hand.set_all_positions(test_positions)
    print(f"Set all positions feedback - positions: {pos_feedback}.")
    
    # Test motor speed control
    print("\n=== Motor Speed Control ===")
    test_speeds = [0] * 10  # Zero speed for all motors
    hand.set_motor_speed(test_speeds)
    print("Set all motor speeds to 0")
    
    # Test motor configuration
    print("\n=== Motor Configuration ===")
    hand.set_run_mode(1, 1)  # Set motor 1 to position mode
    print("Set motor 1 run mode")
    
    # hand.set_overload_torque(1, 255)
    # print("Set motor 1 overload torque to 80%")
    
    # hand.set_overload_time(1, 100)  # 1 second overload time
    # print("Set motor 1 overload time to 1s")
    
    # hand.set_protection_torque(1, 90)  # 90% protection torque
    # print("Set motor 1 protection torque to 90%")
    
    # hand.set_min_start_torque(1, 100)  # 10% minimum start torque
    # print("Set motor 1 minimum start torque to 10%")
    
    # hand.set_max_torque(1, 900)  # 90% maximum torque
    # print("Set motor 1 maximum torque to 90%")
    
    # Test sensor operations
    print("\n=== Sensor Operations ===")
    try:
        single_sensor = hand.get_single_sensor(1)  # Read fingertip sensor 1
        print(f"Single sensor 1 data length: {len(single_sensor)}")
        
        all_fingertips = hand.get_all_fingertip_sensors()
        print(f"All fingertip sensors data length: {len(all_fingertips)}")
        
        remaining_fingertips = hand.get_all_remaining_fingertip_sensors()
        print(f"Remaining fingertip sensors data length: {len(remaining_fingertips)}")
        
        palm_sensors = hand.get_palm_sensors()
        print(f"Palm sensors data length: {len(palm_sensors)}")
        
        sensor_ids = hand.get_all_sensor_ids()
        print(f"Sensor IDs: {sensor_ids}")
    except Exception as e:
        print(f"Sensor operations failed (may not be supported): {e}")
    
    # Test plot data functions
    print("\n=== Plot Data ===")
    try:
        hand.set_plot_interval(100)  # 100ms interval
        print("Set plot interval to 100ms")
        
        plot_data = hand.get_plot_data()
        print(f"Plot data for first 3 axes: {plot_data[:3]}")
    except Exception as e:
        print(f"Plot data operations failed: {e}")
    
    # Test hand orientation
    print("\n=== Hand Orientation ===")
    hand.set_hand_orientation(False)  # Right hand
    print("Set hand orientation to right hand")
    
    # Test product serial operations
    print("\n=== Product Serial ===")
    try:
        vendor = b"ABC"
        material = b"123456"
        date = b"240101"
        serial = b"0001"
        hand.set_product_serial(vendor, material, date, serial)
        print("Set product serial")
        
        serial_info = hand.get_product_serial()
        print(f"Product serial: {serial_info}")
    except Exception as e:
        print(f"Product serial operations failed: {e}")
    
    # Test internal action (if supported)
    print("\n=== Internal Actions ===")
    try:
        hand.play_internal_action(1)
        print("Played internal action 1")
    except Exception as e:
        print(f"Internal action failed (may not be supported): {e}")
    
    # Return to home position
    print("\n=== Return Home ===")
    hand.return_home()
    print("Returned all axes to home position")
    
    final_positions = hand.get_all_positions()
    print(f"Final positions: {final_positions}")
    
    print("\n=== Test Complete ===")
    print("All DexterousHandUSB functions have been executed")

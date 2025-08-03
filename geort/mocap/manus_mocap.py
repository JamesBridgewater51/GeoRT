# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Python ZMQ Client (manus_mocap.py)

    Purpose: To provide a clean, simple API for the main application to receive the processed data.

    Class: ManusMocap

    Functionality:

        It connects to the ZMQ socket opened by manus_mocap_core.py.

        It runs a background thread that continuously listens for new data, decodes it, and stores the latest frame.

        The get() method provides this latest frame to the caller in the standard dictionary format.

"""
import time
import zmq
import numpy as np
import threading 
from geort import save_human_data
import math


class ManusMocap:
    '''
    Applies to any ZMQ-broadcasted mocap data with fixed shape (21,3) and dtype float32.
    Runs a background thread to continuously receive and update latest data.
    '''
    def __init__(self, port=8765):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://localhost:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        self.socket = socket

        self._latest_data = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                msg = self.socket.recv(flags=zmq.NOBLOCK)
                arr = np.frombuffer(msg, dtype=np.float32).reshape(21, 3)
                with self._lock:
                    self._latest_data = arr
            except zmq.Again:
                import time
                time.sleep(0.001)

    def get(self):
        with self._lock:
            if self._latest_data is not None:
                return {"result": self._latest_data.copy(), "status": "recording"}
            else:
                return {"result": None, "status": "no data"}

    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close()


if __name__ == "__main__":
    mocap = ManusMocap()
    data = []
    try:
        for step in range(8000):
            result = mocap.get()
            if result["result"] is not None:
                print(result['result'].shape, result['status'])
                data.append(result["result"])
            else:
                print("No data received")
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\nStopping data collection...")
    finally:
        mocap.close()
        if data:
            import os
            from pathlib import Path
            data_output_path = Path("data/Aug_5_2025/manus_mocap_test_long_fps_10.npy")
            data_output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(data_output_path, np.array(data))
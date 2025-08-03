#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
libcanbus 全量测试脚本（Linux 版，ctypes）
=================================================
用途
----
- 加载 /usr/local/lib/libusb-1.0.so 和 /usr/local/lib/libcanbus.so
- 定义所有 C 结构体/常量/函数原型（根据厂商文档）
- 覆盖测试 CAN 与 CANFD 的初始化、收发、过滤器、时间戳、设备信息等接口
- 回显设备健康度（错误计数/总线状态）与库健壮性

要求
----
1) Linux, Python 3.8+
2) 安装/放置库:
   - /usr/local/lib/libusb-1.0.so
   - /usr/local/lib/libcanbus.so
   并确保: export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
3) 设备权限:
   - root 用户可直接访问
   - 非 root 需放置 udev 规则(如 hcanbus.rules 到 /etc/udev/rules.d/)，或手动赋权后重启

重要差异（Linux vs Windows）
----------------------------
- 厂商说明：Linux 下双通道 USB-CAN 可能“只扫描出一个设备，但包含 0/1 两个通道”。
- 一些函数在 Linux 可能“多一个 channel 参数”。本脚本提供两种模式：
  MODE_A: 仅 devNum（与 Windows 文档一致）
  MODE_B: devNum + channel（Linux 某些固件/版本使用）
- 通过脚本顶部开关 LINUX_EXTRA_CHANNEL 选择，默认 False（MODE_A）。
  若调用失败，可切换为 True 再试。

运行
----
$ sudo -E python3 libcan_full_tester.py
或
$ CAN_DEV=0 CAN_CH=0 python3 libcan_full_tester.py

"""  # noqa: E501

import ctypes as C
import ctypes.util
import os
import sys
import time
import threading
from typing import Optional, Tuple, List

# -------------------------- 用户可调开关 --------------------------
# 对部分函数是否追加 "channel" 参数（Linux 某些版本需要）：
LINUX_EXTRA_CHANNEL = True  # 如果调用失败，改为 True 再试
DEFAULT_DEV = int(os.getenv("CAN_DEV", "0"))
DEFAULT_CH = int(os.getenv("CAN_CH", "0"))  # 仅在 LINUX_EXTRA_CHANNEL=True 时使用
TEST_SECONDS = 5  # 接收线程运行时长
RX_BATCH = 1000   # 每次最大接收帧数（见文档上限）
TX_TIMEOUT_MS = 200
RX_TIMEOUT_MS = 200
RT_MICRO_SPACING = 1000  # 定时发送：相邻帧间隔(微秒)

# -------------------------- 动态库加载 --------------------------
def load_libusb():
    # 优先 /usr/local/lib
    candidates = [
        "/usr/local/lib/libusb-1.0.so",
        ctypes.util.find_library("usb-1.0"),
        "libusb-1.0.so",
    ]
    last_err = None
    for path in candidates:
        if not path:
            continue
        try:
            return C.CDLL(path, mode=os.RTLD_GLOBAL)
        except OSError as e:
            last_err = e
    raise OSError(f"未能加载 libusb-1.0.so: {last_err}")

def load_libcan():
    candidates = [
        "/usr/local/lib/libcanbus.so",
        ctypes.util.find_library("canbus"),
        "libcanbus.so",
    ]
    last_err = None
    for path in candidates:
        if not path:
            continue
        try:
            return C.CDLL(path, mode=os.RTLD_GLOBAL)
        except OSError as e:
            last_err = e
    raise OSError(f"未能加载 libcanbus.so: {last_err}")


# 提前加载 libusb 以确保符号可见
try:
    _libusb = load_libusb()
except Exception as e:
    print(f"[WARN] 加载 libusb 失败：{e}", file=sys.stderr)

libcan = load_libcan()

# -------------------------- C 类型与结构体 --------------------------
c_uint = C.c_uint
c_ushort = C.c_ushort
c_ubyte = C.c_ubyte
c_int = C.c_int
c_char = C.c_char
c_char_p = C.c_char_p
c_void_p = C.c_void_p

class Dev_Info(C.Structure):
    _fields_ = [
        ("HW_Type", c_char * 32),
        ("HW_Ser",  c_char * 32),
        ("HW_Ver",  c_char * 32),
        ("FW_Ver",  c_char * 32),
        ("MF_Date", c_char * 32),
    ]

class Can_Config(C.Structure):
    _fields_ = [
        ("Baudrate",  c_uint),
        ("Pres",      c_ushort),
        ("Tseg1",     c_ubyte),
        ("Tseg2",     c_ubyte),
        ("SJW",       c_ubyte),
        ("Config",    c_ubyte),
        ("Model",     c_ubyte),
        ("Reserved",  c_ubyte),
    ]

class CanFD_Config(C.Structure):
    _fields_ = [
        ("NomBaud",   c_uint),
        ("DatBaud",   c_uint),
        ("NomPre",    c_ushort),
        ("NomTseg1",  c_ubyte),
        ("NomTseg2",  c_ubyte),
        ("NomSJW",    c_ubyte),
        ("DatPre",    c_ubyte),
        ("DatTseg1",  c_ubyte),
        ("DatTseg2",  c_ubyte),
        ("DatSJW",    c_ubyte),
        ("Config",    c_ubyte),
        ("Model",     c_ubyte),
        ("Cantype",   c_ubyte),
    ]

class Can_Msg(C.Structure):
    _fields_ = [
        ("ID",         c_uint),
        ("TimeStamp",  c_uint),
        ("FrameType",  c_ubyte),
        ("DataLen",    c_ubyte),
        ("ExternFlag", c_ubyte),
        ("RemoteFlag", c_ubyte),
        ("BusSatus",   c_ubyte),
        ("ErrSatus",   c_ubyte),
        ("TECounter",  c_ubyte),
        ("RECounter",  c_ubyte),
        ("Data",       c_ubyte * 8),
    ]

class CanFD_Msg(C.Structure):
    _fields_ = [
        ("ID",         c_uint),
        ("TimeStamp",  c_uint),
        ("FrameType",  c_ubyte),
        ("DLC",        c_ubyte),
        ("ExternFlag", c_ubyte),
        ("RemoteFlag", c_ubyte),
        ("BusSatus",   c_ubyte),
        ("ErrSatus",   c_ubyte),
        ("TECounter",  c_ubyte),
        ("RECounter",  c_ubyte),
        ("Data",       c_ubyte * 64),
    ]

class Can_Status(C.Structure):
    _fields_ = [
        ("BusSatus",   c_ubyte),
        ("ErrSatus",   c_ubyte),
        ("TECounter",  c_ubyte),
        ("RECounter",  c_ubyte),
        ("TimeStamp",  c_uint),
    ]

# -------------------------- 常量（按文档） --------------------------
# FrameType bits
FT_TX          = 0x01   # 接收标注为“发送帧回传”（echo）
FT_FAIL        = 0x02   # 传输失败
FT_FD          = 0x04   # CANFD
FT_FDBRS       = 0x08   # CANFD BRS（可变速率）
FT_RT_BEGIN    = 0x10   # 定时发送开始
FT_RT_END      = 0x20   # 定时发送结束
FT_ECHO        = 0x40   # 发送回环到接收队列

# BusSatus bits（组合）
BUS_ERR_WARN   = 0x01
BUS_ERR_PASS   = 0x02
BUS_OFFLINE    = 0x04
BUS_OK         = 0x08
BUS_TX_ERR     = 0x10  # 发送邮箱仲裁错误
BUS_TX_EMPTY   = 0x20
BUS_RX_OVF     = 0x40
BUS_RX_FULL    = 0x80

# ErrSatus (非组合枚举)
ERR_STUFF      = 0x01
ERR_FORM       = 0x02
ERR_ACK        = 0x03
ERR_BIT_RECESS = 0x04
ERR_BIT_DOM    = 0x05
ERR_CRC        = 0x06

# Model
MODEL_NORMAL   = 0
MODEL_LOOPBACK = 1
MODEL_SILENT   = 2
MODEL_SILENTLB = 3

# Cantype
CAN_OPEN_CLASSIC   = 0
CAN_OPEN_ISO_FD    = 1
CAN_OPEN_NONISO_FD = 2

# DLC -> Data length 映射（CAN FD）
DLC_TO_LEN = {
    0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8,
    9:12, 10:16, 11:20, 12:24, 13:32, 14:48, 15:64
}
LEN_TO_DLC = {v: k for k, v in DLC_TO_LEN.items()}

# -------------------------- 函数原型绑定 --------------------------
# 为适配两种签名模式，统一通过包装函数调度。
# 这里仅设置 restype，避免在签名不一致时崩溃；参数按 c_int/c_uint 传入。
def sym(name):
    try:
        return getattr(libcan, name)
    except AttributeError:
        return None

# 通用：设置返回值类型为 c_int
def set_ret(func):
    if func is not None:
        func.restype = c_int
    return func

Reg_HotPlug_Func     = set_ret(sym("Reg_HotPlug_Func"))
CAN_ScanDevice       = set_ret(sym("CAN_ScanDevice"))
CAN_OpenDevice       = set_ret(sym("CAN_OpenDevice"))
CAN_CloseDevice      = set_ret(sym("CAN_CloseDevice"))
CAN_GetDevType       = set_ret(sym("CAN_GetDevType"))
CAN_GetDevPlck       = set_ret(sym("CAN_GetDevPlck"))
CAN_ReadDevInfo      = set_ret(sym("CAN_ReadDevInfo"))
CAN_GetTimeStamp     = set_ret(sym("CAN_GetTimeStamp"))
CAN_SetTimeStamp     = set_ret(sym("CAN_SetTimeStamp"))
CAN_GetDevID         = set_ret(sym("CAN_GetDevID"))
CAN_SetDevID         = set_ret(sym("CAN_SetDevID"))
CAN_SetFilter        = set_ret(sym("CAN_SetFilter"))
CAN_Reset            = set_ret(sym("CAN_Reset"))
CAN_GetStatus        = set_ret(sym("CAN_GetStatus"))
CAN_Init             = set_ret(sym("CAN_Init"))
CAN_Transmit         = set_ret(sym("CAN_Transmit"))
CAN_TransmitRt       = set_ret(sym("CAN_TransmitRt"))
CAN_GetReceiveNum    = set_ret(sym("CAN_GetReceiveNum"))
CAN_Receive          = set_ret(sym("CAN_Receive"))
CANFD_Init           = set_ret(sym("CANFD_Init"))
CANFD_Transmit       = set_ret(sym("CANFD_Transmit"))
CANFD_TransmitRt     = set_ret(sym("CANFD_TransmitRt"))
CANFD_GetReceiveNum  = set_ret(sym("CANFD_GetReceiveNum"))
CANFD_Receive        = set_ret(sym("CANFD_Receive"))

# -------------------------- 工具函数 --------------------------
def _to_bytes(arr_like, length):
    """把 Python 列表/bytes 填入定长 ctypes 数组，超出截断，不足补零"""
    out = (c_ubyte * length)()
    if arr_like is None:
        return out
    b = bytes(arr_like)
    n = min(length, len(b))
    for i in range(n):
        out[i] = b[i]
    return out

def devinfo_to_dict(di: Dev_Info) -> dict:
    def f(x):
        return x.split(b'\x00', 1)[0].decode('utf-8', 'ignore')
    return {
        "HW_Type": f(bytes(di.HW_Type)),
        "HW_Ser":  f(bytes(di.HW_Ser)),
        "HW_Ver":  f(bytes(di.HW_Ver)),
        "FW_Ver":  f(bytes(di.FW_Ver)),
        "MF_Date": f(bytes(di.MF_Date)),
    }

def print_status(prefix: str, st: Can_Status):
    print(f"{prefix} Bus=0x{st.BusSatus:02x} Err=0x{st.ErrSatus:02x} "
          f"TE={st.TECounter} RE={st.RECounter} TS={st.TimeStamp}")

# -------------------------- 包装器：两种签名模式 --------------------------
def _open_device(dev: int, ch: int) -> int:
    if CAN_OpenDevice is None:
        raise RuntimeError("libcanbus 中未导出 CAN_OpenDevice")
    if not LINUX_EXTRA_CHANNEL:
        return CAN_OpenDevice(c_uint(dev))
    else:
        # 某些 Linux 版本：CAN_OpenDevice(dev, channel)
        return CAN_OpenDevice(c_uint(dev), c_uint(ch))

def _close_device(dev: int, ch: int) -> int:
    if CAN_CloseDevice is None:
        return 0
    if not LINUX_EXTRA_CHANNEL:
        return CAN_CloseDevice(c_uint(dev))
    else:
        return CAN_CloseDevice(c_uint(dev), c_uint(ch))

def _read_devinfo(dev: int, ch: int) -> Tuple[int, Dev_Info]:
    di = Dev_Info()
    if not LINUX_EXTRA_CHANNEL:
        ret = CAN_ReadDevInfo(c_uint(dev), C.byref(di))
    else:
        ret = CAN_ReadDevInfo(c_uint(dev), c_uint(ch), C.byref(di))
    return ret, di

def _get_devid(dev: int, ch: int) -> Tuple[int, int]:
    out = c_uint(0)
    if not LINUX_EXTRA_CHANNEL:
        ret = CAN_GetDevID(c_uint(dev), C.byref(out))
    else:
        ret = CAN_GetDevID(c_uint(dev), c_uint(ch), C.byref(out))
    return ret, out.value

def _set_devid(dev: int, ch: int, val: int) -> int:
    if not LINUX_EXTRA_CHANNEL:
        return CAN_SetDevID(c_uint(dev), c_uint(val))
    else:
        return CAN_SetDevID(c_uint(dev), c_uint(ch), c_uint(val))

def _get_plck(dev: int, ch: int) -> Tuple[int, int]:
    if CAN_GetDevPlck is None:
        return -1, 0
    if not LINUX_EXTRA_CHANNEL:
        ret = CAN_GetDevPlck(c_uint(dev))
    else:
        ret = CAN_GetDevPlck(c_uint(dev), c_uint(ch))
    return (0 if ret >= 0 else -1), int(ret)

def _get_devtype(dev: int, ch: int) -> Tuple[int, int]:
    if CAN_GetDevType is None:
        return -1, -1
    if not LINUX_EXTRA_CHANNEL:
        ret = CAN_GetDevType(c_uint(dev))
    else:
        ret = CAN_GetDevType(c_uint(dev), c_uint(ch))
    return (0 if ret >= 0 else -1), int(ret)

def _get_timestamp(dev: int, ch: int) -> Tuple[int, int]:
    ts = c_uint(0)
    if not LINUX_EXTRA_CHANNEL:
        ret = CAN_GetTimeStamp(c_uint(dev), C.byref(ts))
    else:
        ret = CAN_GetTimeStamp(c_uint(dev), c_uint(ch), C.byref(ts))
    return ret, ts.value

def _set_timestamp(dev: int, ch: int, ts: int, mode: int = 1) -> int:
    if not LINUX_EXTRA_CHANNEL:
        return CAN_SetTimeStamp(c_uint(dev), c_uint(ts), C.c_char(mode))
    else:
        return CAN_SetTimeStamp(c_uint(dev), c_uint(ch), c_uint(ts), C.c_char(mode))

def _get_status(dev: int, ch: int) -> Tuple[int, Can_Status]:
    st = Can_Status()
    if not LINUX_EXTRA_CHANNEL:
        ret = CAN_GetStatus(c_uint(dev), C.byref(st))
    else:
        ret = CAN_GetStatus(c_uint(dev), c_uint(ch), C.byref(st))
    return ret, st

def _reset(dev: int, ch: int) -> int:
    if not LINUX_EXTRA_CHANNEL:
        return CAN_Reset(c_uint(dev))
    else:
        return CAN_Reset(c_uint(dev), c_uint(ch))

def _set_filter(dev: int, ch: int, number: int, ftype: int, ftID: int, ftMask: int, enable: int) -> int:
    if not LINUX_EXTRA_CHANNEL:
        return CAN_SetFilter(c_uint(dev), C.c_char(number), C.c_char(ftype), c_uint(ftID), c_uint(ftMask), C.c_char(enable))
    else:
        return CAN_SetFilter(c_uint(dev), c_uint(ch), C.c_char(number), C.c_char(ftype), c_uint(ftID), c_uint(ftMask), C.c_char(enable))

def _can_init(dev: int, ch: int, cfg: Can_Config) -> int:
    if not LINUX_EXTRA_CHANNEL:
        return CAN_Init(c_uint(dev), C.byref(cfg))
    else:
        return CAN_Init(c_uint(dev), c_uint(ch), C.byref(cfg))

def _canfd_init(dev: int, ch: int, cfg: CanFD_Config) -> int:
    if CANFD_Init is None:
        return -1
    if not LINUX_EXTRA_CHANNEL:
        return CANFD_Init(c_uint(dev), C.byref(cfg))
    else:
        return CANFD_Init(c_uint(dev), c_uint(ch), C.byref(cfg))

def _can_transmit(dev: int, ch: int, msgs: List[Can_Msg], timeout_ms: int) -> int:
    arr = (Can_Msg * len(msgs))(*msgs)
    if not LINUX_EXTRA_CHANNEL:
        return CAN_Transmit(c_uint(dev), arr, c_uint(len(msgs)), c_int(timeout_ms))
    else:
        return CAN_Transmit(c_uint(dev), c_uint(ch), arr, c_uint(len(msgs)), c_int(timeout_ms))

def _can_transmit_rt(dev: int, ch: int, msgs: List[Can_Msg], timeout_ms: int) -> int:
    arr = (Can_Msg * len(msgs))(*msgs)
    txitems = c_uint(0)
    if not LINUX_EXTRA_CHANNEL:
        return CAN_TransmitRt(c_uint(dev), arr, c_uint(len(msgs)), C.byref(txitems), c_int(timeout_ms))
    else:
        return CAN_TransmitRt(c_uint(dev), c_uint(ch), arr, c_uint(len(msgs)), C.byref(txitems), c_int(timeout_ms))

def _can_receive(dev: int, ch: int, max_len: int, timeout_ms: int) -> Tuple[int, List[Can_Msg]]:
    arr = (Can_Msg * max_len)()
    if not LINUX_EXTRA_CHANNEL:
        n = CAN_Receive(c_uint(dev), arr, c_int(max_len), c_int(timeout_ms))
    else:
        n = CAN_Receive(c_uint(dev), c_uint(ch), arr, c_int(max_len), c_int(timeout_ms))
    if n < 0:
        return n, []
    return n, list(arr)[:n]

def _can_get_rxnum(dev: int, ch: int) -> int:
    if not LINUX_EXTRA_CHANNEL:
        return CAN_GetReceiveNum(c_uint(dev))
    else:
        return CAN_GetReceiveNum(c_uint(dev), c_uint(ch))

def _canfd_transmit(dev: int, ch: int, msgs: List[CanFD_Msg], timeout_ms: int) -> int:
    arr = (CanFD_Msg * len(msgs))(*msgs)
    if not LINUX_EXTRA_CHANNEL:
        return CANFD_Transmit(c_uint(dev), arr, c_uint(len(msgs)), c_int(timeout_ms))
    else:
        return CANFD_Transmit(c_uint(dev), c_uint(ch), arr, c_uint(len(msgs)), c_int(timeout_ms))

def _canfd_transmit_rt(dev: int, ch: int, msgs: List[CanFD_Msg], timeout_ms: int) -> int:
    arr = (CanFD_Msg * len(msgs))(*msgs)
    txitems = c_uint(0)
    if not LINUX_EXTRA_CHANNEL:
        return CANFD_TransmitRt(c_uint(dev), arr, c_uint(len(msgs)), C.byref(txitems), c_int(timeout_ms))
    else:
        return CANFD_TransmitRt(c_uint(dev), c_uint(ch), arr, c_uint(len(msgs)), C.byref(txitems), c_int(timeout_ms))

def _canfd_receive(dev: int, ch: int, max_len: int, timeout_ms: int) -> Tuple[int, List[CanFD_Msg]]:
    arr = (CanFD_Msg * max_len)()
    if not LINUX_EXTRA_CHANNEL:
        n = CANFD_Receive(c_uint(dev), arr, c_int(max_len), c_int(timeout_ms))
    else:
        n = CANFD_Receive(c_uint(dev), c_uint(ch), arr, c_int(max_len), c_int(timeout_ms))
    if n < 0:
        return n, []
    return n, list(arr)[:n]

def _canfd_get_rxnum(dev: int, ch: int) -> int:
    if not LINUX_EXTRA_CHANNEL:
        return CANFD_GetReceiveNum(c_uint(dev))
    else:
        return CANFD_GetReceiveNum(c_uint(dev), c_uint(ch))

# -------------------------- 热插拔回调（可选） --------------------------
_hotplug_event = threading.Event()
@C.CFUNCTYPE(None)
def _hotplug_cb():
    print("[HotPlug] 设备插拔事件")
    _hotplug_event.set()

def maybe_register_hotplug():
    if Reg_HotPlug_Func is None:
        return
    try:
        ret = Reg_HotPlug_Func(_hotplug_cb)
        if ret == 0:
            print("[OK] 注册热插拔回调成功")
        else:
            print(f"[WARN] 注册热插拔回调失败 ret={ret}")
    except Exception as e:
        print(f"[WARN] 注册热插拔回调异常: {e}")

# -------------------------- 构造测试数据 --------------------------
def make_can_msg(std_id=0x123, data=b"\x01\x02\x03\x04\x05\x06\x07\x08", ext=False, remote=False, timestamp_us=0, frame_type=0):
    m = Can_Msg()
    m.ID = c_uint(std_id)
    m.TimeStamp = c_uint(timestamp_us)
    m.FrameType = c_ubyte(frame_type)  # 普通发送: 0；定时发送需设置 FT_RT_BEGIN/FT_RT_END；FD 不应在这里设置
    m.DataLen = c_ubyte(min(len(data), 8))
    m.ExternFlag = c_ubyte(1 if ext else 0)
    m.RemoteFlag = c_ubyte(1 if remote else 0)
    m.BusSatus = c_ubyte(0)
    m.ErrSatus = c_ubyte(0)
    m.TECounter = c_ubyte(0)
    m.RECounter = c_ubyte(0)
    m.Data = _to_bytes(data, 8)
    return m

def make_canfd_msg(std_id=0x123, payload_len=64, ext=False, brs=True, timestamp_us=0):
    if payload_len not in LEN_TO_DLC:
        # 向上取整到可表达的长度
        cand = [l for l in sorted(LEN_TO_DLC) if l >= payload_len]
        payload_len = cand[0] if cand else 64
    dlc = LEN_TO_DLC[payload_len]
    m = CanFD_Msg()
    m.ID = c_uint(std_id)
    m.TimeStamp = c_uint(timestamp_us)
    m.FrameType = c_ubyte(FT_FD | (FT_FDBRS if brs else 0))
    m.DLC = c_ubyte(dlc)
    m.ExternFlag = c_ubyte(1 if ext else 0)
    m.RemoteFlag = c_ubyte(0)  # CANFD 无远程帧
    m.BusSatus = c_ubyte(0)
    m.ErrSatus = c_ubyte(0)
    m.TECounter = c_ubyte(0)
    m.RECounter = c_ubyte(0)
    # 填入递增数据
    buf = bytes((i % 256 for i in range(payload_len)))
    m.Data = _to_bytes(buf, 64)
    return m

# -------------------------- 接收线程 --------------------------
class RxThread(threading.Thread):
    def __init__(self, dev: int, ch: int, use_fd: bool = False):
        super().__init__(daemon=True)
        self.dev = dev
        self.ch = ch
        self.use_fd = use_fd
        self.rx_count = 0
        self.stop_event = threading.Event()

    def run(self):
        deadline = time.time() + TEST_SECONDS
        while not self.stop_event.is_set() and time.time() < deadline:
            if self.use_fd:
                n, frames = _canfd_receive(self.dev, self.ch, RX_BATCH, RX_TIMEOUT_MS)
            else:
                n, frames = _can_receive(self.dev, self.ch, RX_BATCH, RX_TIMEOUT_MS)
            if n < 0:
                print(f"[RX] 接收错误 n={n}")
                time.sleep(0.05)
                continue
            if n > 0:
                self.rx_count += n
                # 简要打印前 1 帧
                f0 = frames[0]
                if self.use_fd:
                    plen = DLC_TO_LEN.get(int(f0.DLC), 0)
                    print(f"[RX-FD] n={n} ID=0x{int(f0.ID):X} DLC={int(f0.DLC)} len={plen} TS={int(f0.TimeStamp)}")
                else:
                    print(f"[RX] n={n} ID=0x{int(f0.ID):X} len={int(f0.DataLen)} TS={int(f0.TimeStamp)}")
        print(f"[RX] 线程结束，总接收：{self.rx_count}")

    def stop(self):
        self.stop_event.set()

# -------------------------- 主测试流程 --------------------------
def test_can_stack(dev: int, ch: int):
    print(f"=== 开始测试: dev={dev}, ch={ch}, EXTRA_CH={LINUX_EXTRA_CHANNEL} ===")

    maybe_register_hotplug()

    # 扫描
    if CAN_ScanDevice is None:
        print("[ERR] 未导出 CAN_ScanDevice")
        return
    devs = CAN_ScanDevice()
    print(f"[OK] 扫描到设备数: {devs}")

    # 打开
    ret = _open_device(dev, ch)
    if ret != 0:
        print(f"[ERR] 打开设备失败 ret={ret}")
        return
    print("[OK] 打开设备成功")

    try:
        # 类型/频率/ID/信息
        r, tp = _get_devtype(dev, ch)
        print(f"[OK] 设备类型: {'CANFD' if tp==1 else 'CAN2.0'} (ret={r})")

        r, plck = _get_plck(dev, ch)
        if r == 0:
            print(f"[OK] 设备频率(Plck): {plck}")

        # r, di = _read_devinfo(dev, ch)
        # if r == 0:
        #     print(f"[OK] 设备信息: {devinfo_to_dict(di)}")

        r, devid = _get_devid(dev, ch)
        if r == 0:
            print(f"[OK] 通道ID: {devid} (仅区分通道用途)")

        # 时间戳
        _set_timestamp(dev, ch, 0, mode=1)
        r, ts = _get_timestamp(dev, ch)
        print(f"[OK] 当前时间戳: {ts} (ret={r})")

        # 复位（会清除过滤器）
        r = _reset(dev, ch)
        print(f"[OK] 复位 ret={r}，重新配置过滤器")

        # 过滤器：默认 0 号使能全通；这里显式设置 0 号 32bit 全开
        r = _set_filter(dev, ch, number=0, ftype=1, ftID=0, ftMask=0, enable=1)
        print(f"[OK] 设置过滤器 ret={r}")

        # ---------------- CAN 初始化 + 测试 ----------------
        can_cfg = Can_Config()
        can_cfg.Baudrate = c_uint(500000)         # 500k
        can_cfg.Pres = c_ushort(0)                # 0 让库自动计算
        can_cfg.Tseg1 = c_ubyte(0)
        can_cfg.Tseg2 = c_ubyte(0)
        can_cfg.SJW = c_ubyte(0)
        can_cfg.Config = c_ubyte(0x01 | 0x04)     # 内部终端电阻 + 自动重发
        can_cfg.Model  = c_ubyte(MODEL_NORMAL)
        can_cfg.Reserved = c_ubyte(0)

        r = _can_init(dev, ch, can_cfg)
        if r != 0:
            print(f"[ERR] CAN 初始化失败 ret={r}")
        else:
            print("[OK] CAN 初始化成功 -> 开始收发")
            rx_th = RxThread(dev, ch, use_fd=False)
            rx_th.start()

            # 普通发送 3 帧
            msgs = [
                make_can_msg(0x123, b"\x10\x20\x30\x40\x50\x60\x70\x80"),
                make_can_msg(0x124, b"\x01\x02\x03\x04"),
                make_can_msg(0x7FF, b"\xAA\xBB\xCC\xDD\xEE\xFF\x11\x22"),
            ]
            n = _can_transmit(dev, ch, msgs, TX_TIMEOUT_MS)
            print(f"[TX] CAN 发送帧数: {n}")

            # 定时发送 10 帧，每 1ms
            rt_msgs = []
            for i in range(10):
                ft = 0
                if i == 0: ft |= FT_RT_BEGIN
                if i == 9: ft |= FT_RT_END
                rt_msgs.append(make_can_msg(0x321, bytes([i])*8, timestamp_us=i*RT_MICRO_SPACING, frame_type=ft))
            n = _can_transmit_rt(dev, ch, rt_msgs, TX_TIMEOUT_MS)
            print(f"[TX-RT] CAN 定时发送帧数: {n}")

            # 缓冲区计数
            try:
                rn = _can_get_rxnum(dev, ch)
                print(f"[RXNUM] CAN 缓冲区待读: {rn}")
            except Exception as e:
                print(f"[WARN] 获取接收计数异常: {e}")

            # 状态
            r, st = _get_status(dev, ch)
            if r == 0:
                print_status("[STATUS-CAN]", st)

            time.sleep(TEST_SECONDS)
            rx_th.stop()
            rx_th.join()

        # ---------------- CAN FD 初始化 + 测试（若支持） ----------------
        if tp == 1 and CANFD_Init is not None:
            fd_cfg = CanFD_Config()
            fd_cfg.NomBaud = c_uint(1000000)      # 1M 常规
            fd_cfg.DatBaud = c_uint(5000000)      # 5M 数据
            fd_cfg.NomPre = c_ushort(0)           # 自动
            fd_cfg.NomTseg1 = c_ubyte(0)
            fd_cfg.NomTseg2 = c_ubyte(0)
            fd_cfg.NomSJW = c_ubyte(0)
            fd_cfg.DatPre = c_ubyte(0)
            fd_cfg.DatTseg1 = c_ubyte(0)
            fd_cfg.DatTseg2 = c_ubyte(0)
            fd_cfg.DatSJW = c_ubyte(0)
            fd_cfg.Config = c_ubyte(0x01 | 0x04)  # 终端电阻 + 自动重发
            fd_cfg.Model = c_ubyte(MODEL_NORMAL)
            fd_cfg.Cantype = c_ubyte(CAN_OPEN_ISO_FD)

            r = _canfd_init(dev, ch, fd_cfg)
            if r != 0:
                print(f"[ERR] CANFD 初始化失败 ret={r}")
            else:
                print("[OK] CANFD 初始化成功 -> 开始收发")
                rx_th2 = RxThread(dev, ch, use_fd=True)
                rx_th2.start()

                # 发送 2 帧 64B & 20B
                m1 = make_canfd_msg(0x0101, payload_len=64, brs=True, timestamp_us=0)
                m2 = make_canfd_msg(0x0101, payload_len=20, brs=True, timestamp_us=0)
                n = _canfd_transmit(dev, ch, [m1, m2], TX_TIMEOUT_MS)
                print(f"[TX-FD] 发送帧数: {n}")

                # 定时发送 10 帧 FD
                rt_fd_msgs = []
                for i in range(10):
                    m = make_canfd_msg(0x0, payload_len=32, brs=True, timestamp_us=i*RT_MICRO_SPACING)
                    # 开始/结束标识
                    if i == 0:
                        m.FrameType = c_ubyte(int(m.FrameType) | FT_RT_BEGIN)
                    if i == 9:
                        m.FrameType = c_ubyte(int(m.FrameType) | FT_RT_END)
                    rt_fd_msgs.append(m)
                n = _canfd_transmit_rt(dev, ch, rt_fd_msgs, TX_TIMEOUT_MS)
                print(f"[TX-RT-FD] 定时发送帧数: {n}")

                # 缓冲区计数
                try:
                    rn = _canfd_get_rxnum(dev, ch)
                    print(f"[RXNUM-FD] 缓冲区待读: {rn}")
                except Exception as e:
                    print(f"[WARN] 获取FD接收计数异常: {e}")

                # 状态
                r, st = _get_status(dev, ch)
                if r == 0:
                    print_status("[STATUS-FD]", st)

                time.sleep(TEST_SECONDS)
                rx_th2.stop()
                rx_th2.join()

        print("[OK] 测试流程结束")

    finally:
        _close_device(dev, ch)
        print("[OK] 设备已关闭")


if __name__ == "__main__":
    print(f"libcanbus tester | EXTRA_CH={LINUX_EXTRA_CHANNEL}  DEV={DEFAULT_DEV} CH={DEFAULT_CH}")
    test_can_stack(DEFAULT_DEV, DEFAULT_CH)
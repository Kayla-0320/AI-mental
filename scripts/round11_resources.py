"""Windows提交内存余量检查，避免将资源故障误记为媒体缺失。"""
from __future__ import annotations

import ctypes
import sys


def available_commit_bytes() -> int:
    """读取进程可用的提交余量，不修改系统或其他应用。"""
    if sys.platform!="win32":
        return 2**63-1
    class MemoryStatus(ctypes.Structure):
        _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
            ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
            ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
            ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
    status=MemoryStatus()
    status.dwLength=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("无法获取Windows内存余量")
    return int(status.ullAvailPageFile)


def require_memory(minimum_gib: float = 5.) -> None:
    """余量不足时暂停本工具，保护已有缓存和桌面应用。"""
    if available_commit_bytes()<minimum_gib*1024**3:
        raise MemoryError("提交内存余量不足，停止媒体任务；不得将解码失败缓存成观测缺失")

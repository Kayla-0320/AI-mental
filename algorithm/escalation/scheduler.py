"""
咨询师排班系统 —— 根据当前时间选择可用咨询师

⚠️ 声明：比赛 Demo 为模拟实现，使用硬编码排班表。
真实部署需要对接咨询师日历系统和实时状态接口。

排班规则：
    - 工作日（周一至周五）：09:00-18:00 有咨询师值班
    - 夜间 / 周末：仅保留紧急热线
    - 按优先级分配：优先分配给空闲的资深咨询师
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Counselor:
    """咨询师信息

    Attributes:
        counselor_id: 咨询师唯一标识
        name: 姓名
        specialty: 专长领域
        experience_years: 从业年限
        is_available: 当前是否可用
        priority: 优先级（数字越小优先级越高）
    """
    counselor_id: str
    name: str
    specialty: str
    experience_years: int
    is_available: bool = True
    priority: int = 5


@dataclass
class ScheduleEntry:
    """排班条目

    Attributes:
        counselor_id: 咨询师 ID
        day_of_week: 星期几（0=周一, 6=周日）
        start_hour: 开始时间（24小时制）
        end_hour: 结束时间（24小时制）
    """
    counselor_id: str
    day_of_week: int  # 0=Monday, 6=Sunday
    start_hour: int
    end_hour: int


# ============================================================
# 模拟咨询师数据
# ============================================================

MOCK_COUNSELORS = {
    "counselor_001": Counselor(
        counselor_id="counselor_001",
        name="张咨询师",
        specialty="青少年危机干预",
        experience_years=8,
        is_available=True,
        priority=1,
    ),
    "counselor_002": Counselor(
        counselor_id="counselor_002",
        name="李咨询师",
        specialty="认知行为治疗",
        experience_years=5,
        is_available=True,
        priority=2,
    ),
    "counselor_003": Counselor(
        counselor_id="counselor_003",
        name="王咨询师",
        specialty="家庭治疗",
        experience_years=3,
        is_available=True,
        priority=3,
    ),
}

# 模拟排班表
MOCK_SCHEDULE = [
    # 工作日白天：张咨询师（优先级最高）
    ScheduleEntry("counselor_001", day_of_week=0, start_hour=9, end_hour=18),
    ScheduleEntry("counselor_001", day_of_week=1, start_hour=9, end_hour=18),
    ScheduleEntry("counselor_001", day_of_week=2, start_hour=9, end_hour=18),
    ScheduleEntry("counselor_001", day_of_week=3, start_hour=9, end_hour=18),
    ScheduleEntry("counselor_001", day_of_week=4, start_hour=9, end_hour=18),
    # 工作日白天：李咨询师（备选）
    ScheduleEntry("counselor_002", day_of_week=0, start_hour=9, end_hour=18),
    ScheduleEntry("counselor_002", day_of_week=1, start_hour=9, end_hour=18),
    ScheduleEntry("counselor_002", day_of_week=2, start_hour=9, end_hour=18),
    ScheduleEntry("counselor_002", day_of_week=3, start_hour=9, end_hour=18),
    ScheduleEntry("counselor_002", day_of_week=4, start_hour=9, end_hour=18),
    # 周末：王咨询师
    ScheduleEntry("counselor_003", day_of_week=5, start_hour=10, end_hour=16),
    ScheduleEntry("counselor_003", day_of_week=6, start_hour=10, end_hour=16),
]


# ============================================================
# 排班系统
# ============================================================

class CounselorScheduler:
    """咨询师排班系统

    ⚠️ 比赛 Demo 为模拟实现，使用硬编码排班表。
    真实部署需要对接咨询师日历系统。
    """

    def __init__(
        self,
        counselors: Optional[dict[str, Counselor]] = None,
        schedule: Optional[list[ScheduleEntry]] = None,
    ):
        """初始化排班系统

        Args:
            counselors: 咨询师字典 {id: Counselor}
            schedule: 排班表
        """
        self._counselors = counselors or MOCK_COUNSELORS
        self._schedule = schedule or MOCK_SCHEDULE

    def get_available_counselor(
        self,
        current_time: Optional[datetime] = None,
    ) -> Optional[Counselor]:
        """根据当前时间获取可用咨询师

        按优先级选择当前时段值班的咨询师。

        Args:
            current_time: 当前时间（None 则使用系统时间）

        Returns:
            Optional[Counselor]: 可用咨询师，无可用时返回 None
        """
        if current_time is None:
            current_time = datetime.now()

        day_of_week = current_time.weekday()  # 0=Monday
        current_hour = current_time.hour

        # 查找当前时段值班的咨询师
        on_duty = []
        for entry in self._schedule:
            if (entry.day_of_week == day_of_week and
                    entry.start_hour <= current_hour < entry.end_hour):
                counselor = self._counselors.get(entry.counselor_id)
                if counselor and counselor.is_available:
                    on_duty.append(counselor)

        if not on_duty:
            return None

        # 按优先级排序，返回最高优先级的
        on_duty.sort(key=lambda c: c.priority)
        return on_duty[0]

    def get_all_on_duty(
        self,
        current_time: Optional[datetime] = None,
    ) -> list[Counselor]:
        """获取当前所有值班咨询师

        Args:
            current_time: 当前时间

        Returns:
            list[Counselor]: 值班咨询师列表（按优先级排序）
        """
        if current_time is None:
            current_time = datetime.now()

        day_of_week = current_time.weekday()
        current_hour = current_time.hour

        on_duty = []
        for entry in self._schedule:
            if (entry.day_of_week == day_of_week and
                    entry.start_hour <= current_hour < entry.end_hour):
                counselor = self._counselors.get(entry.counselor_id)
                if counselor and counselor.is_available:
                    on_duty.append(counselor)

        on_duty.sort(key=lambda c: c.priority)
        return on_duty

    def is_during_office_hours(
        self,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """判断当前是否为工作时间

        Args:
            current_time: 当前时间

        Returns:
            bool: 是否为工作日 09:00-18:00
        """
        if current_time is None:
            current_time = datetime.now()

        day_of_week = current_time.weekday()
        current_hour = current_time.hour

        # 周一至周五 09:00-18:00
        return day_of_week < 5 and 9 <= current_hour < 18

    def get_counselor(self, counselor_id: str) -> Optional[Counselor]:
        """根据 ID 获取咨询师信息

        Args:
            counselor_id: 咨询师 ID

        Returns:
            Optional[Counselor]: 咨询师信息
        """
        return self._counselors.get(counselor_id)


# ============================================================
# 全局排班实例
# ============================================================

_global_scheduler: Optional[CounselorScheduler] = None


def get_scheduler() -> CounselorScheduler:
    """获取全局排班系统实例"""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = CounselorScheduler()
    return _global_scheduler

"""复用相同主验收协议，对本地7B与语音专家进行最终选型及训练。"""
from __future__ import annotations

from experiments import round10_robust_final as final
from experiments import round10_large_speech as large


def configure() -> None:
    """只替换预先定义的两种语义专家族，不更改主指标和测试协议。"""
    final.FAMILY_STAGES = ("large_speech", "large_speech5")
    final.FINAL_DIRECTORY = "large_final"
    final.REUSE_COARSE = None
    final.configure = large.configure
    final.data_for = large.data_for


if __name__ == "__main__":
    configure()
    import json
    from threadpoolctl import threadpool_info
    print(json.dumps({"training_numeric_runtime": threadpool_info()}), flush=True)
    final.main()

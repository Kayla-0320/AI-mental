"""浏览器回归专用的本机算法进程；不加载真实用户数据，不调用外部模型。"""
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'algorithm'))
from fastapi import FastAPI
from api.intervention import router
from intervention.support_understanding import prepare_support

app = FastAPI()
app.include_router(router, prefix='/api/v1')

@app.get('/fixture')
def fixture() -> dict:
    """只生成明确标注为测试用的自编对话。"""
    text = '今天朋友没有回消息，我觉得他不想理我，有点难过。'
    return {'text': text, 'support_state': asdict(prepare_support('round1-demo', 'user-1', text, [])),
            'source': 'synthetic_engineering_fixture_not_user_study'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[1]), log_level='warning')

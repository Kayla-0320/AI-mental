// 端到端验证「待审核咨询师 → 管理端审核通过 → 出现在可预约列表」这条真实数据流。
// 用法: node _verify_consultant_flow.mjs   （在 server 目录下运行以复用 prisma 依赖）
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
const API = 'http://127.0.0.1:3000';
let token = null;

async function call(method, path, body) {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let json = null;
  try { json = await res.json(); } catch { /* ignore */ }
  return { status: res.status, json };
}

const stamp = Date.now();
const email = `flow-consultant-${stamp}@mental.com`;
let userId = null, profileId = null;

try {
  // 1) 造一个「刚注册、未审核」的咨询师：isAvailable 用库默认值 true，verifiedAt 为空
  const user = await prisma.user.create({
    data: {
      email, password: 'x', nickname: `流程验证咨询师${stamp}`,
      role: 'CONSULTANT',
      consultantProfile: {
        create: {
          realName: `流程验证${stamp}`, title: '初级心理咨询师',
          license: `TEST-${stamp}`, specialties: '["焦虑","睡眠"]',
          introduction: '端到端流程验证用，验证结束即删除',
        },
      },
    },
    include: { consultantProfile: true },
  });
  userId = user.id;
  profileId = user.consultantProfile.id;
  console.log(`① 造数据: profileId=${profileId} (verifiedAt=${user.consultantProfile.verifiedAt}, isAvailable=${user.consultantProfile.isAvailable})`);

  // 2) 登录管理员
  const login = await call('POST', '/api/auth/login', { email: 'admin@mental.com', password: '123456' });
  token = login.json?.data?.accessToken;
  if (!token) throw new Error(`管理员登录失败: ${JSON.stringify(login)}`);

  // 3) 管理端列表应能看到它，且 status=PENDING
  let list = await call('GET', '/api/admin/consultants?status=PENDING');
  const found = (list.json?.data?.consultants || []).find((c) => c.id === profileId);
  console.log(`② 管理端 PENDING 列表: HTTP ${list.status}, 找到该咨询师=${Boolean(found)}, status=${found?.status}`);

  // 4) 患者端可预约列表**不应**包含它（未审核/未通过不该对外）
  const publicList = await call('GET', '/api/expert/consultants?pageSize=100');
  const inPublic = (publicList.json?.data?.consultants || []).some((c) => c.id === profileId);
  console.log(`③ 患者端可预约列表包含它=${inPublic}  (预期 false)`);

  // 5) 审核通过
  const approve = await call('PUT', `/api/admin/consultants/${profileId}/status`, { status: 'APPROVED' });
  console.log(`④ 审核通过: HTTP ${approve.status}, 返回 status=${approve.json?.data?.status}, verifiedAt=${approve.json?.data?.verifiedAt}`);

  // 6) 通过后应出现在患者端可预约列表
  const publicAfter = await call('GET', '/api/expert/consultants?pageSize=100');
  const inPublicAfter = (publicAfter.json?.data?.consultants || []).some((c) => c.id === profileId);
  console.log(`⑤ 通过后患者端列表包含它=${inPublicAfter}  (预期 true)`);

  // 7) 拒绝后应又从患者端消失
  const reject = await call('PUT', `/api/admin/consultants/${profileId}/status`, { status: 'REJECTED' });
  const publicAfterReject = await call('GET', '/api/expert/consultants?pageSize=100');
  const inPublicReject = (publicAfterReject.json?.data?.consultants || []).some((c) => c.id === profileId);
  console.log(`⑥ 拒绝: HTTP ${reject.status} status=${reject.json?.data?.status}; 患者端仍包含它=${inPublicReject}  (预期 false)`);

  const allOk = Boolean(found) && found.status === 'PENDING' && inPublic === false
    && approve.status === 200 && approve.json?.data?.status === 'APPROVED'
    && inPublicAfter === true && inPublicReject === false;
  console.log(`\n${allOk ? '✓ 全流程通过' : '✗ 流程存在不符合预期的环节'}`);
  process.exitCode = allOk ? 0 : 1;
} catch (e) {
  console.error('验证出错:', e);
  process.exitCode = 1;
} finally {
  // 清理：删除本次验证造的数据（级联删除 profile）
  if (userId) {
    await prisma.user.delete({ where: { id: userId } }).catch(() => {});
    console.log(`\n已清理测试数据 (userId=${userId})`);
  }
  await prisma.$disconnect();
}

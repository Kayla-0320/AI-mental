import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  console.log('开始创建默认账号...\n');

  const hashedPassword = await bcrypt.hash('123456', 12);

  // 1. 管理员账号
  const admin = await prisma.user.upsert({
    where: { email: 'admin@mental.com' },
    update: {},
    create: {
      email: 'admin@mental.com',
      password: hashedPassword,
      nickname: '系统管理员',
      role: 'ADMIN',
    },
  });
  console.log(`[管理员] 邮箱: admin@mental.com | 密码: 123456`);

  // 2. 咨询师账号
  const consultant = await prisma.user.upsert({
    where: { email: 'consultant@mental.com' },
    update: {},
    create: {
      email: 'consultant@mental.com',
      password: hashedPassword,
      nickname: '张医生',
      role: 'CONSULTANT',
    },
  });

  await prisma.consultantProfile.upsert({
    where: { userId: consultant.id },
    update: {},
    create: {
      userId: consultant.id,
      realName: '张明华',
      title: '高级心理咨询师',
      license: 'CPC-2024-001',
      specialties: JSON.stringify(['焦虑症', '抑郁症', '人际关系', '压力管理']),
      introduction: '拥有10年心理咨询经验，擅长认知行为疗法(CBT)和正念疗法。致力于帮助来访者建立健康的思维模式和情绪管理能力。',
      pricePerSession: 300,
      rating: 4.8,
      totalSessions: 156,
      isAvailable: true,
    },
  });
  console.log(`[咨询师] 邮箱: consultant@mental.com | 密码: 123456`);

  // 3. 第二个咨询师
  const consultant2 = await prisma.user.upsert({
    where: { email: 'li@mental.com' },
    update: {},
    create: {
      email: 'li@mental.com',
      password: hashedPassword,
      nickname: '李老师',
      role: 'CONSULTANT',
    },
  });

  await prisma.consultantProfile.upsert({
    where: { userId: consultant2.id },
    update: {},
    create: {
      userId: consultant2.id,
      realName: '李雪',
      title: '中级心理咨询师',
      license: 'CPC-2024-002',
      specialties: JSON.stringify(['青少年心理', '亲子教育', '情绪管理', '自我成长']),
      introduction: '专注于青少年心理健康和家庭教育，擅长沙盘游戏疗法和家庭系统治疗。温柔耐心，善于倾听。',
      pricePerSession: 200,
      rating: 4.6,
      totalSessions: 89,
      isAvailable: true,
    },
  });
  console.log(`[咨询师] 邮箱: li@mental.com | 密码: 123456`);

  // 4. 测试患者账号
  const patient = await prisma.user.upsert({
    where: { email: 'patient@mental.com' },
    update: {},
    create: {
      email: 'patient@mental.com',
      password: hashedPassword,
      nickname: '小明',
      role: 'PATIENT',
    },
  });

  await prisma.patientProfile.upsert({
    where: { userId: patient.id },
    update: {},
    create: {
      userId: patient.id,
      riskLevel: 'LOW',
    },
  });
  console.log(`[患者]   邮箱: patient@mental.com | 密码: 123456`);

  // 5. 添加一些学习内容
  const learningContents = [
    {
      title: '认识焦虑：了解你的情绪信号',
      type: 'article',
      category: 'emotion',
      level: 'beginner',
      summary: '焦虑是人类正常的情绪反应，但过度的焦虑会影响生活质量。本文帮助你认识焦虑的来源和应对方法。',
      content: '焦虑是一种常见的情绪体验...',
      duration: 10,
      isPublished: true,
      publishedAt: new Date(),
    },
    {
      title: '5分钟呼吸放松法',
      type: 'audio',
      category: 'stress',
      level: 'beginner',
      summary: '通过简单的呼吸练习，快速缓解紧张和压力。适合在任何场合使用。',
      content: '音频内容...',
      duration: 5,
      isPublished: true,
      publishedAt: new Date(),
    },
    {
      title: '改善睡眠质量的心理技巧',
      type: 'article',
      category: 'sleep',
      level: 'beginner',
      summary: '失眠往往与心理因素密切相关。学习这些技巧，帮助你建立健康的睡眠习惯。',
      content: '睡眠与心理健康密切相关...',
      duration: 15,
      isPublished: true,
      publishedAt: new Date(),
    },
    {
      title: '认知行为疗法入门',
      type: 'course',
      category: 'self_growth',
      level: 'intermediate',
      summary: '了解CBT的基本原理，学会识别和挑战消极思维模式，建立更健康的认知方式。',
      content: '认知行为疗法(CBT)是一种...',
      duration: 30,
      isPublished: true,
      publishedAt: new Date(),
    },
    {
      title: '正念冥想基础课程',
      type: 'video',
      category: 'stress',
      level: 'beginner',
      summary: '从零开始学习正念冥想，掌握基本的冥想技巧，培养对当下的觉察能力。',
      content: '视频内容...',
      duration: 20,
      isPublished: true,
      publishedAt: new Date(),
    },
    {
      title: '建立健康的人际关系边界',
      type: 'article',
      category: 'relationship',
      level: 'intermediate',
      summary: '学会在人际关系中设定健康的边界，既保护自己又不伤害他人。',
      content: '人际关系边界是指...',
      duration: 12,
      isPublished: true,
      publishedAt: new Date(),
    },
  ];

  for (const content of learningContents) {
    await prisma.learningContent.create({ data: content });
  }
  console.log(`\n已添加 ${learningContents.length} 条学习内容`);

  console.log('\n所有默认数据创建完成！');
}

main()
  .catch((e) => {
    console.error('种子数据创建失败:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

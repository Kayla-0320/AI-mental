import prisma from '../config/database';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { config } from '../config';
import { AppError } from '../middlewares/errorHandler';

interface RegisterInput {
  email: string;
  password: string;
  nickname: string;
  phone?: string;
  role?: 'PATIENT' | 'CONSULTANT' | 'ADMIN';
}

interface LoginInput {
  email: string;
  password: string;
}

class AuthService {
  async register(input: RegisterInput) {
    const existingUser = await prisma.user.findFirst({
      where: {
        OR: [
          { email: input.email },
          ...(input.phone ? [{ phone: input.phone }] : []),
        ],
      },
    });

    if (existingUser) {
      throw new AppError('该邮箱或手机号已被注册', 400);
    }

    const hashedPassword = await bcrypt.hash(input.password, 12);

    const user = await prisma.user.create({
      data: {
        email: input.email,
        password: hashedPassword,
        nickname: input.nickname,
        phone: input.phone,
        role: input.role || 'PATIENT',
      },
      select: {
        id: true,
        email: true,
        nickname: true,
        role: true,
        avatar: true,
        createdAt: true,
      },
    });

    // 如果是患者角色，创建患者档案
    if (user.role === 'PATIENT') {
      await prisma.patientProfile.create({
        data: { userId: user.id },
      });
    }

    const { accessToken, refreshToken } = this.generateTokens(user.id, user.role);

    // 存储 refresh token
    await prisma.refreshToken.create({
      data: {
        token: refreshToken,
        userId: user.id,
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
      },
    });

    return { user, accessToken, refreshToken };
  }

  async login(input: LoginInput) {
    const user = await prisma.user.findUnique({
      where: { email: input.email },
    });

    if (!user || !user.isActive) {
      throw new AppError('邮箱或密码错误', 401);
    }

    const isPasswordValid = await bcrypt.compare(input.password, user.password);
    if (!isPasswordValid) {
      throw new AppError('邮箱或密码错误', 401);
    }

    // 更新最后登录时间
    await prisma.user.update({
      where: { id: user.id },
      data: { lastLoginAt: new Date() },
    });

    const { accessToken, refreshToken } = this.generateTokens(user.id, user.role);

    // 存储 refresh token（使用 upsert 避免重复登录冲突）
    await prisma.refreshToken.upsert({
      where: { userId: user.id },
      update: {
        token: refreshToken,
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
      create: {
        token: refreshToken,
        userId: user.id,
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
    });

    const userProfile = {
      id: user.id,
      email: user.email,
      nickname: user.nickname,
      role: user.role,
      avatar: user.avatar,
    };

    return { user: userProfile, accessToken, refreshToken };
  }

  async refreshToken(token: string) {
    const storedToken = await prisma.refreshToken.findUnique({
      where: { token },
      include: { user: true },
    });

    if (!storedToken || storedToken.expiresAt < new Date()) {
      if (storedToken) {
        await prisma.refreshToken.delete({ where: { id: storedToken.id } });
      }
      throw new AppError('刷新令牌无效或已过期', 401);
    }

    // 删除旧的 refresh token
    await prisma.refreshToken.delete({ where: { id: storedToken.id } });

    const { accessToken, refreshToken: newRefreshToken } = this.generateTokens(
      storedToken.user.id,
      storedToken.user.role
    );

    await prisma.refreshToken.create({
      data: {
        token: newRefreshToken,
        userId: storedToken.user.id,
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
    });

    return { accessToken, refreshToken: newRefreshToken };
  }

  async getProfile(userId: string) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        nickname: true,
        role: true,
        avatar: true,
        phone: true,
        createdAt: true,
        patientProfile: true,
        consultantProfile: true,
      },
    });

    if (!user) {
      throw new AppError('用户不存在', 404);
    }

    return user;
  }

  private generateTokens(userId: string, role: string) {
    const accessToken = jwt.sign(
      { userId, role },
      config.jwt.secret,
      { expiresIn: config.jwt.expiresIn } as jwt.SignOptions
    );

    const refreshToken = jwt.sign(
      { userId },
      config.jwt.refreshSecret,
      { expiresIn: config.jwt.refreshExpiresIn } as jwt.SignOptions
    );

    return { accessToken, refreshToken };
  }
}

export default new AuthService();

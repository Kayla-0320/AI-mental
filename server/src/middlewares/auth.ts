import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { config } from '../config';
import prisma from '../config/database';

export interface AuthRequest extends Request {
  userId?: string;
  userRole?: string;
  /** 用户当前在数据库中的角色（可能与 token 内的快照不同，鉴权以数据库为准） */
  userActive?: boolean;
}

/**
 * 认证中间件。
 *
 * ⚠️ 这里**不只看签名**，还要确认用户仍然存在且未被停用。
 *
 * 原因是「删除我的数据 / 删除账号」必须真的生效：JWT 是无状态的，
 * 签发后 7 天内一直有效。如果只验签名，一个已经注销的账号仍然可以继续调用接口
 * （此前实测：账号删除后 `/api/privacy/summary` 依旧返回 200），
 * 那「删除」就只是删了行，没有撤销访问权。
 *
 * 代价是每个请求多一次主键查询。对当前规模可以接受；
 * 若将来成为瓶颈，应换成「撤权黑名单 + 短时 access token」而不是去掉这道校验。
 */
export const authenticate = async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ code: 401, message: '未提供认证令牌' });
    return;
  }

  const token = authHeader.split(' ')[1];

  let decoded: { userId: string; role: string };
  try {
    decoded = jwt.verify(token, config.jwt.secret) as { userId: string; role: string };
  } catch {
    res.status(401).json({ code: 401, message: '认证令牌无效或已过期' });
    return;
  }

  try {
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
      select: { id: true, role: true, isActive: true },
    });

    if (!user) {
      // 账号已被删除：令牌立即失效，而不是等它自然过期
      res.status(401).json({ code: 401, message: '账号不存在或已注销' });
      return;
    }
    if (!user.isActive) {
      res.status(403).json({ code: 403, message: '账号已被停用' });
      return;
    }

    req.userId = user.id;
    // 角色以数据库为准：token 里的 role 是签发时的快照，管理员改过角色后不应继续沿用旧权限
    req.userRole = user.role;
    req.userActive = user.isActive;
    next();
  } catch (err) {
    next(err);
  }
};

export const authorize = (...roles: string[]) => {
  return (req: AuthRequest, res: Response, next: NextFunction): void => {
    if (!req.userRole || !roles.includes(req.userRole)) {
      res.status(403).json({ code: 403, message: '权限不足' });
      return;
    }
    next();
  };
};

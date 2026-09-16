import { Response } from 'express';
import { AuthRequest } from '../middlewares/auth';
import authService from '../services/auth.service';

class AuthController {
  async register(req: AuthRequest, res: Response) {
    try {
      const result = await authService.register(req.body);
      res.status(201).json({ code: 0, message: '注册成功', data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async login(req: AuthRequest, res: Response) {
    try {
      const result = await authService.login(req.body);
      res.json({ code: 0, message: '登录成功', data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async refreshToken(req: AuthRequest, res: Response) {
    try {
      const result = await authService.refreshToken(req.body.refreshToken);
      res.json({ code: 0, message: '刷新成功', data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getProfile(req: AuthRequest, res: Response) {
    try {
      const result = await authService.getProfile(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }
}

export default new AuthController();

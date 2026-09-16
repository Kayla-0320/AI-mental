import { Response } from 'express';
import { AuthRequest } from '../middlewares/auth';
import profileService from '../services/profile.service';
import prisma from '../config/database';

class ProfileController {
  async getProfile(req: AuthRequest, res: Response) {
    try {
      const patientProfile = await prisma.patientProfile.findUnique({ where: { userId: req.userId! } });
      if (!patientProfile) return res.json({ code: 0, data: null });
      const profile = await profileService.getLatestProfile(patientProfile.id);
      res.json({ code: 0, data: profile });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async updateProfile(req: AuthRequest, res: Response) {
    try {
      const result = await profileService.updateProfile(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getMoodTrend(req: AuthRequest, res: Response) {
    try {
      const days = parseInt(req.query.days as string) || 30;
      const result = await profileService.getMoodTrend(req.userId!, days);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getAssessments(req: AuthRequest, res: Response) {
    try {
      const result = await profileService.getAssessments(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async submitAssessment(req: AuthRequest, res: Response) {
    try {
      const { type, title, answers } = req.body;
      const result = await profileService.submitAssessment(req.userId!, type, title, answers);
      res.status(201).json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async recordMood(req: AuthRequest, res: Response) {
    try {
      const { mood, score, note, tags } = req.body;
      const result = await profileService.recordMood(req.userId!, mood, score, note, tags);
      res.status(201).json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getMoodRecords(req: AuthRequest, res: Response) {
    try {
      const startDate = req.query.startDate ? new Date(req.query.startDate as string) : undefined;
      const endDate = req.query.endDate ? new Date(req.query.endDate as string) : undefined;
      const result = await profileService.getMoodRecords(req.userId!, startDate, endDate);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getTodayEmotionalState(req: AuthRequest, res: Response) {
    try {
      const result = await profileService.getTodayEmotionalState(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getCopingStrategies(req: AuthRequest, res: Response) {
    try {
      const result = await profileService.getCopingStrategies(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async generateRealityTask(req: AuthRequest, res: Response) {
    try {
      const { anxietyLevel, context } = req.body;
      const aiService = (await import('../services/ai.service')).default;
      const result = await aiService.generateRealityTask(anxietyLevel || 50, context);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async reportAnxiety(req: AuthRequest, res: Response) {
    try {
      const { anxietyIndex, level, keyboard, blink, context } = req.body;
      const patientProfile = await prisma.patientProfile.findUnique({ where: { userId: req.userId! } });
      if (patientProfile) {
        await prisma.moodRecord.create({
          data: {
            userId: req.userId!,
            mood: 'anxious',
            score: Math.round(100 - anxietyIndex),
            note: `[焦虑检测] ${context || ''} 指数:${anxietyIndex} 等级:${level}`,
            tags: JSON.stringify({
              type: 'anxiety_detection',
              anxietyIndex,
              level,
              keyboard: keyboard?.anxietyIndex,
              blink: blink?.anxietyIndex,
            }),
          },
        });
        if (level === 'high') {
          await prisma.patientProfile.update({
            where: { userId: req.userId! },
            data: { riskLevel: 'MEDIUM' },
          });
        }
      }
      res.json({ code: 0, message: '上报成功' });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getPatientAnxiety(req: AuthRequest, res: Response) {
    try {
      const { patientId } = req.query;
      if (!patientId) return res.json({ code: 0, data: null });
      const result = await profileService.getPatientAnxiety(patientId as string);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getPatientConsultationData(req: AuthRequest, res: Response) {
    try {
      const { patientId } = req.query;
      if (!patientId) return res.json({ code: 400, message: '缺少患者 ID' });
      const result = await profileService.getPatientConsultationData(patientId as string);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getMembership(req: AuthRequest, res: Response) {
    try {
      const profile = await prisma.patientProfile.findUnique({
        where: { userId: req.userId! },
        select: { membershipType: true, membershipExpiresAt: true },
      });
      res.json({ code: 0, data: profile || { membershipType: 'FREE', membershipExpiresAt: null } });
    } catch (error: any) {
      res.status(500).json({ code: 500, message: error.message });
    }
  }

  async updateMembership(req: AuthRequest, res: Response) {
    try {
      const { membershipType } = req.body;
      if (!['FREE', 'BASIC', 'PREMIUM'].includes(membershipType)) {
        return res.status(400).json({ code: 400, message: '无效的会员类型' });
      }
      const profile = await prisma.patientProfile.upsert({
        where: { userId: req.userId! },
        create: {
          userId: req.userId!,
          membershipType,
          membershipExpiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
        },
        update: {
          membershipType,
          membershipExpiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
        },
      });
      res.json({ code: 0, data: profile });
    } catch (error: any) {
      res.status(500).json({ code: 500, message: error.message });
    }
  }
}

export default new ProfileController();

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
      const {
        anxietyIndex, level, context, multimodal,
        emotionProbs, dominantEmotion, confidence,
        keyboard, text, voice, facial, fusionWeights, evidence,
      } = req.body;

      const patientProfile = await prisma.patientProfile.findUnique({ where: { userId: req.userId! } });
      if (patientProfile) {
        // 构建多模态标签数据
        const perceptionTags: Record<string, any> = {
          type: 'multimodal_anxiety_detection',
          anxietyIndex, level,
          dominantEmotion, confidence,
          activeModalities: multimodal ? Object.entries(multimodal).filter(([, v]) => v).map(([k]) => k) : [],
        };

        // 键盘动力学摘要
        if (keyboard) {
          perceptionTags.keyboard = {
            typingSpeed: keyboard.typingSpeed,
            anxietyIndex: keyboard.anxietyIndex,
            rhythmScore: keyboard.rhythmScore,
            pauseCount: keyboard.pauseCount,
            backspaceBurst: keyboard.backspaceBurst,
            pressureIndex: keyboard.pressureIndex,
          };
        }

        // 文本语义摘要
        if (text) {
          perceptionTags.text = {
            dominantEmotion: text.dominantEmotion,
            sentiment: text.sentiment,
            crisisLevel: text.crisisLevel,
            crisisMarkers: text.crisisMarkers,
            cognitiveMarkers: text.cognitiveMarkers,
            emotionalIntensity: text.emotionalIntensity,
          };
        }

        // 声音声学摘要
        if (voice) {
          perceptionTags.voice = {
            pitchMean: voice.pitch?.mean,
            energyMean: voice.energy?.mean,
            speechRate: voice.rate?.speechRate,
            pauseRatio: voice.pauses?.ratio,
            spectralCentroid: voice.spectral?.centroid,
            detectedEmotion: voice.detectedEmotion,
          };
        }

        // 面部微表情摘要
        if (facial) {
          perceptionTags.facial = {
            dominantExpression: facial.dominantExpression,
            expressionIntensity: facial.expressionIntensity,
            microExpressionCount: facial.microExpressions?.count,
            emotionMapping: facial.emotionMapping,
            confidence: facial.confidence,
          };
        }

        // 融合权重
        if (fusionWeights) {
          perceptionTags.fusionWeights = fusionWeights;
        }

        // 危机信号时在 note 中特别标记
        const crisisNote = text?.crisisLevel === 'high'
          ? `[多模态感知][危机!] ${dominantEmotion || '未知'} | 危机词:${text?.crisisMarkers?.join(',')} | ${context || ''}`
          : `[多模态感知] ${dominantEmotion || '未知'} | 置信度:${Math.round((confidence || 0) * 100)}% | ${context || ''}`;

        await prisma.moodRecord.create({
          data: {
            userId: req.userId!,
            mood: dominantEmotion === '快乐' ? 'happy' : dominantEmotion === '悲伤' ? 'sad' : 'anxious',
            score: Math.round(100 - anxietyIndex),
            note: crisisNote,
            tags: JSON.stringify(perceptionTags),
          },
        });

        // 高风险时更新风险等级
        if (level === 'high' || level === 'crisis') {
          await prisma.patientProfile.update({
            where: { userId: req.userId! },
            data: { riskLevel: level === 'crisis' ? 'HIGH' : 'MEDIUM' },
          });
        }

      }
      res.json({ code: 0, message: '多模态数据上报成功' });
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

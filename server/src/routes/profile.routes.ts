
import { Router } from 'express';
import profileController from '../controllers/profile.controller';
import { authenticate } from '../middlewares/auth';

const router = Router();

router.use(authenticate);

router.get('/profile/psychological', profileController.getProfile);
router.post('/profile/update', profileController.updateProfile);
router.get('/profile/mood-trend', profileController.getMoodTrend);
router.get('/assessments', profileController.getAssessments);
router.post('/assessments', profileController.submitAssessment);
router.post('/mood', profileController.recordMood);
router.get('/mood', profileController.getMoodRecords);
router.get('/profile/emotional-state', profileController.getTodayEmotionalState);
router.get('/profile/coping-strategies', profileController.getCopingStrategies);
router.post('/profile/reality-task', profileController.generateRealityTask);
router.post('/profile/anxiety-report', profileController.reportAnxiety);
router.get('/profile/patient-anxiety', profileController.getPatientAnxiety);
router.get('/profile/consultation-data', profileController.getPatientConsultationData);
router.get('/membership', profileController.getMembership);
router.put('/membership', profileController.updateMembership);

export default router;

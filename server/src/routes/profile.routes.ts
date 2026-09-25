
import { Router } from 'express';
import profileController from '../controllers/profile.controller';
import { authenticate } from '../middlewares/auth';

const router = Router();

router.use(authenticate);

router.get('/psychological', profileController.getProfile);
router.post('/update', profileController.updateProfile);
router.get('/mood-trend', profileController.getMoodTrend);
router.get('/assessments', profileController.getAssessments);
router.post('/assessments', profileController.submitAssessment);
router.post('/mood', profileController.recordMood);
router.get('/mood', profileController.getMoodRecords);
router.get('/emotional-state', profileController.getTodayEmotionalState);
router.get('/coping-strategies', profileController.getCopingStrategies);
router.post('/reality-task', profileController.generateRealityTask);
router.post('/anxiety-report', profileController.reportAnxiety);
router.get('/patient-anxiety', profileController.getPatientAnxiety);
router.get('/consultation-data', profileController.getPatientConsultationData);
router.get('/membership', profileController.getMembership);
router.put('/membership', profileController.updateMembership);

export default router;

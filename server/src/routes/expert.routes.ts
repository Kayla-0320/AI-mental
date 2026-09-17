import { Router } from 'express';
import expertController from '../controllers/expert.controller';
import { authenticate } from '../middlewares/auth';

const router = Router();

router.use(authenticate);

router.get('/consultants', expertController.getConsultants);
router.get('/consultants/:consultantId', expertController.getConsultantDetail);
router.get('/my-profile', expertController.getMyProfile);
router.put('/my-profile', expertController.updateMyProfile);
router.post('/bookings', expertController.createBooking);
router.get('/bookings', expertController.getBookings);
router.get('/bookings/:bookingId', expertController.getBookingById);
router.put('/bookings/:bookingId/status', expertController.updateBookingStatus);
router.post('/bookings/:bookingId/feedback', expertController.submitFeedback);
router.post('/bookings/:bookingId/messages', expertController.sendMessage);
router.get('/bookings/:bookingId/messages', expertController.getMessages);

export default router;

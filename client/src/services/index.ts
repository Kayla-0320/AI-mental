import api from './api';

// 认证相关
export const authApi = {
  register: (data: { email: string; password: string; nickname: string; phone?: string }) =>
    api.post('/auth/register', data),
  login: (data: { email: string; password: string }) =>
    api.post('/auth/login', data),
  getProfile: () => api.get('/auth/profile'),
};

// AI 咨询相关
export const consultationApi = {
  createConversation: (title?: string) =>
    api.post('/consultation/conversations', { title }),
  getConversations: (page = 1, pageSize = 20) =>
    api.get(`/consultation/conversations?page=${page}&pageSize=${pageSize}`),
  getMessages: (conversationId: string, page = 1) =>
    api.get(`/consultation/conversations/${conversationId}/messages?page=${page}`),
  sendMessage: (conversationId: string, content: string, contentType = 'text') =>
    api.post(`/consultation/conversations/${conversationId}/messages`, { content, contentType }),
  sendSocraticMessage: (conversationId: string, content: string) =>
    api.post(`/consultation/conversations/${conversationId}/socratic`, { content }),
  deleteConversation: (conversationId: string) =>
    api.delete(`/consultation/conversations/${conversationId}`),
};

// 心理画像相关
export const profileApi = {
  getPsychological: () => api.get('/profile/psychological'),
  updateProfile: () => api.post('/profile/update'),
  getMoodTrend: (days = 30) => api.get(`/profile/mood-trend?days=${days}`),
  getAssessments: () => api.get('/profile/assessments'),
  submitAssessment: (data: { type: string; title: string; answers: any }) =>
    api.post('/profile/assessments', data),
  recordMood: (data: { mood: string; score: number; note?: string; tags?: string[] }) =>
    api.post('/profile/mood', data),
  getMoodRecords: (startDate?: string, endDate?: string) =>
    api.get(`/profile/mood?startDate=${startDate}&endDate=${endDate}`),
  getTodayEmotionalState: () => api.get('/profile/emotional-state'),
  getCopingStrategies: () => api.get('/profile/coping-strategies'),
  generateRealityTask: (data: { anxietyLevel: number; context?: string }) =>
    api.post('/profile/reality-task', data),
};

// 疗愈室相关
export const healingApi = {
  createSession: (data: { type: string; title: string; duration?: number }) =>
    api.post('/healing/sessions', data),
  completeSession: (sessionId: string, data: any) =>
    api.put(`/healing/sessions/${sessionId}/complete`, data),
  getSessions: (type?: string, page = 1) =>
    api.get(`/healing/sessions?type=${type || ''}&page=${page}`),
  getStats: () => api.get('/healing/stats'),
};

// 专家连线相关
export const expertApi = {
  getConsultants: (params?: any) =>
    api.get('/expert/consultants', { params }),
  getConsultantDetail: (id: string) =>
    api.get(`/expert/consultants/${id}`),
  createBooking: (data: { consultantId: string; scheduledAt: string; notes?: string; type?: string }) =>
    api.post('/expert/bookings', data),
  getBookings: () => api.get('/expert/bookings'),
  updateBookingStatus: (bookingId: string, status: string) =>
    api.put(`/expert/bookings/${bookingId}/status`, { status }),
  submitFeedback: (bookingId: string, data: { rating: number; feedback?: string }) =>
    api.post(`/expert/bookings/${bookingId}/feedback`, data),
  sendMessage: (bookingId: string, content: string) =>
    api.post(`/expert/bookings/${bookingId}/messages`, { content }),
  getMessages: (bookingId: string, page = 1) =>
    api.get(`/expert/bookings/${bookingId}/messages?page=${page}`),
  getMyProfile: () => api.get('/expert/my-profile'),
  updateMyProfile: (data: { title?: string; specialties?: string; introduction?: string; pricePerSession?: number }) =>
    api.put('/expert/my-profile', data),
};

// 治疗规划相关
export const treatmentApi = {
  createPlan: (data: { title: string; description?: string }) =>
    api.post('/treatment/plans', data),
  getPlans: () => api.get('/treatment/plans'),
  getPlanDetail: (planId: string) =>
    api.get(`/treatment/plans/${planId}`),
  updatePlanStatus: (planId: string, status: string) =>
    api.put(`/treatment/plans/${planId}/status`, { status }),
  addTask: (planId: string, data: any) =>
    api.post(`/treatment/plans/${planId}/tasks`, data),
  updateTaskStatus: (taskId: string, status: string, notes?: string) =>
    api.put(`/treatment/tasks/${taskId}/status`, { status, notes }),
};

// 持续学习相关
export const learningApi = {
  getContents: (params?: any) =>
    api.get('/learning/contents', { params }),
  getContentDetail: (contentId: string) =>
    api.get(`/learning/contents/${contentId}`),
  getUserRecords: () => api.get('/learning/records'),
  updateProgress: (contentId: string, progress: number, notes?: string) =>
    api.put(`/learning/contents/${contentId}/progress`, { progress, notes }),
  getRecommendations: () => api.get('/learning/recommendations'),
  getStats: () => api.get('/learning/stats'),
};

// 危机干预 & 紧急联系人
export const crisisApi = {
  getHotlines: () => api.get('/crisis/hotlines'),
  detectCrisis: (text: string) => api.post('/crisis/detect', { text }),
  getContacts: () => api.get('/crisis/contacts'),
  addContact: (data: { name: string; relation: string; phone: string; isPrimary?: boolean }) =>
    api.post('/crisis/contacts', data),
  updateContact: (id: string, data: any) => api.put(`/crisis/contacts/${id}`, data),
  deleteContact: (id: string) => api.delete(`/crisis/contacts/${id}`),
};

// 心情打卡 & 情绪趋势
export const moodApi = {
  checkIn: (data: { mood: string; score: number; energy?: number; sleepHours?: number; note?: string; tags?: string[] }) =>
    api.post('/mood/checkin', data),
  getCheckInHistory: (days = 30) => api.get(`/mood/checkin/history?days=${days}`),
  getMoodTrend: (days = 30) => api.get(`/mood/trend?days=${days}`),
  getCheckInCalendar: (year: number, month: number) =>
    api.get(`/mood/checkin/calendar?year=${year}&month=${month}`),
  getTodayStatus: () => api.get('/mood/checkin/today'),
};

// 咨询评价
export const reviewApi = {
  createReview: (data: { bookingId: string; rating: number; comment?: string; tags?: string[]; isAnonymous?: boolean }) =>
    api.post('/reviews', data),
  getConsultantReviews: (consultantId: string, page = 1, limit = 10) =>
    api.get(`/reviews/consultant/${consultantId}?page=${page}&limit=${limit}`),
};

// 社区互助
export const communityApi = {
  getPosts: (category?: string, page = 1, limit = 10) =>
    api.get(`/community/posts?category=${category || ''}&page=${page}&limit=${limit}`),
  createPost: (data: { title: string; content: string; category: string; tags?: string[]; isAnonymous?: boolean }) =>
    api.post('/community/posts', data),
  getPost: (id: string) => api.get(`/community/posts/${id}`),
  toggleLike: (id: string) => api.post(`/community/posts/${id}/like`),
  addComment: (id: string, content: string, isAnonymous = true) =>
    api.post(`/community/posts/${id}/comments`, { content, isAnonymous }),
  getPending: () => api.get('/community/pending'),
  approvePost: (id: string, approved: boolean) => api.put(`/community/posts/${id}/approve`, { approved }),
};

// 成就/睡眠/支付/报告/反馈
export const extraApi = {
  getAchievements: () => api.get('/extra/achievements'),
  getAllAchievements: () => api.get('/extra/achievements/all'),
  recordSleep: (data: { bedTime: string; wakeTime: string; quality: number; dreamRecall?: string; notes?: string }) =>
    api.post('/extra/sleep', data),
  getSleepRecords: (days = 30) => api.get(`/extra/sleep?days=${days}`),
  getSleepStats: (days = 30) => api.get(`/extra/sleep/stats?days=${days}`),
  createPayment: (data: { bookingId: string; amount: number; method: string; description?: string }) =>
    api.post('/extra/payment', data),
  getPayments: () => api.get('/extra/payments'),
  generateWeeklyReport: () => api.post('/extra/reports/weekly'),
  getReports: (type?: string) => api.get(`/extra/reports?type=${type || ''}`),
  createFeedback: (data: { type: string; title: string; content: string; contact?: string }) =>
    api.post('/extra/feedback', data),
  getFeedbacks: () => api.get('/extra/feedback'),
  getAllFeedbacks: (status?: string) => api.get(`/extra/feedback/all?status=${status || ''}`),
  replyFeedback: (id: string, reply: string) => api.put(`/extra/feedback/${id}/reply`, { reply }),
};

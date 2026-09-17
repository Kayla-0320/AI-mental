import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import PatientLayout from './pages/patient/PatientLayout';
import ConsultantLayout from './pages/consultant/ConsultantLayout';
import AdminLayout from './pages/admin/AdminLayout';
import Login from './pages/Login';
import Register from './pages/Register';
import Home from './pages/patient/Home';
import Chat from './pages/patient/Chat';
import Healing from './pages/patient/Healing';
import Experts from './pages/patient/Experts';
import PatientRoom from './pages/patient/PatientRoom';
import Settings from './pages/patient/Settings';
import Companions from './pages/patient/Companions';
import MyGrowth from './pages/patient/MyGrowth';
import Profile from './pages/patient/Profile';
import SocraticChat from './pages/patient/SocraticChat';
import ConsultantDashboard from './pages/consultant/ConsultantDashboard';
import ConsultantAppointments from './pages/consultant/ConsultantAppointments';
import ConsultantConsultations from './pages/consultant/ConsultantConsultations';
import ConsultantProfiles from './pages/consultant/ConsultantProfiles';
import ConsultantRoom from './pages/consultant/ConsultantRoom';
import ConsultantSettings from './pages/consultant/ConsultantSettings';
import Dashboard from './pages/admin/Dashboard';
import AdminUsers from './pages/admin/AdminUsers';
import AdminConsultants from './pages/admin/AdminConsultants';
import AdminReview from './pages/admin/AdminReview';
import AdminCrisis from './pages/admin/AdminCrisis';
import AdminFeedback from './pages/admin/AdminFeedback';

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
}

function RoleRoute({ children, role }: { children: React.ReactNode; role: string }) {
  const userRole = useAuthStore((state) => state.user?.role);
  if (userRole === role) return <>{children}</>;
  // 角色不匹配，跳转到对应页面
  if (userRole === 'PATIENT') return <Navigate to="/" replace />;
  if (userRole === 'CONSULTANT') return <Navigate to="/consultant" replace />;
  if (userRole === 'ADMIN') return <Navigate to="/admin" replace />;
  return <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      {/* 患者端 */}
      <Route path="/" element={<PrivateRoute><RoleRoute role="PATIENT"><PatientLayout /></RoleRoute></PrivateRoute>}>
        <Route index element={<Home />} />
        <Route path="chat" element={<Chat />} />
        <Route path="chat/:conversationId" element={<Chat />} />
        <Route path="socratic" element={<SocraticChat />} />
        <Route path="companions" element={<Companions />} />
        <Route path="healing" element={<Healing />} />
        <Route path="assessment" element={<Navigate to="/profile" replace />} />
        <Route path="anxiety" element={<Navigate to="/profile" replace />} />
        <Route path="growth" element={<MyGrowth />} />
        <Route path="profile" element={<Profile />} />
        <Route path="experts" element={<Experts />} />
        <Route path="experts/room/:bookingId" element={<PatientRoom />} />
        <Route path="settings" element={<Settings />} />
        {/* 旧路由重定向 */}
        <Route path="treehole" element={<Navigate to="/companions" replace />} />
        <Route path="community" element={<Navigate to="/companions" replace />} />
        <Route path="achievements" element={<Navigate to="/growth" replace />} />
        <Route path="sleep" element={<Navigate to="/healing" replace />} />
        <Route path="feedback" element={<Navigate to="/settings" replace />} />
        <Route path="treatment" element={<Navigate to="/chat" replace />} />
        <Route path="learning" element={<Navigate to="/healing" replace />} />
      </Route>

      {/* 咨询师端 */}
      <Route path="/consultant" element={<PrivateRoute><RoleRoute role="CONSULTANT"><ConsultantLayout /></RoleRoute></PrivateRoute>}>
        <Route index element={<ConsultantDashboard />} />
        <Route path="appointments" element={<ConsultantAppointments />} />
        <Route path="consultations" element={<ConsultantConsultations />} />
        <Route path="consultations/:bookingId" element={<ConsultantRoom />} />
        <Route path="profiles" element={<ConsultantProfiles />} />
        <Route path="settings" element={<ConsultantSettings />} />
      </Route>

      {/* 管理后台 */}
      <Route path="/admin" element={<PrivateRoute><RoleRoute role="ADMIN"><AdminLayout /></RoleRoute></PrivateRoute>}>
        <Route index element={<Dashboard />} />
        <Route path="users" element={<AdminUsers />} />
        <Route path="consultants" element={<AdminConsultants />} />
        <Route path="review" element={<AdminReview />} />
        <Route path="crisis" element={<AdminCrisis />} />
        <Route path="feedback" element={<AdminFeedback />} />
      </Route>

      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

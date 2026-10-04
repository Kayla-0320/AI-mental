import { create } from 'zustand';

interface UserInfo {
  id: string;
  email: string;
  nickname: string;
  role: string;
  avatar?: string;
}

interface AuthState {
  user: UserInfo | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  setAuth: (user: UserInfo, accessToken: string, refreshToken: string) => void;
  logout: () => void;
  updateUser: (user: Partial<UserInfo>) => void;
}

function loadUser(): UserInfo | null {
  try {
    const raw = localStorage.getItem('user');
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

export const useAuthStore = create<AuthState>((set) => ({
  user: loadUser(),
  accessToken: localStorage.getItem('accessToken'),
  isAuthenticated: !!localStorage.getItem('accessToken'),

  setAuth: (user, accessToken, refreshToken) => {
    localStorage.setItem('accessToken', accessToken);
    localStorage.setItem('refreshToken', refreshToken);
    localStorage.setItem('user', JSON.stringify(user));
    set({ user, accessToken, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('user');
    set({ user: null, accessToken: null, isAuthenticated: false });
  },

  updateUser: (userData) => {
    set((state) => {
      const user = state.user ? { ...state.user, ...userData } : null;
      if (user) localStorage.setItem('user', JSON.stringify(user));
      return { user };
    });
  },
}));

import { useState, useEffect } from 'react';
import { Card, Typography, Table, Tag, Space, Button, Input, message, Modal, Select, Popconfirm } from 'antd';
import { UserOutlined, SearchOutlined, LockOutlined, UnlockOutlined } from '@ant-design/icons';
import api from '../../services/api';

const { Title, Text } = Typography;

const roleColors: Record<string, string> = { PATIENT: 'blue', CONSULTANT: 'green', ADMIN: 'red' };
const roleLabels: Record<string, string> = { PATIENT: '用户', CONSULTANT: '咨询师', ADMIN: '管理员' };

export default function AdminUsers() {
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);

  useEffect(() => { loadUsers(); }, [page]);

  const loadUsers = async () => {
    setLoading(true);
    try {
      const res = await api.get(`/auth/users?page=${page}&limit=10&search=${search}`) as any;
      const list = res.data?.users || [];
      setUsers(list.map((u: any, i: number) => ({ ...u, key: u.id })));
      setTotal(res.data?.total || list.length);
    } catch {
      // 模拟数据
      setUsers([
        { key: '1', id: '1', nickname: '小明', email: 'xm@test.com', role: 'PATIENT', status: 'ACTIVE', createdAt: new Date().toISOString() },
        { key: '2', id: '2', nickname: '小红', email: 'xh@test.com', role: 'PATIENT', status: 'ACTIVE', createdAt: new Date().toISOString() },
        { key: '3', id: '3', nickname: '张咨询师', email: 'zz@test.com', role: 'CONSULTANT', status: 'ACTIVE', createdAt: new Date().toISOString() },
      ]);
      setTotal(3);
    } finally { setLoading(false); }
  };

  const handleToggleStatus = async (userId: string, currentStatus: string) => {
    try {
      await api.put(`/auth/users/${userId}/status`, { status: currentStatus === 'ACTIVE' ? 'DISABLED' : 'ACTIVE' });
      message.success('操作成功');
      loadUsers();
    } catch {
      message.error('操作失败');
    }
  };

  const columns = [
    { title: '昵称', dataIndex: 'nickname', key: 'nickname' },
    { title: '邮箱', dataIndex: 'email', key: 'email' },
    { title: '角色', dataIndex: 'role', key: 'role', render: (r: string) => <Tag color={roleColors[r]}>{roleLabels[r]}</Tag> },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color={s === 'ACTIVE' ? 'green' : 'red'}>{s === 'ACTIVE' ? '正常' : '禁用'}</Tag> },
    {
      title: '注册时间', dataIndex: 'createdAt', key: 'createdAt',
      render: (d: string) => new Date(d).toLocaleDateString(),
    },
    {
      title: '操作', key: 'action',
      render: (_: any, record: any) => (
        <Popconfirm
          title={`确定${record.status === 'ACTIVE' ? '禁用' : '启用'}该用户？`}
          onConfirm={() => handleToggleStatus(record.id, record.status)}
        >
          <Button type="link" size="small" danger={record.status === 'ACTIVE'}
            icon={record.status === 'ACTIVE' ? <LockOutlined /> : <UnlockOutlined />}>
            {record.status === 'ACTIVE' ? '禁用' : '启用'}
          </Button>
        </Popconfirm>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4}><UserOutlined /> 用户管理</Title>
        <Input.Search
          placeholder="搜索用户昵称或邮箱"
          value={search}
          onChange={e => setSearch(e.target.value)}
          onSearch={() => { setPage(1); loadUsers(); }}
          style={{ width: 300 }}
          allowClear
        />
      </div>
      <Card style={{ borderRadius: 12 }}>
        <Table columns={columns} dataSource={users} loading={loading}
          pagination={{ current: page, total, pageSize: 10, onChange: setPage }} />
      </Card>
    </div>
  );
}

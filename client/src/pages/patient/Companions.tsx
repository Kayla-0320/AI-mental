import { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Button, Space, Tag, Input, Modal, message, Empty, Avatar, Pagination, Tabs } from 'antd';
import { CommentOutlined, HeartOutlined, SendOutlined, PlusOutlined, EyeInvisibleOutlined } from '@ant-design/icons';
import { communityApi } from '../../services';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

const categories = [
  { key: '', label: '全部' },
  { key: 'share', label: '心情分享' },
  { key: 'question', label: '求助问答' },
  { key: 'encouragement', label: '互相鼓励' },
  { key: 'experience', label: '康复经验' },
];

export default function Companions() {
  const [activeTab, setActiveTab] = useState('community');
  const [posts, setPosts] = useState<any[]>([]);
  const [category, setCategory] = useState('');
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [createModal, setCreateModal] = useState(false);
  const [newPost, setNewPost] = useState({ title: '', content: '', category: 'share', isAnonymous: false });
  const [commentModal, setCommentModal] = useState(false);
  const [currentPost, setCurrentPost] = useState<any>(null);
  const [comment, setComment] = useState('');

  const loadPosts = () => {
    communityApi.getPosts(category, page).then((res: any) => {
      if (res.code === 0) {
        setPosts(res.data.posts);
        setTotal(res.data.total);
      }
    });
  };

  useEffect(() => { loadPosts(); }, [category, page]);

  const handleCreate = async () => {
    if (!newPost.title || !newPost.content) return message.warning('请填写标题和内容');
    try {
      await communityApi.createPost(newPost);
      message.success('发布成功，等待审核后显示');
      setCreateModal(false);
      setNewPost({ title: '', content: '', category: 'share', isAnonymous: false });
      loadPosts();
    } catch (e: any) { message.error(e.message); }
  };

  const handleLike = async (id: string) => {
    try {
      const res: any = await communityApi.toggleLike(id);
      if (res.code === 0) loadPosts();
    } catch {}
  };

  const handleComment = async () => {
    if (!comment || !currentPost) return;
    try {
      await communityApi.addComment(currentPost.id, comment);
      message.success('评论成功，等待审核');
      setComment('');
      setCommentModal(false);
    } catch (e: any) { message.error(e.message); }
  };

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16, color: '#5a4a6a' }}>
        🫂 同伴空间
      </Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        在这里，你不是一个人。和懂你的人聊聊吧。
      </Text>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'community',
            label: '💬 社区广场',
            children: (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                  <Space wrap>
                    {categories.map(c => (
                      <Tag
                        key={c.key}
                        onClick={() => { setCategory(c.key); setPage(1); }}
                        style={{
                          cursor: 'pointer',
                          padding: '6px 16px',
                          borderRadius: 20,
                          background: category === c.key ? '#ffb6c1' : '#fff5f7',
                          color: category === c.key ? '#fff' : '#ff8fab',
                          border: 'none',
                          fontWeight: category === c.key ? 600 : 400,
                        }}
                      >
                        {c.label}
                      </Tag>
                    ))}
                  </Space>
                  <Button className="cloud-btn" icon={<PlusOutlined />} onClick={() => setCreateModal(true)}>
                    发帖
                  </Button>
                </div>

                {posts.length === 0 ? (
                  <Empty description="还没有人发帖，来做第一个分享的人吧~" />
                ) : (
                  <Row gutter={[16, 16]}>
                    {posts.map((post: any) => (
                      <Col xs={24} sm={12} lg={8} key={post.id}>
                        <Card className="cloud-card" hoverable onClick={() => { setCurrentPost(post); setCommentModal(true); }}>
                          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                            <Avatar size={28} style={{ backgroundColor: post.isAnonymous ? '#d9d9d9' : '#ffb6c1' }}>
                              {post.isAnonymous ? <EyeInvisibleOutlined /> : (post.user?.nickname?.[0] || '匿')}
                            </Avatar>
                            <Text style={{ marginLeft: 8, fontSize: 13, color: '#8a7a9a' }}>
                              {post.isAnonymous ? '匿名树洞' : post.user?.nickname}
                            </Text>
                            <Tag color="pink" style={{ marginLeft: 'auto', fontSize: 11, borderRadius: 20 }}>
                              {categories.find(c => c.key === post.category)?.label || post.category}
                            </Tag>
                          </div>
                          <Title level={5} style={{ marginBottom: 4, color: '#5a4a6a' }}>{post.title}</Title>
                          <Paragraph ellipsis={{ rows: 2 }} style={{ color: '#8a7a9a', fontSize: 13 }}>
                            {post.content}
                          </Paragraph>
                          <Space size="middle" style={{ marginTop: 8 }}>
                            <Button
                              type="text"
                              size="small"
                              icon={<HeartOutlined />}
                              onClick={(e) => { e.stopPropagation(); handleLike(post.id); }}
                              style={{ color: '#ff8fab' }}
                            >
                              {post.likeCount || 0}
                            </Button>
                            <Text style={{ fontSize: 12, color: '#b0a0c0' }}>
                              <CommentOutlined style={{ marginRight: 4 }} />{post.commentCount || 0}
                            </Text>
                          </Space>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                )}

                {posts.length > 0 && total > 10 && (
                  <div style={{ textAlign: 'center', marginTop: 24 }}>
                    <Pagination current={page} total={total} pageSize={10} onChange={setPage} showSizeChanger={false} />
                  </div>
                )}
              </div>
            ),
          },
          {
            key: 'treehole',
            label: '🕳️ 匿名树洞',
            children: (
              <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                <div style={{ fontSize: 64, marginBottom: 16 }}>🌙</div>
                <Title level={4} style={{ color: '#5a4a6a' }}>这里是安全的树洞</Title>
                <Paragraph style={{ color: '#8a7a9a', maxWidth: 400, margin: '0 auto 24px' }}>
                  说出你不敢说的话。所有内容完全匿名，没有人会知道是你。
                </Paragraph>
                <Button
                  className="cloud-btn"
                  size="large"
                  icon={<PlusOutlined />}
                  onClick={() => { setNewPost({ ...newPost, isAnonymous: true, category: 'share' }); setCreateModal(true); }}
                >
                  匿名倾诉
                </Button>
                <div style={{ marginTop: 32, padding: 20, background: '#fff5f7', borderRadius: 16, maxWidth: 500, margin: '32px auto 0' }}>
                  <Text type="secondary" style={{ fontSize: 13 }}>
                    💡 小提示：如果你现在很痛苦，也可以直接去「聊聊」找 AI 倾诉，它会一直陪着你。
                  </Text>
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* 创建帖子弹窗 */}
      <Modal
        open={createModal}
        onCancel={() => setCreateModal(false)}
        onOk={handleCreate}
        title={newPost.isAnonymous ? '匿名倾诉' : '发布新帖'}
        okText="发布"
        cancelText="取消"
      >
        <Input
          placeholder="标题（可以写「无题」）"
          value={newPost.title}
          onChange={e => setNewPost({ ...newPost, title: e.target.value })}
          style={{ marginBottom: 12, borderRadius: 12 }}
        />
        {!newPost.isAnonymous && (
          <Space style={{ marginBottom: 12 }}>
            {categories.filter(c => c.key).map(c => (
              <Tag
                key={c.key}
                onClick={() => setNewPost({ ...newPost, category: c.key })}
                style={{
                  cursor: 'pointer',
                  background: newPost.category === c.key ? '#ffb6c1' : '#fff5f7',
                  color: newPost.category === c.key ? '#fff' : '#ff8fab',
                  borderRadius: 20,
                }}
              >
                {c.label}
              </Tag>
            ))}
          </Space>
        )}
        <TextArea
          rows={4}
          placeholder={newPost.isAnonymous ? '说出你心里的话...' : '分享你的想法...'}
          value={newPost.content}
          onChange={e => setNewPost({ ...newPost, content: e.target.value })}
          style={{ borderRadius: 12 }}
        />
      </Modal>

      {/* 评论弹窗 */}
      <Modal
        open={commentModal}
        onCancel={() => setCommentModal(false)}
        onOk={handleComment}
        title={currentPost?.title}
        okText="发表评论"
      >
        {currentPost && (
          <>
            <Paragraph style={{ color: '#5a4a6a', background: '#fff5f7', padding: 12, borderRadius: 12 }}>
              {currentPost.content}
            </Paragraph>
            <TextArea
              rows={3}
              placeholder="写下你的评论..."
              value={comment}
              onChange={e => setComment(e.target.value)}
              style={{ borderRadius: 12 }}
            />
          </>
        )}
      </Modal>
    </div>
  );
}

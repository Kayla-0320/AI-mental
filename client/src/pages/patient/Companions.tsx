import { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Button, Space, Tag, Input, Modal, message, Empty, Avatar, Pagination, Tabs, Divider, List } from 'antd';
import { CommentOutlined, HeartOutlined, SendOutlined, PlusOutlined, EyeInvisibleOutlined, ArrowLeftOutlined, MessageOutlined } from '@ant-design/icons';
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
  const [postDetail, setPostDetail] = useState<any>(null);
  const [postComments, setPostComments] = useState<any[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);

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

  const openPostDetail = async (post: any) => {
    setPostDetail(post);
    setDetailLoading(true);
    try {
      const res = await communityApi.getPost(post.id) as any;
      if (res.data) {
        setPostDetail(res.data.post || res.data);
        setPostComments(res.data.comments || []);
      }
    } catch {
      setPostComments([]);
    } finally {
      setDetailLoading(false);
    }
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

  const handleDetailComment = async () => {
    if (!comment || !postDetail) return;
    try {
      await communityApi.addComment(postDetail.id, comment);
      message.success('评论成功，等待审核');
      setComment('');
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
                  <div style={{ textAlign: 'center', padding: '60px 20px' }}>
                    <div style={{ fontSize: 56, marginBottom: 16 }}>🌱</div>
                    <Title level={5} style={{ color: '#5a4a6a', marginBottom: 8 }}>还没有人发帖</Title>
                    <Text type="secondary" style={{ display: 'block', marginBottom: 20 }}>
                      来做第一个分享的人吧，你的每一句话都可能温暖另一个人
                    </Text>
                    <Button className="cloud-btn" icon={<PlusOutlined />} onClick={() => setCreateModal(true)}>
                      发布第一条帖子
                    </Button>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {posts.map((post: any) => (
                      <Card
                        key={post.id}
                        hoverable
                        onClick={() => openPostDetail(post)}
                        style={{ borderRadius: 12, border: 'none', boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}
                        bodyStyle={{ padding: '16px 20px' }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
                          <Avatar size={32} style={{ backgroundColor: post.isAnonymous ? '#d9d9d9' : '#ffb6c1' }}>
                            {post.isAnonymous ? <EyeInvisibleOutlined /> : (post.user?.nickname?.[0] || '匿')}
                          </Avatar>
                          <div style={{ marginLeft: 10, flex: 1 }}>
                            <Text style={{ fontSize: 13, color: '#8a7a9a' }}>
                              {post.isAnonymous ? '匿名树洞' : post.user?.nickname}
                            </Text>
                          </div>
                          <Tag color="pink" style={{ fontSize: 11, borderRadius: 20, margin: 0 }}>
                            {categories.find(c => c.key === post.category)?.label || post.category}
                          </Tag>
                        </div>
                        <Title level={5} style={{ marginBottom: 6, color: '#5a4a6a' }}>{post.title}</Title>
                        <Paragraph ellipsis={{ rows: 3 }} style={{ color: '#8a7a9a', fontSize: 14, marginBottom: 8 }}>
                          {post.content}
                        </Paragraph>
                        <Space size="large">
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
                          <Text style={{ fontSize: 11, color: '#ccc' }}>
                            {new Date(post.createdAt).toLocaleDateString('zh-CN')}
                          </Text>
                        </Space>
                      </Card>
                    ))}
                  </div>
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
                <Paragraph style={{ color: '#8a7a9a', maxWidth: 400, margin: '0 auto 16px' }}>
                  说出你不敢说的话。所有内容完全匿名，没有人会知道是你。
                </Paragraph>
                <div style={{
                  padding: '10px 16px', background: '#fff7e6', borderRadius: 8,
                  maxWidth: 400, margin: '0 auto 20px',
                }}>
                  <Text style={{ fontSize: 12, color: '#d46b08' }}>
                    💡 为了保护你的隐私，请不要透露学校、班级、姓名等可识别身份的信息
                  </Text>
                </div>
                <Button
                  className="cloud-btn"
                  size="large"
                  icon={<PlusOutlined />}
                  onClick={() => { setNewPost({ ...newPost, isAnonymous: true, category: 'share' }); setCreateModal(true); }}
                >
                  匿名倾诉
                </Button>
                <div style={{ marginTop: 24, padding: 16, background: '#fff5f7', borderRadius: 16, maxWidth: 400, margin: '24px auto 0' }}>
                  <Text type="secondary" style={{ fontSize: 13 }}>
                    💛 如果你现在很痛苦，可以去「AI 倾诉」找 AI 聊聊，它会一直陪着你。
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

      {/* 帖子详情弹窗 */}
      <Modal
        open={!!postDetail}
        onCancel={() => { setPostDetail(null); setPostComments([]); setComment(''); }}
        footer={null}
        width={600}
        centered
      >
        {postDetail && (
          <div style={{ padding: '8px 0' }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
              <Avatar size={36} style={{ backgroundColor: postDetail.isAnonymous ? '#d9d9d9' : '#ffb6c1' }}>
                {postDetail.isAnonymous ? <EyeInvisibleOutlined /> : (postDetail.user?.nickname?.[0] || '匿')}
              </Avatar>
              <div style={{ marginLeft: 10, flex: 1 }}>
                <Text strong style={{ fontSize: 14, color: '#5a4a6a' }}>
                  {postDetail.isAnonymous ? '匿名树洞' : postDetail.user?.nickname}
                </Text>
                <div><Text type="secondary" style={{ fontSize: 11 }}>
                  {new Date(postDetail.createdAt).toLocaleString('zh-CN')}
                </Text></div>
              </div>
              <Tag color="pink" style={{ borderRadius: 20 }}>
                {categories.find(c => c.key === postDetail.category)?.label || postDetail.category}
              </Tag>
            </div>
            <Title level={4} style={{ color: '#5a4a6a', marginBottom: 8 }}>{postDetail.title}</Title>
            <Paragraph style={{ color: '#5a4a6a', fontSize: 15, lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
              {postDetail.content}
            </Paragraph>
            <Divider />
            <div style={{ marginBottom: 16 }}>
              <Text strong style={{ fontSize: 14, color: '#5a4a6a' }}>
                <MessageOutlined /> 评论 ({postComments.length})
              </Text>
            </div>
            {postComments.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '20px 0' }}>
                <Text type="secondary">还没有评论，来说点什么吧~</Text>
              </div>
            ) : (
              <List
                dataSource={postComments}
                style={{ maxHeight: 300, overflow: 'auto', marginBottom: 16 }}
                renderItem={(c: any) => (
                  <List.Item style={{ padding: '10px 0' }}>
                    <div style={{ width: '100%' }}>
                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
                        <Avatar size={24} style={{ backgroundColor: '#d9d9d9' }}>
                          {c.isAnonymous ? <EyeInvisibleOutlined /> : (c.user?.nickname?.[0] || '匿')}
                        </Avatar>
                        <Text style={{ marginLeft: 8, fontSize: 12, color: '#8a7a9a' }}>
                          {c.isAnonymous ? '匿名' : c.user?.nickname}
                        </Text>
                        <Text style={{ marginLeft: 'auto', fontSize: 11, color: '#ccc' }}>
                          {new Date(c.createdAt).toLocaleString('zh-CN')}
                        </Text>
                      </div>
                      <div style={{ marginLeft: 32, fontSize: 14, color: '#5a4a6a' }}>{c.content}</div>
                    </div>
                  </List.Item>
                )}
              />
            )}
            <div style={{ display: 'flex', gap: 8 }}>
              <Input.TextArea
                rows={2}
                placeholder="写下你的评论..."
                value={comment}
                onChange={e => setComment(e.target.value)}
                style={{ borderRadius: 12, flex: 1 }}
              />
              <Button type="primary" onClick={handleDetailComment} disabled={!comment.trim()}
                style={{ borderRadius: 12, alignSelf: 'flex-end' }}>
                发送
              </Button>
            </div>
          </div>
        )}
      </Modal>

      {/* 评论弹窗（保留兼容） */}
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

/**
 * 24 小时人工心理援助热线 —— 全站唯一一份号码清单。
 *
 * 为什么抽成独立模块：这些号码原先写在 `HomeQuickActions.tsx`（心理电话弹窗）和
 * `Home.tsx`（紧急求助弹窗）两处，改号时极易只改一处。现在首页的「心理电话」弹窗、
 * 页脚提示都从这里引用，`CrisisResourceCard` 等模块若将来也要展示，同样引这里。
 *
 * ⚠️ 改动号码前请确认真实性：这是面向青少年的危机资源，错误号码比没有号码更危险。
 */
export interface Hotline {
  name: string;
  phone: string;
  note: string;
}

export const HOTLINES: Hotline[] = [
  { name: '全国 24 小时心理援助热线', phone: '400-161-9995', note: '24 小时有人接听' },
  { name: '北京心理危机研究与干预中心', phone: '010-82951332', note: '24 小时危机干预' },
  { name: '生命热线', phone: '400-821-1215', note: '24 小时情绪支持' },
];

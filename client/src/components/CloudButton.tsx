import React from 'react';
import { Button, ButtonProps } from 'antd';

interface CloudButtonProps extends Omit<ButtonProps, 'variant'> {
  cloudVariant?: 'primary' | 'secondary';
  cloudSize?: 'small' | 'middle' | 'large';
  children: React.ReactNode;
}

export default function CloudButton({
  cloudVariant = 'primary',
  cloudSize = 'middle',
  children,
  style,
  ...rest
}: CloudButtonProps) {
  const sizeClass = cloudSize === 'small' ? 'cloud-btn-sm' : cloudSize === 'large' ? 'cloud-btn-lg' : '';
  const variantClass = cloudVariant === 'secondary' ? 'cloud-btn-secondary' : '';

  return (
    <Button
      className={`cloud-btn ${sizeClass} ${variantClass}`}
      style={{
        borderRadius: 50,
        ...style,
      }}
      {...rest}
    >
      {children}
    </Button>
  );
}

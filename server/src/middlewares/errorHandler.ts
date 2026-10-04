import { Request, Response, NextFunction } from 'express';

export class AppError extends Error {
  statusCode: number;
  code: number;

  constructor(message: string, statusCode: number = 500) {
    super(message);
    this.statusCode = statusCode;
    this.code = statusCode;
    Error.captureStackTrace(this, this.constructor);
  }
}

export const errorHandler = (err: Error, _req: Request, res: Response, _next: NextFunction): void => {
  if (err instanceof AppError) {
    res.status(err.statusCode).json({
      code: err.code,
      message: err.message,
    });
    return;
  }

  console.error('Unhandled error:', err);
  res.status(500).json({
    code: 500,
    message: '服务器内部错误',
  });
};

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import simdLoader from '@mediapipe/tasks-vision/vision_wasm_internal.js?url';
import simdBinary from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url';
import scalarLoader from '@mediapipe/tasks-vision/vision_wasm_nosimd_internal.js?url';
import scalarBinary from '@mediapipe/tasks-vision/vision_wasm_nosimd_internal.wasm?url';

/** 与已安装 JS 包同版的本地运行时；构建时由 Vite 一起打包。 */
export async function createFaceLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks();
  const scalar = fileset.wasmLoaderPath.includes('nosimd');
  fileset.wasmLoaderPath = scalar ? scalarLoader : simdLoader;
  fileset.wasmBinaryPath = scalar ? scalarBinary : simdBinary;
  const options = {
    modelAssetPath: `${import.meta.env.BASE_URL}models/face_landmarker.task`,
  };
  try {
    return await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { ...options, delegate: 'GPU' },
      runningMode: 'VIDEO', numFaces: 1,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
    });
  } catch (error) {
    console.warn('[面部分析] GPU 初始化不可用，尝试本地 CPU 检测器:', error);
    return FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { ...options, delegate: 'CPU' },
      runningMode: 'VIDEO', numFaces: 1,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
    });
  }
}

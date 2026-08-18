#![cfg(feature = "cuda")]

use std::ffi::c_void;
use crate::error::{BitNetError, Result};
use crate::quantization::PackedMatrix;
use crate::ops::linear::LinearWeight;

extern "C" {
    fn launch_bitgemv_1bit(
        W_packed: *const u64,
        x_packed: *const u64,
        y:        *mut   f32,
        w_scale:  f32,
        x_scale:  f32,
        N:        i32,
        K_words:  i32,
        K:        i32,
        stream:   *mut c_void,
    );

    fn launch_bitgemv_ternary(
        W_mag:    *const u64,
        W_sign:   *const u64,
        x_packed: *const u64,
        y:        *mut   f32,
        w_scale:  f32,
        x_scale:  f32,
        N:        i32,
        K_words:  i32,
        stream:   *mut c_void,
    );

    fn launch_rmsnorm_fp16(
        x:      *const u16,
        y:      *mut   u16,
        w:      *const u16,
        rows:   i32,
        dim:    i32,
        stream: *mut c_void,
    );
}

pub fn is_available() -> bool {
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub struct GpuBuffer {
    ptr:  *mut c_void,
    size: usize,
}

extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: i32) -> i32;
}

const MEMCPY_H2D: i32 = 1;
const MEMCPY_D2H: i32 = 2;

impl GpuBuffer {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            let err = cudaMalloc(&mut ptr, bytes);
            if err != 0 {
                return Err(BitNetError::Cuda(
                    format!("cudaMalloc({} bytes) failed with code {}", bytes, err)
                ));
            }
        }
        Ok(Self { ptr, size: bytes })
    }

    pub fn upload(&self, data: &[u8]) -> Result<()> {
        assert!(data.len() <= self.size);
        unsafe {
            let err = cudaMemcpy(self.ptr, data.as_ptr() as _, data.len(), MEMCPY_H2D);
            if err != 0 {
                return Err(BitNetError::Cuda(
                    format!("cudaMemcpy H2D failed: {}", err)
                ));
            }
        }
        Ok(())
    }

    pub fn download(&self, dst: &mut [u8]) -> Result<()> {
        assert!(dst.len() <= self.size);
        unsafe {
            let err = cudaMemcpy(dst.as_mut_ptr() as _, self.ptr, dst.len(), MEMCPY_D2H);
            if err != 0 {
                return Err(BitNetError::Cuda(
                    format!("cudaMemcpy D2H failed: {}", err)
                ));
            }
        }
        Ok(())
    }

    pub fn as_ptr<T>(&self) -> *const T { self.ptr as _ }
    pub fn as_mut_ptr<T>(&self) -> *mut T { self.ptr as _ }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr); }
    }
}

pub fn linear_forward_cuda(
    lw:       &LinearWeight,
    x_packed: &[u64],
    x_scale:  f32,
    mut y:    Vec<f32>,
) -> Result<Vec<f32>> {
    use crate::quantization::QuantMode;

    let w       = &lw.weight;
    let n       = w.rows as i32;
    let k_words = w.words_per_row as i32;
    // Real (unpadded) row width. bitgemv_1bit needs this in addition to
    // k_words to correct for the XNOR-padding contamination described in
    // bitgemv.cu — the last packed word may contain unused bits beyond
    // the true column count, and those are otherwise miscounted as
    // "agreements" by the popcount/XNOR formula.
    let k       = w.cols as i32;
    let stream  = std::ptr::null_mut();

    let gpu_mag = GpuBuffer::alloc(w.mag_bytes())?;
    gpu_mag.upload(bytemuck::cast_slice(&w.mag))?;

    let gpu_x = GpuBuffer::alloc(x_packed.len() * 8)?;
    gpu_x.upload(bytemuck::cast_slice(x_packed))?;

    let gpu_y = GpuBuffer::alloc(n as usize * 4)?;
    gpu_y.upload(bytemuck::cast_slice(&y))?;

    unsafe {
        match w.mode {
            QuantMode::Binary => {
                launch_bitgemv_1bit(
                    gpu_mag.as_ptr(), gpu_x.as_ptr(), gpu_y.as_mut_ptr(),
                    w.scale, x_scale, n, k_words, k, stream,
                );
            }
            QuantMode::Ternary => {
                let gpu_sign = GpuBuffer::alloc(w.sign_bytes())?;
                gpu_sign.upload(bytemuck::cast_slice(&w.sign))?;
                launch_bitgemv_ternary(
                    gpu_mag.as_ptr(), gpu_sign.as_ptr(), gpu_x.as_ptr(),
                    gpu_y.as_mut_ptr(), w.scale, x_scale, n, k_words, stream,
                );
            }
        }
    }

    let mut y_bytes = vec![0u8; n as usize * 4];
    gpu_y.download(&mut y_bytes)?;
    y = bytemuck::cast_slice(&y_bytes).to_vec();

    Ok(y)
                    }


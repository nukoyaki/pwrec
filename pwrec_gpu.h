#pragma once
#include <stdint.h>

#include <libavcodec/avcodec.h>
#include <libavutil/dict.h>

struct gpu_buf;

typedef void (*gpu_emit_fn)(AVPacket *pkt);

int gpu_init(void);
void gpu_destroy(void);

struct gpu_buf *gpu_import(int fd, uint32_t size);
void gpu_release(struct gpu_buf *b);

AVCodecContext *gpu_encoder_open(const char *codec_name, int width, int height,
				 enum AVPixelFormat sw_fmt, AVRational time_base,
				 AVRational framerate, AVDictionary **opts);

int gpu_encode(struct gpu_buf *b, uint32_t offset, uint32_t stride, int64_t pts, gpu_emit_fn emit);

void gpu_flush(gpu_emit_fn emit);

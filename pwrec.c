#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <pipewire/pipewire.h>
#include <spa/buffer/meta.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/video/format-utils.h>
#include <spa/pod/builder.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libavutil/parseutils.h>
#include <libavutil/pixdesc.h>

extern char **environ;

#define VIDEO_TB_DEN 1000000

static struct {
	const char *video_target;
	const char *audio_target;
	const char *output;
	const char *ffmpeg;
	int no_audio;
	int cfr_num, cfr_den;
	AVRational nominal_fps;
	int rate, channels;
	long queue_mb;
	double audio_offset_ms;
	int verbose;
	char **enc_args;
	int n_enc_args;
} opt = {
	.ffmpeg = "ffmpeg",
	.rate = 48000,
	.channels = 2,
	.queue_mb = 4096,
	.nominal_fps = { 60, 1 },
};

static const char *default_enc_args[] = {
	"-c:v", "hevc_nvenc", "-b:v", "30M", "-pix_fmt", "yuv444p",
	"-vf", "scale=out_color_matrix=bt709:out_range=tv,"
	       "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
	"-c:a", "flac", "-compression_level", "12",
	"-af", "aresample=async=1:first_pts=0",
	NULL,
};

struct pkt {
	struct pkt *next;
	int stream;
	int64_t pts;
	size_t size;
	uint8_t data[];
};

static struct {
	pthread_mutex_t lock;
	pthread_cond_t cond_nonempty, cond_space;
	struct pkt *head, *tail;
	size_t video_bytes, max_video_bytes;
	int video_count, peak_video_count;
	int eof;
} q = {
	.lock = PTHREAD_MUTEX_INITIALIZER,
	.cond_nonempty = PTHREAD_COND_INITIALIZER,
	.cond_space = PTHREAD_COND_INITIALIZER,
};

static int q_push(struct pkt *p)
{
	pthread_mutex_lock(&q.lock);
	if (p->stream == 0) {
		while (!q.eof && q.video_count > 0 &&
		       q.video_bytes + p->size > q.max_video_bytes)
			pthread_cond_wait(&q.cond_space, &q.lock);
		q.video_bytes += p->size;
		q.video_count++;
		if (q.video_count > q.peak_video_count)
			q.peak_video_count = q.video_count;
	}
	if (q.eof) {
		pthread_mutex_unlock(&q.lock);
		free(p);
		return -1;
	}
	p->next = NULL;
	if (q.tail)
		q.tail->next = p;
	else
		q.head = p;
	q.tail = p;
	pthread_cond_signal(&q.cond_nonempty);
	pthread_mutex_unlock(&q.lock);
	return 0;
}

static struct pkt *q_pop(void)
{
	pthread_mutex_lock(&q.lock);
	while (!q.head && !q.eof)
		pthread_cond_wait(&q.cond_nonempty, &q.lock);
	struct pkt *p = q.head;
	if (p) {
		q.head = p->next;
		if (!q.head)
			q.tail = NULL;
		if (p->stream == 0) {
			q.video_bytes -= p->size;
			q.video_count--;
			pthread_cond_signal(&q.cond_space);
		}
	}
	pthread_mutex_unlock(&q.lock);
	return p;
}

static void q_close(void)
{
	pthread_mutex_lock(&q.lock);
	q.eof = 1;
	pthread_cond_broadcast(&q.cond_nonempty);
	pthread_cond_broadcast(&q.cond_space);
	pthread_mutex_unlock(&q.lock);
}


static atomic_int quit;
static pthread_mutex_t sync_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t sync_cond = PTHREAD_COND_INITIALIZER;

static struct {
	int have_format;
	enum AVPixelFormat pix_fmt;
	int width, height, bpp;
	uint32_t spa_format;

	int audio_ready;
	int started;
	int64_t t0_mono;
	int64_t t0_video;
	int using_hdr_pts;

	uint64_t frames, frames_waiting, format_changes;
	int64_t last_vts, min_dt, max_dt;
	int64_t first_vts_mono, last_vts_mono;
	uint64_t audio_samples, audio_resyncs;
	int64_t audio_pos;
	int audio_started;
	double audio_drift_ms;
} st = { .min_dt = INT64_MAX };

static int parse_rate(AVRational *r, const char *str)
{
	if (av_parse_video_rate(r, str) < 0 || r->num <= 0 || r->den <= 0)
		return -1;
	double v = av_q2d(*r);
	double ntsc = v * 1.001;
	if (r->den != 1 && fabs(ntsc - llround(ntsc)) < 0.01 &&
	    fabs(v - round(v)) > 0.01)
		*r = (AVRational){ (int)llround(ntsc) * 1000, 1001 };
	return 0;
}

static int64_t mono_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct stream_data {
	struct pw_thread_loop *loop;
	struct pw_context *context;
	struct pw_core *core;
	struct pw_stream *stream;
	struct spa_hook listener;
};

static struct stream_data vid, aud;

static void video_param_changed(void *data, uint32_t id, const struct spa_pod *param)
{
	if (param == NULL || id != SPA_PARAM_Format)
		return;

	struct spa_video_info_raw info;
	uint32_t mtype, msub;
	if (spa_format_parse(param, &mtype, &msub) < 0 ||
	    mtype != SPA_MEDIA_TYPE_video || msub != SPA_MEDIA_SUBTYPE_raw)
		return;
	if (spa_format_video_raw_parse(param, &info) < 0)
		return;

	enum AVPixelFormat pf;
	switch (info.format) {
	case SPA_VIDEO_FORMAT_BGRA: pf = AV_PIX_FMT_BGRA; break;
	case SPA_VIDEO_FORMAT_BGRx: pf = AV_PIX_FMT_BGR0; break;
	case SPA_VIDEO_FORMAT_RGBA: pf = AV_PIX_FMT_RGBA; break;
	case SPA_VIDEO_FORMAT_RGBx: pf = AV_PIX_FMT_RGB0; break;
	default:
		fprintf(stderr, "pwrec: unsupported video format %u\n", info.format);
		return;
	}

	pthread_mutex_lock(&sync_lock);
	if (st.have_format) {
		if (info.size.width != (uint32_t)st.width ||
		    info.size.height != (uint32_t)st.height ||
		    info.format != st.spa_format) {
			st.format_changes++;
			fprintf(stderr, "\npwrec: video format changed mid-recording "
				"(%ux%u) - stopping\n", info.size.width, info.size.height);
			atomic_store(&quit, 1);
		}
	} else {
		st.width = info.size.width;
		st.height = info.size.height;
		st.pix_fmt = pf;
		st.spa_format = info.format;
		st.bpp = 4;
		st.have_format = 1;
		fprintf(stderr, "pwrec: video %dx%d %s, producer max %u/%u fps\n",
			st.width, st.height, av_get_pix_fmt_name(pf),
			info.max_framerate.num, info.max_framerate.denom);
	}
	pthread_cond_broadcast(&sync_cond);
	pthread_mutex_unlock(&sync_lock);

	uint8_t buf[256];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
	const struct spa_pod *params[1];
	params[0] = spa_pod_builder_add_object(&b,
		SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
		SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
		SPA_PARAM_META_size, SPA_POD_Int(sizeof(struct spa_meta_header)));
	pw_stream_update_params(vid.stream, params, 1);
}

static void video_process(void *data)
{
	struct pw_buffer *b = pw_stream_dequeue_buffer(vid.stream);
	if (!b)
		return;
	struct spa_buffer *buf = b->buffer;
	struct spa_data *d = &buf->datas[0];
	int64_t now_mono = mono_ns();

	if (!st.have_format || !d->data || !d->chunk ||
	    (d->chunk->flags & SPA_CHUNK_FLAG_CORRUPTED) || d->chunk->size == 0)
		goto done;

	struct spa_meta_header *h = spa_buffer_find_meta_data(buf, SPA_META_Header, sizeof(*h));
	if (h && (h->flags & SPA_META_HEADER_FLAG_CORRUPTED))
		goto done;

	int64_t vts = -1;
	int hdr = 0;
	if (h && h->pts > 0) {
		vts = h->pts;
		hdr = 1;
	} else {
		struct pw_time t;
		if (pw_stream_get_time_n(vid.stream, &t, sizeof(t)) == 0 && t.now > 0 &&
		    llabs(now_mono - t.now) < 50000000LL)
			vts = t.now;
		else
			vts = now_mono;
	}

	pthread_mutex_lock(&sync_lock);
	if (!st.started) {
		if (!st.audio_ready && !opt.no_audio) {
			st.frames_waiting++;
			pthread_mutex_unlock(&sync_lock);
			goto done;
		}
		st.started = 1;
		st.using_hdr_pts = hdr;
		st.t0_video = vts;
		st.t0_mono = hdr ? now_mono : vts;
		st.last_vts = vts;
		st.first_vts_mono = now_mono;
		fprintf(stderr, "pwrec: recording (timestamps from %s)\n",
			opt.cfr_num ? "frame count (CFR)" :
			hdr ? "producer pts" : "producer push time");
	} else {
		int64_t dt = vts - st.last_vts;
		if (dt <= 0) {
			dt = 0;
			vts = st.last_vts + 1000;
		}
		if (dt < st.min_dt) st.min_dt = dt;
		if (dt > st.max_dt) st.max_dt = dt;
		st.last_vts = vts;
	}
	st.last_vts_mono = now_mono;
	uint64_t index = st.frames++;
	pthread_mutex_unlock(&sync_lock);

	int64_t pts = opt.cfr_num ? (int64_t)index
		: av_rescale(vts - st.t0_video, VIDEO_TB_DEN, 1000000000LL);

	size_t row = (size_t)st.width * st.bpp;
	size_t size = row * st.height;
	uint32_t stride = d->chunk->stride ? (uint32_t)d->chunk->stride : row;
	uint32_t off = d->chunk->offset % d->maxsize;
	if ((uint64_t)off + (uint64_t)stride * (st.height - 1) + row > d->maxsize) {
		fprintf(stderr, "pwrec: short video buffer, skipping\n");
		goto done;
	}

	struct pkt *p = malloc(sizeof(*p) + size);
	if (!p)
		goto done;
	p->stream = 0;
	p->pts = pts;
	p->size = size;
	const uint8_t *src = (const uint8_t *)d->data + off;
	if (stride == row) {
		memcpy(p->data, src, size);
	} else {
		for (int y = 0; y < st.height; y++)
			memcpy(p->data + y * row, src + (size_t)y * stride, row);
	}

	q_push(p);

done:
	pw_stream_queue_buffer(vid.stream, b);
}

static const struct pw_stream_events video_events = {
	PW_VERSION_STREAM_EVENTS,
	.param_changed = video_param_changed,
	.process = video_process,
};

static void audio_process(void *data)
{
	struct pw_buffer *b = pw_stream_dequeue_buffer(aud.stream);
	if (!b)
		return;
	struct spa_data *d = &b->buffer->datas[0];
	if (!d->data || !d->chunk || d->chunk->size == 0)
		goto done;

	uint32_t frame_size = sizeof(int32_t) * opt.channels;
	uint32_t off = d->chunk->offset % d->maxsize;
	uint32_t size = SPA_MIN(d->chunk->size, d->maxsize - off);
	uint32_t n = size / frame_size;
	if (n == 0)
		goto done;

	struct pw_time t;
	int64_t ta = (pw_stream_get_time_n(aud.stream, &t, sizeof(t)) == 0 && t.now > 0)
		? t.now : mono_ns();

	pthread_mutex_lock(&sync_lock);
	if (!st.audio_ready) {
		st.audio_ready = 1;
		pthread_cond_broadcast(&sync_cond);
	}
	if (!st.started) {
		pthread_mutex_unlock(&sync_lock);
		goto done;
	}

	int64_t ideal = av_rescale(ta - st.t0_mono, opt.rate, 1000000000LL)
		+ llround(opt.audio_offset_ms * opt.rate / 1000.0);
	uint32_t skip = 0;
	if (!st.audio_started) {
		st.audio_started = 1;
		st.audio_pos = ideal;
	} else {
		int64_t drift = ideal - st.audio_pos;
		st.audio_drift_ms = drift * 1000.0 / opt.rate;
		if (llabs(drift) > opt.rate / 10) {
			st.audio_resyncs++;
			st.audio_pos = ideal;
		}
	}
	if (st.audio_pos < 0) {
		int64_t s = -st.audio_pos;
		if (s >= n) {
			st.audio_pos += n;
			pthread_mutex_unlock(&sync_lock);
			goto done;
		}
		skip = s;
		st.audio_pos = 0;
	}
	int64_t pts = st.audio_pos;
	st.audio_pos += n - skip;
	st.audio_samples += n - skip;
	pthread_mutex_unlock(&sync_lock);

	size_t bytes = (size_t)(n - skip) * frame_size;
	struct pkt *p = malloc(sizeof(*p) + bytes);
	if (p) {
		p->stream = 1;
		p->pts = pts;
		p->size = bytes;
		memcpy(p->data, (uint8_t *)d->data + off + skip * frame_size, bytes);
		q_push(p);
	}
done:
	pw_stream_queue_buffer(aud.stream, b);
}

static const struct pw_stream_events audio_events = {
	PW_VERSION_STREAM_EVENTS,
	.process = audio_process,
};

static void stream_state_changed(void *data, enum pw_stream_state old,
				 enum pw_stream_state state, const char *error)
{
	const char *name = data;
	if (opt.verbose || state == PW_STREAM_STATE_ERROR)
		fprintf(stderr, "pwrec: %s stream: %s%s%s\n", name,
			pw_stream_state_as_string(state), error ? " - " : "", error ? error : "");
	if (state == PW_STREAM_STATE_ERROR ||
	    (state == PW_STREAM_STATE_UNCONNECTED && old != PW_STREAM_STATE_UNCONNECTED)) {
		fprintf(stderr, "\npwrec: %s stream disconnected - stopping\n", name);
		atomic_store(&quit, 1);
	}
}

static const struct pw_stream_events video_state_events = {
	PW_VERSION_STREAM_EVENTS, .state_changed = stream_state_changed,
};
static const struct pw_stream_events audio_state_events = {
	PW_VERSION_STREAM_EVENTS, .state_changed = stream_state_changed,
};
static struct spa_hook vid_state_listener, aud_state_listener;

static int setup_loop(struct stream_data *s, const char *name)
{
	s->loop = pw_thread_loop_new(name, NULL);
	s->context = pw_context_new(pw_thread_loop_get_loop(s->loop), NULL, 0);
	if (!s->context)
		return -1;
	if (pw_thread_loop_start(s->loop) < 0)
		return -1;
	pw_thread_loop_lock(s->loop);
	s->core = pw_context_connect(s->context, NULL, 0);
	pw_thread_loop_unlock(s->loop);
	return s->core ? 0 : -1;
}

static int connect_video(void)
{
	if (setup_loop(&vid, "pwrec-video") < 0)
		return -1;
	pw_thread_loop_lock(vid.loop);
	struct pw_properties *props = pw_properties_new(
		PW_KEY_MEDIA_TYPE, "Video",
		PW_KEY_MEDIA_CATEGORY, "Capture",
		PW_KEY_MEDIA_ROLE, "Production",
		PW_KEY_NODE_NAME, "pwrec-video",
		PW_KEY_TARGET_OBJECT, opt.video_target,
		PW_KEY_NODE_DONT_RECONNECT, "true",
		NULL);
	vid.stream = pw_stream_new(vid.core, "pwrec video", props);
	pw_stream_add_listener(vid.stream, &vid.listener, &video_events, NULL);
	pw_stream_add_listener(vid.stream, &vid_state_listener, &video_state_events, "video");

	uint8_t buf[1024];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
	const struct spa_pod *params[1];
	params[0] = spa_pod_builder_add_object(&b,
		SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
		SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format, SPA_POD_CHOICE_ENUM_Id(5,
			SPA_VIDEO_FORMAT_BGRA, SPA_VIDEO_FORMAT_BGRA, SPA_VIDEO_FORMAT_BGRx,
			SPA_VIDEO_FORMAT_RGBA, SPA_VIDEO_FORMAT_RGBx),
		SPA_FORMAT_VIDEO_size, SPA_POD_CHOICE_RANGE_Rectangle(
			&SPA_RECTANGLE(1920, 1080), &SPA_RECTANGLE(1, 1), &SPA_RECTANGLE(16384, 16384)),
		SPA_FORMAT_VIDEO_framerate, SPA_POD_CHOICE_RANGE_Fraction(
			&SPA_FRACTION(0, 1), &SPA_FRACTION(0, 1), &SPA_FRACTION(1000, 1)));

	int r = pw_stream_connect(vid.stream, PW_DIRECTION_INPUT, PW_ID_ANY,
		PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS, params, 1);
	pw_thread_loop_unlock(vid.loop);
	return r;
}

static int connect_audio(void)
{
	if (setup_loop(&aud, "pwrec-audio") < 0)
		return -1;
	pw_thread_loop_lock(aud.loop);
	struct pw_properties *props = pw_properties_new(
		PW_KEY_MEDIA_TYPE, "Audio",
		PW_KEY_MEDIA_CATEGORY, "Capture",
		PW_KEY_MEDIA_ROLE, "Production",
		PW_KEY_NODE_NAME, "pwrec-audio",
		PW_KEY_STREAM_CAPTURE_SINK, "true",
		NULL);
	if (opt.audio_target)
		pw_properties_set(props, PW_KEY_TARGET_OBJECT, opt.audio_target);
	aud.stream = pw_stream_new(aud.core, "pwrec audio", props);
	pw_stream_add_listener(aud.stream, &aud.listener, &audio_events, NULL);
	pw_stream_add_listener(aud.stream, &aud_state_listener, &audio_state_events, "audio");

	uint8_t buf[1024];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
	struct spa_audio_info_raw info = {
		.format = SPA_AUDIO_FORMAT_S32,
		.rate = opt.rate,
		.channels = opt.channels,
	};
	if (opt.channels == 2) {
		info.position[0] = SPA_AUDIO_CHANNEL_FL;
		info.position[1] = SPA_AUDIO_CHANNEL_FR;
	} else {
		info.flags = SPA_AUDIO_FLAG_UNPOSITIONED;
	}
	const struct spa_pod *params[1];
	params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);

	int r = pw_stream_connect(aud.stream, PW_DIRECTION_INPUT, PW_ID_ANY,
		PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS, params, 1);
	pw_thread_loop_unlock(aud.loop);
	return r;
}

static void destroy_stream(struct stream_data *s)
{
	if (!s->loop)
		return;
	pw_thread_loop_lock(s->loop);
	if (s->stream)
		pw_stream_destroy(s->stream);
	if (s->core)
		pw_core_disconnect(s->core);
	pw_thread_loop_unlock(s->loop);
	pw_thread_loop_stop(s->loop);
	if (s->context)
		pw_context_destroy(s->context);
	pw_thread_loop_destroy(s->loop);
}

static AVFormatContext *oc;
static AVRational video_src_tb;
static int ffmpeg_fd = -1;
static AVStream *vst, *ast;
static int mux_error;

static int open_muxer(int fd)
{
	char url[32];
	snprintf(url, sizeof(url), "pipe:%d", fd);
	if (avformat_alloc_output_context2(&oc, NULL, "nut", url) < 0)
		return -1;

	vst = avformat_new_stream(oc, NULL);
	vst->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
	vst->codecpar->codec_id = AV_CODEC_ID_RAWVIDEO;
	vst->codecpar->format = st.pix_fmt;
	vst->codecpar->codec_tag = avcodec_pix_fmt_to_codec_tag(st.pix_fmt);
	vst->codecpar->bits_per_coded_sample = 32;
	vst->codecpar->width = st.width;
	vst->codecpar->height = st.height;
	vst->codecpar->color_range = AVCOL_RANGE_JPEG;
	vst->codecpar->color_space = AVCOL_SPC_RGB;
	vst->codecpar->color_trc = AVCOL_TRC_IEC61966_2_1;
	vst->codecpar->color_primaries = AVCOL_PRI_BT709;
	if (opt.cfr_num) {
		vst->time_base = video_src_tb = (AVRational){ opt.cfr_den, opt.cfr_num };
		vst->avg_frame_rate = vst->r_frame_rate = (AVRational){ opt.cfr_num, opt.cfr_den };
	} else {
		vst->time_base = video_src_tb = (AVRational){ 1, VIDEO_TB_DEN };
		vst->avg_frame_rate = vst->r_frame_rate = opt.nominal_fps;
	}

	if (!opt.no_audio) {
		ast = avformat_new_stream(oc, NULL);
		ast->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
		ast->codecpar->codec_id = AV_CODEC_ID_PCM_S32LE;
		ast->codecpar->sample_rate = opt.rate;
		av_channel_layout_default(&ast->codecpar->ch_layout, opt.channels);
		ast->codecpar->bits_per_coded_sample = 32;
		ast->codecpar->bits_per_raw_sample = 32;
		ast->time_base = (AVRational){ 1, opt.rate };
	}

	if (avio_open(&oc->pb, url, AVIO_FLAG_WRITE) < 0)
		return -1;
	if (avformat_write_header(oc, NULL) < 0)
		return -1;
	return 0;
}

static void *mux_thread(void *arg)
{
	AVPacket *pkt = av_packet_alloc();
	struct pkt *p;
	while ((p = q_pop())) {
		if (!mux_error) {
			AVStream *s = p->stream == 0 ? vst : ast;
			AVRational src_tb = p->stream == 0 ? video_src_tb : (AVRational){ 1, opt.rate };
			pkt->data = p->data;
			pkt->size = p->size;
			pkt->stream_index = s->index;
			pkt->pts = pkt->dts = av_rescale_q(p->pts, src_tb, s->time_base);
			pkt->duration = p->stream == 1 ? (int64_t)(p->size / (4 * opt.channels)) : 0;
			pkt->flags = AV_PKT_FLAG_KEY;
			int r = av_write_frame(oc, pkt);
			if (r < 0) {
				fprintf(stderr, "\npwrec: write to ffmpeg failed: %s\n", av_err2str(r));
				mux_error = 1;
				atomic_store(&quit, 1);
			}
			av_packet_unref(pkt);
		}
		free(p);
	}
	av_packet_free(&pkt);
	return NULL;
}

static pid_t spawn_ffmpeg(int *write_fd)
{
	int fds[2];
	if (pipe2(fds, O_CLOEXEC) < 0)
		return -1;
	fcntl(fds[1], F_SETPIPE_SZ, 1 << 20);

	const char *argv[128];
	int n = 0;
	argv[n++] = opt.ffmpeg;
	argv[n++] = "-hide_banner";
	argv[n++] = "-nostdin";
	argv[n++] = "-loglevel"; argv[n++] = opt.verbose ? "info" : "warning";
	argv[n++] = "-stats";
	argv[n++] = "-f"; argv[n++] = "nut";
	argv[n++] = "-i"; argv[n++] = "pipe:0";
	for (int i = 0; i < opt.n_enc_args && n < 110; i++)
		argv[n++] = opt.enc_args[i];
	argv[n++] = "-fps_mode"; argv[n++] = "passthrough";
	argv[n++] = "-enc_time_base:v"; argv[n++] = "demux";
	argv[n++] = "-y";
	argv[n++] = opt.output;
	argv[n] = NULL;

	if (opt.verbose) {
		fprintf(stderr, "pwrec: running:");
		for (int i = 0; argv[i]; i++)
			fprintf(stderr, " %s", argv[i]);
		fprintf(stderr, "\n");
	}

	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, fds[0], 0);
	posix_spawnattr_t attr;
	posix_spawnattr_init(&attr);
	posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
	posix_spawnattr_setpgroup(&attr, 0);

	pid_t pid;
	int r = posix_spawnp(&pid, opt.ffmpeg, &fa, &attr, (char **)argv, environ);
	posix_spawn_file_actions_destroy(&fa);
	posix_spawnattr_destroy(&attr);
	close(fds[0]);
	if (r != 0) {
		fprintf(stderr, "pwrec: cannot run %s: %s\n", opt.ffmpeg, strerror(r));
		close(fds[1]);
		return -1;
	}
	*write_fd = fds[1];
	return pid;
}

static void on_signal(int sig) { atomic_store(&quit, 1); }

static void usage(FILE *f)
{
	fprintf(f,
"usage: pwrec [options] VIDEO_NODE OUTPUT [-- ffmpeg output args...]\n"
"\n"
"records the PipeWire video node VIDEO_NODE (node.name) plus\n"
"the default sink's monitor for audio, encoded by ffmpeg into OUTPUT. Every frame the\n"
"producer pushes becomes exactly one output frame, timestamped by the producer.\n"
"\n"
"  -a, --audio TARGET     capture this node instead of the default sink monitor\n"
"  -n, --no-audio         video only\n"
"  -c, --cfr FPS          force a constant frame rate: frame n gets pts n/FPS,\n"
"                         e.g. the engine's fps cap (60, 59.94 = 60000/1001).\n"
"                         default is VFR using producer timing.\n"
"  -r, --rate HZ          audio sample rate (default 48000)\n"
"      --channels N       audio channels (default 2)\n"
"  -o, --audio-offset MS  shift audio relative to video (default 0)\n"
"  -q, --queue-mb MB      max raw video buffered before back-pressure (4096)\n"
"      --nominal-fps FPS  frame rate the encoder's rate control assumes in VFR\n"
"                         mode (default 60); real timestamps are kept\n"
"      --ffmpeg PATH      ffmpeg binary (default: ffmpeg)\n"
"  -v, --verbose          print the ffmpeg command line and more\n"
"\n"
"any args after -- replace the default encoder args, which are:\n"
"  ");
	for (int i = 0; default_enc_args[i]; i++)
		fprintf(f, "%s ", default_enc_args[i]);
	fprintf(f, "\n(-fps_mode passthrough -enc_time_base:v demux are always added.)\n");
}

int main(int argc, char **argv)
{
	static const struct option lo[] = {
		{ "audio", required_argument, 0, 'a' },
		{ "no-audio", no_argument, 0, 'n' },
		{ "cfr", required_argument, 0, 'c' },
		{ "rate", required_argument, 0, 'r' },
		{ "channels", required_argument, 0, 'C' },
		{ "audio-offset", required_argument, 0, 'o' },
		{ "queue-mb", required_argument, 0, 'q' },
		{ "nominal-fps", required_argument, 0, 'N' },
		{ "ffmpeg", required_argument, 0, 'F' },
		{ "verbose", no_argument, 0, 'v' },
		{ "help", no_argument, 0, 'h' },
		{ 0 },
	};
	int c;
	while ((c = getopt_long(argc, argv, "+a:nc:r:o:q:vh", lo, NULL)) != -1) {
		switch (c) {
		case 'a': opt.audio_target = optarg; break;
		case 'n': opt.no_audio = 1; break;
		case 'c': {
			AVRational r;
			if (parse_rate(&r, optarg) < 0) {
				fprintf(stderr, "pwrec: bad --cfr value '%s'\n", optarg);
				return 2;
			}
			opt.cfr_num = r.num; opt.cfr_den = r.den;
			break;
		}
		case 'r': opt.rate = atoi(optarg); break;
		case 'C': opt.channels = atoi(optarg); break;
		case 'o': opt.audio_offset_ms = atof(optarg); break;
		case 'q': opt.queue_mb = atol(optarg); break;
		case 'N': {
			AVRational r;
			if (parse_rate(&r, optarg) < 0) {
				fprintf(stderr, "pwrec: bad --nominal-fps value '%s'\n", optarg);
				return 2;
			}
			opt.nominal_fps = r;
			break;
		}
		case 'F': opt.ffmpeg = optarg; break;
		case 'v': opt.verbose = 1; break;
		case 'h': usage(stdout); return 0;
		default: usage(stderr); return 2;
		}
	}
	if (argc - optind < 2) {
		usage(stderr);
		return 2;
	}
	opt.video_target = argv[optind++];
	opt.output = argv[optind++];
	if (optind < argc && strcmp(argv[optind], "--") == 0)
		optind++;
	if (optind < argc) {
		opt.enc_args = &argv[optind];
		opt.n_enc_args = argc - optind;
	} else {
		opt.enc_args = (char **)default_enc_args;
		while (default_enc_args[opt.n_enc_args])
			opt.n_enc_args++;
		if (opt.no_audio)
			opt.n_enc_args -= 6;
	}
	if (opt.rate <= 0 || opt.channels <= 0 || opt.channels > 64 || opt.queue_mb <= 0) {
		fprintf(stderr, "pwrec: bad option value\n");
		return 2;
	}
	q.max_video_bytes = (size_t)opt.queue_mb << 20;

	signal(SIGPIPE, SIG_IGN);
	struct sigaction sa = { .sa_handler = on_signal };
	sigaction(SIGINT, &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);

	pw_init(&argc, &argv);

	if (connect_video() < 0 || (!opt.no_audio && connect_audio() < 0)) {
		fprintf(stderr, "pwrec: failed to connect to PipeWire\n");
		return 1;
	}

	fprintf(stderr, "pwrec: waiting for '%s'...\n", opt.video_target);
	pthread_mutex_lock(&sync_lock);
	while (!atomic_load(&quit) && (!st.have_format || (!opt.no_audio && !st.audio_ready))) {
		struct timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		ts.tv_nsec += 200000000;
		if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
		pthread_cond_timedwait(&sync_cond, &sync_lock, &ts);
	}
	pthread_mutex_unlock(&sync_lock);

	int ret = 0;
	pid_t pid = -1;
	pthread_t mt;
	int mux_running = 0;
	if (atomic_load(&quit)) {
		ret = 1;
		goto out;
	}

	pid = spawn_ffmpeg(&ffmpeg_fd);
	if (pid < 0 || open_muxer(ffmpeg_fd) < 0) {
		fprintf(stderr, "pwrec: failed to start ffmpeg/muxer\n");
		ret = 1;
		atomic_store(&quit, 1);
		goto out;
	}
	pthread_create(&mt, NULL, mux_thread, NULL);
	mux_running = 1;

	while (!atomic_load(&quit)) {
		usleep(100000);
		int status;
		if (waitpid(pid, &status, WNOHANG) == pid) {
			fprintf(stderr, "\npwrec: ffmpeg exited early\n");
			pid = -1;
			ret = 1;
			break;
		}
		if (opt.verbose) {
			pthread_mutex_lock(&sync_lock);
			double secs = (st.last_vts_mono - st.first_vts_mono) / 1e9;
			fprintf(stderr, "\rpwrec: %llu frames, %.2f fps avg, dt %.2f..%.2f ms, "
				"queue %d, audio drift %+.1f ms   ",
				(unsigned long long)st.frames,
				secs > 0 ? (st.frames - 1) / secs : 0.0,
				st.min_dt == INT64_MAX ? 0 : st.min_dt / 1e6, st.max_dt / 1e6,
				q.video_count, st.audio_drift_ms);
			pthread_mutex_unlock(&sync_lock);
		}
	}

out:
	fprintf(stderr, "\npwrec: stopping...\n");
	destroy_stream(&vid);
	destroy_stream(&aud);
	if (mux_running) {
		pthread_mutex_lock(&q.lock);
		q.eof = 1;
		pthread_cond_broadcast(&q.cond_nonempty);
		pthread_mutex_unlock(&q.lock);
		pthread_join(mt, NULL);
	} else {
		q_close();
	}
	if (oc) {
		if (!mux_error)
			av_write_trailer(oc);
		avio_closep(&oc->pb);
		avformat_free_context(oc);
	}
	if (ffmpeg_fd >= 0)
		close(ffmpeg_fd);
	if (pid > 0) {
		int status;
		waitpid(pid, &status, 0);
		if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
			fprintf(stderr, "pwrec: ffmpeg failed\n");
			ret = 1;
		}
	}

	double secs = (st.last_vts_mono - st.first_vts_mono) / 1e9;
	fprintf(stderr, "pwrec: %llu frames in %.2f s (%.3f fps avg), frame interval "
		"%.2f..%.2f ms, peak queue %d frames, %llu audio resyncs\n",
		(unsigned long long)st.frames, secs,
		secs > 0 ? (st.frames - 1) / secs : 0.0,
		st.min_dt == INT64_MAX ? 0 : st.min_dt / 1e6, st.max_dt / 1e6,
		q.peak_video_count, (unsigned long long)st.audio_resyncs);
	if (opt.cfr_num && secs > 0) {
		double expect = secs * opt.cfr_num / opt.cfr_den + 1;
		double diff_ms = (st.frames - expect) * 1000.0 * opt.cfr_den / opt.cfr_num;
		if (fabs(diff_ms) > 20)
			fprintf(stderr, "pwrec: warning: producer delivered %.0f ms %s video than "
				"wall-clock at %d/%d fps; audio will be out of sync by that much "
				"at the end (use VFR or a matching --cfr)\n", fabs(diff_ms),
				diff_ms > 0 ? "more" : "less", opt.cfr_num, opt.cfr_den);
	}
	pw_deinit();
	return ret;
}

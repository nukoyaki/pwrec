#define _GNU_SOURCE
#include "pwrec_gpu.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <vulkan/vulkan.h>

#include <ffnvcodec/dynlink_loader.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>

#ifndef CUDA_EXTERNAL_MEMORY_DEDICATED
#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1
#endif

struct gpu_buf {
	VkBuffer buffer;
	VkDeviceMemory memory;
	uint32_t size;
};

static struct {
	CudaFunctions *cu;
	AVBufferRef *hw_dev;
	CUcontext cu_ctx;

	VkInstance instance;
	VkPhysicalDevice phys;
	VkDevice dev;
	VkQueue queue;
	VkCommandPool pool;
	VkCommandBuffer cmd;
	VkFence fence;
	PFN_vkGetMemoryFdKHR GetMemoryFdKHR;
	PFN_vkGetMemoryFdPropertiesKHR GetMemoryFdPropertiesKHR;

	VkBuffer stage;
	VkDeviceMemory stage_mem;
	CUexternalMemory stage_ext;
	CUdeviceptr stage_ptr;
	uint32_t row, height;
	VkBufferCopy *regions;

	AVBufferRef *frames;
	AVCodecContext *enc;
	AVFrame *frame;
	AVPacket *pkt;
} g;

#define VK_TRY(expr, what)                                                       \
	do {                                                                     \
		VkResult r_ = (expr);                                            \
		if (r_ != VK_SUCCESS) {                                          \
			fprintf(stderr, "pwrec: gpu: %s failed (VkResult %d)\n", \
				what, r_);                                       \
			goto fail;                                               \
		}                                                                \
	} while (0)

static int cu_check(CUresult r, const char *what)
{
	if (r == CUDA_SUCCESS)
		return 0;
	const char *name = NULL;
	if (g.cu->cuGetErrorName)
		g.cu->cuGetErrorName(r, &name);
	fprintf(stderr, "pwrec: gpu: %s failed (%s)\n", what, name ? name : "?");
	return -1;
}

static int find_memory_type(uint32_t bits, VkMemoryPropertyFlags want)
{
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
			return (int)i;
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
		if (bits & (1u << i))
			return (int)i;
	return -1;
}

int gpu_init(void)
{
	if (cuda_load_functions(&g.cu, NULL) < 0) {
		fprintf(stderr, "pwrec: gpu: can't load libcuda\n");
		return -1;
	}
	if (!g.cu->cuImportExternalMemory || !g.cu->cuExternalMemoryGetMappedBuffer ||
	    !g.cu->cuDeviceGetUuid) {
		fprintf(stderr, "pwrec: gpu: libcuda lacks external memory support\n");
		goto fail;
	}

	int r = av_hwdevice_ctx_create(&g.hw_dev, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0);
	if (r < 0) {
		fprintf(stderr, "pwrec: gpu: can't create CUDA device: %s\n", av_err2str(r));
		goto fail;
	}
	g.cu_ctx = ((AVCUDADeviceContext *)((AVHWDeviceContext *)g.hw_dev->data)->hwctx)->cuda_ctx;

	CUuuid uuid;
	CUdevice cu_dev;
	CUcontext dummy;
	if (cu_check(g.cu->cuCtxPushCurrent(g.cu_ctx), "cuCtxPushCurrent"))
		goto fail;
	int bad = cu_check(g.cu->cuCtxGetDevice(&cu_dev), "cuCtxGetDevice") ||
		  cu_check(g.cu->cuDeviceGetUuid(&uuid, cu_dev), "cuDeviceGetUuid");
	g.cu->cuCtxPopCurrent(&dummy);
	if (bad)
		goto fail;

	VkApplicationInfo app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "pwrec",
		.apiVersion = VK_API_VERSION_1_1,
	};
	VkInstanceCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app,
	};
	VK_TRY(vkCreateInstance(&ici, NULL, &g.instance), "vkCreateInstance");

	uint32_t n = 0;
	vkEnumeratePhysicalDevices(g.instance, &n, NULL);
	VkPhysicalDevice phys[16];
	if (n > 16)
		n = 16;
	vkEnumeratePhysicalDevices(g.instance, &n, phys);
	for (uint32_t i = 0; i < n && !g.phys; i++) {
		VkPhysicalDeviceIDProperties id = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
		VkPhysicalDeviceProperties2 p = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id };
		vkGetPhysicalDeviceProperties2(phys[i], &p);
		if (memcmp(id.deviceUUID, uuid.bytes, VK_UUID_SIZE) == 0)
			g.phys = phys[i];
	}
	if (!g.phys) {
		fprintf(stderr, "pwrec: gpu: no Vulkan device matches the CUDA device\n");
		goto fail;
	}

	uint32_t nq = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, NULL);
	VkQueueFamilyProperties qf[32];
	if (nq > 32)
		nq = 32;
	vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, qf);
	uint32_t family = UINT32_MAX;
	for (uint32_t i = 0; i < nq && family == UINT32_MAX; i++)
		if (qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))
			family = i;
	if (family == UINT32_MAX) {
		fprintf(stderr, "pwrec: gpu: no usable queue\n");
		goto fail;
	}

	const char *exts[] = {
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
	};
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = family,
		.queueCount = 1,
		.pQueuePriorities = &prio,
	};
	VkDeviceCreateInfo dci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &qci,
		.enabledExtensionCount = 2,
		.ppEnabledExtensionNames = exts,
	};
	VK_TRY(vkCreateDevice(g.phys, &dci, NULL, &g.dev), "vkCreateDevice (needs DMA-BUF support)");
	vkGetDeviceQueue(g.dev, family, 0, &g.queue);

	g.GetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(g.dev, "vkGetMemoryFdKHR");
	g.GetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(g.dev, "vkGetMemoryFdPropertiesKHR");
	if (!g.GetMemoryFdKHR || !g.GetMemoryFdPropertiesKHR) {
		fprintf(stderr, "pwrec: gpu: missing external memory functions\n");
		goto fail;
	}

	VkCommandPoolCreateInfo pci = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = family,
	};
	VK_TRY(vkCreateCommandPool(g.dev, &pci, NULL, &g.pool), "vkCreateCommandPool");
	VkCommandBufferAllocateInfo cai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = g.pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VK_TRY(vkAllocateCommandBuffers(g.dev, &cai, &g.cmd), "vkAllocateCommandBuffers");
	VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VK_TRY(vkCreateFence(g.dev, &fci, NULL, &g.fence), "vkCreateFence");

	return 0;

fail:
	gpu_destroy();
	return -1;
}

void gpu_destroy(void)
{
	if (g.enc)
		avcodec_free_context(&g.enc);
	av_frame_free(&g.frame);
	av_packet_free(&g.pkt);
	av_buffer_unref(&g.frames);

	if (g.stage_ext && g.cu) {
		CUcontext dummy;
		g.cu->cuCtxPushCurrent(g.cu_ctx);
		g.cu->cuDestroyExternalMemory(g.stage_ext);
		g.cu->cuCtxPopCurrent(&dummy);
		g.stage_ext = NULL;
	}

	if (g.dev) {
		vkDeviceWaitIdle(g.dev);
		if (g.stage)
			vkDestroyBuffer(g.dev, g.stage, NULL);
		if (g.stage_mem)
			vkFreeMemory(g.dev, g.stage_mem, NULL);
		if (g.fence)
			vkDestroyFence(g.dev, g.fence, NULL);
		if (g.pool)
			vkDestroyCommandPool(g.dev, g.pool, NULL);
		vkDestroyDevice(g.dev, NULL);
	}
	if (g.instance)
		vkDestroyInstance(g.instance, NULL);

	av_buffer_unref(&g.hw_dev);
	if (g.cu)
		cuda_free_functions(&g.cu);
	free(g.regions);
	memset(&g, 0, sizeof(g));
}

struct gpu_buf *gpu_import(int fd, uint32_t size)
{
	struct gpu_buf *b = calloc(1, sizeof(*b));
	if (!b)
		return NULL;
	b->size = size;

	int dup = fcntl(fd, F_DUPFD_CLOEXEC, 0);
	if (dup < 0) {
		free(b);
		return NULL;
	}

	VkMemoryFdPropertiesKHR fdp = { .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
	VK_TRY(g.GetMemoryFdPropertiesKHR(g.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dup, &fdp),
	       "vkGetMemoryFdPropertiesKHR");

	VkExternalMemoryBufferCreateInfo ebi = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	VkBufferCreateInfo bci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = &ebi,
		.size = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VK_TRY(vkCreateBuffer(g.dev, &bci, NULL, &b->buffer), "vkCreateBuffer (dmabuf)");

	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(g.dev, b->buffer, &req);
	int type = find_memory_type(req.memoryTypeBits & fdp.memoryTypeBits, 0);
	if (type < 0) {
		fprintf(stderr, "pwrec: gpu: no memory type can import the dmabuf\n");
		goto fail;
	}

	VkMemoryDedicatedAllocateInfo ded = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
		.buffer = b->buffer,
	};
	VkImportMemoryFdInfoKHR imp = {
		.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
		.pNext = &ded,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		.fd = dup,
	};
	VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = &imp,
		.allocationSize = req.size,
		.memoryTypeIndex = (uint32_t)type,
	};
	VK_TRY(vkAllocateMemory(g.dev, &mai, NULL, &b->memory), "vkAllocateMemory (dmabuf import)");
	dup = -1;

	VK_TRY(vkBindBufferMemory(g.dev, b->buffer, b->memory, 0), "vkBindBufferMemory (dmabuf)");
	return b;

fail:
	if (dup >= 0)
		close(dup);
	gpu_release(b);
	return NULL;
}

void gpu_release(struct gpu_buf *b)
{
	if (!b)
		return;
	if (g.dev) {
		if (b->buffer)
			vkDestroyBuffer(g.dev, b->buffer, NULL);
		if (b->memory)
			vkFreeMemory(g.dev, b->memory, NULL);
	}
	free(b);
}

static int create_stage(uint32_t row, uint32_t height)
{
	VkDeviceSize size = (VkDeviceSize)row * height;

	VkExternalMemoryBufferCreateInfo ebi = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
	};
	VkBufferCreateInfo bci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = &ebi,
		.size = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VK_TRY(vkCreateBuffer(g.dev, &bci, NULL, &g.stage), "vkCreateBuffer (stage)");

	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(g.dev, g.stage, &req);
	int type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (type < 0)
		goto fail;

	VkMemoryDedicatedAllocateInfo ded = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
		.buffer = g.stage,
	};
	VkExportMemoryAllocateInfo exp = {
		.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
		.pNext = &ded,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
	};
	VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = &exp,
		.allocationSize = req.size,
		.memoryTypeIndex = (uint32_t)type,
	};
	VK_TRY(vkAllocateMemory(g.dev, &mai, NULL, &g.stage_mem), "vkAllocateMemory (stage)");
	VK_TRY(vkBindBufferMemory(g.dev, g.stage, g.stage_mem, 0), "vkBindBufferMemory (stage)");

	int fd = -1;
	VkMemoryGetFdInfoKHR gfi = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
		.memory = g.stage_mem,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
	};
	VK_TRY(g.GetMemoryFdKHR(g.dev, &gfi, &fd), "vkGetMemoryFdKHR");

	CUcontext dummy;
	if (cu_check(g.cu->cuCtxPushCurrent(g.cu_ctx), "cuCtxPushCurrent")) {
		close(fd);
		goto fail;
	}
	CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd = {
		.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
		.handle.fd = fd,
		.size = req.size,
		.flags = CUDA_EXTERNAL_MEMORY_DEDICATED,
	};
	int bad = cu_check(g.cu->cuImportExternalMemory(&g.stage_ext, &hd), "cuImportExternalMemory");
	if (bad) {
		close(fd);
	} else {
		CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd = { .offset = 0, .size = size };
		bad = cu_check(g.cu->cuExternalMemoryGetMappedBuffer(&g.stage_ptr, g.stage_ext, &bd),
			       "cuExternalMemoryGetMappedBuffer");
	}
	g.cu->cuCtxPopCurrent(&dummy);
	if (bad)
		goto fail;

	g.row = row;
	g.height = height;
	g.regions = calloc(height, sizeof(*g.regions));
	return g.regions ? 0 : -1;

fail:
	return -1;
}

AVCodecContext *gpu_encoder_open(const char *codec_name, int width, int height,
				 enum AVPixelFormat sw_fmt, AVRational time_base,
				 AVRational framerate, AVDictionary **opts)
{
	const AVCodec *codec = avcodec_find_encoder_by_name(codec_name);
	if (!codec) {
		fprintf(stderr, "pwrec: gpu: unknown encoder '%s'\n", codec_name);
		return NULL;
	}

	if (create_stage((uint32_t)width * 4, (uint32_t)height) < 0)
		return NULL;

	g.frames = av_hwframe_ctx_alloc(g.hw_dev);
	if (!g.frames)
		return NULL;
	AVHWFramesContext *fc = (AVHWFramesContext *)g.frames->data;
	fc->format = AV_PIX_FMT_CUDA;
	fc->sw_format = sw_fmt;
	fc->width = width;
	fc->height = height;
	int r = av_hwframe_ctx_init(g.frames);
	if (r < 0) {
		fprintf(stderr, "pwrec: gpu: CUDA frames: %s\n", av_err2str(r));
		return NULL;
	}

	g.enc = avcodec_alloc_context3(codec);
	g.frame = av_frame_alloc();
	g.pkt = av_packet_alloc();
	if (!g.enc || !g.frame || !g.pkt)
		return NULL;

	g.enc->width = width;
	g.enc->height = height;
	g.enc->time_base = time_base;
	g.enc->framerate = framerate;
	g.enc->sample_aspect_ratio = (AVRational){ 1, 1 };
	g.enc->pix_fmt = AV_PIX_FMT_CUDA;
	g.enc->hw_frames_ctx = av_buffer_ref(g.frames);
	g.enc->color_range = AVCOL_RANGE_MPEG;
	g.enc->colorspace = AVCOL_SPC_BT709;
	g.enc->color_primaries = AVCOL_PRI_BT709;
	g.enc->color_trc = AVCOL_TRC_BT709;
	g.enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

	r = avcodec_open2(g.enc, codec, opts);
	if (r < 0) {
		fprintf(stderr, "pwrec: gpu: can't open %s: %s\n", codec_name, av_err2str(r));
		return NULL;
	}
	return g.enc;
}

static void drain(gpu_emit_fn emit)
{
	while (avcodec_receive_packet(g.enc, g.pkt) == 0) {
		emit(g.pkt);
		av_packet_unref(g.pkt);
	}
}

int gpu_encode(struct gpu_buf *b, uint32_t offset, uint32_t stride, int64_t pts, gpu_emit_fn emit)
{
	if (!g.enc || !b)
		return -1;
	if ((uint64_t)offset + (uint64_t)stride * (g.height - 1) + g.row > b->size) {
		fprintf(stderr, "pwrec: gpu: short dmabuf, skipping\n");
		return -1;
	}

	uint32_t nreg;
	if (stride == g.row) {
		g.regions[0] = (VkBufferCopy){ offset, 0, (VkDeviceSize)g.row * g.height };
		nreg = 1;
	} else {
		for (uint32_t y = 0; y < g.height; y++)
			g.regions[y] = (VkBufferCopy){ offset + (VkDeviceSize)y * stride,
						       (VkDeviceSize)y * g.row, g.row };
		nreg = g.height;
	}

	VkCommandBufferBeginInfo cbi = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	VK_TRY(vkResetCommandBuffer(g.cmd, 0), "vkResetCommandBuffer");
	VK_TRY(vkBeginCommandBuffer(g.cmd, &cbi), "vkBeginCommandBuffer");
	vkCmdCopyBuffer(g.cmd, b->buffer, g.stage, nreg, g.regions);
	VK_TRY(vkEndCommandBuffer(g.cmd), "vkEndCommandBuffer");

	VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &g.cmd,
	};
	VK_TRY(vkQueueSubmit(g.queue, 1, &si, g.fence), "vkQueueSubmit");
	VK_TRY(vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
	vkResetFences(g.dev, 1, &g.fence);

	int r = av_hwframe_get_buffer(g.frames, g.frame, 0);
	if (r < 0) {
		fprintf(stderr, "pwrec: gpu: no CUDA frame: %s\n", av_err2str(r));
		return -1;
	}

	CUDA_MEMCPY2D cp = {
		.srcMemoryType = CU_MEMORYTYPE_DEVICE,
		.srcDevice = g.stage_ptr,
		.srcPitch = g.row,
		.dstMemoryType = CU_MEMORYTYPE_DEVICE,
		.dstDevice = (CUdeviceptr)g.frame->data[0],
		.dstPitch = g.frame->linesize[0],
		.WidthInBytes = g.row,
		.Height = g.height,
	};
	CUcontext dummy;
	int bad = cu_check(g.cu->cuCtxPushCurrent(g.cu_ctx), "cuCtxPushCurrent") ||
		  cu_check(g.cu->cuMemcpy2D(&cp), "cuMemcpy2D");
	g.cu->cuCtxPopCurrent(&dummy);
	if (bad) {
		av_frame_unref(g.frame);
		return -1;
	}

	g.frame->pts = pts;
	r = avcodec_send_frame(g.enc, g.frame);
	av_frame_unref(g.frame);
	if (r < 0) {
		fprintf(stderr, "pwrec: gpu: encode failed: %s\n", av_err2str(r));
		return -1;
	}
	drain(emit);
	return 0;

fail:
	return -1;
}

void gpu_flush(gpu_emit_fn emit)
{
	if (!g.enc)
		return;
	avcodec_send_frame(g.enc, NULL);
	drain(emit);
}

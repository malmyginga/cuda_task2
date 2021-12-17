#include <cstdio>
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define RGBA_CHANNELS 4
#define RGB_CHANNELS 3
#define WARP_SIZE 32
#define KERNEL3_SIZE 9
#define KERNEL5_SIZE 25

#define SAFE_CALL( CallInstruction ) { \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) { \
         printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		 throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL( KernelCallInstruction ){ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel execution, aborting..."; \
    } \
}

float edge_detection[3][3] = {
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1
};

float identity[3][3] = {
    0, 0, 0,
    0, 1, 0,
    0, 0, 0
};

float sharpen[3][3] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};

float gaussian_blur[5][5] = {
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1
};

__constant__ float filter3[KERNEL3_SIZE];
__constant__ float filter5[KERNEL5_SIZE];

__device__ void filtering3(unsigned char *img, float *filter, int filter_size, unsigned char *p_out, int width) {
    float img_color = 0.;
    img_color = 0.;

    img_color += (float)*(img - width - 1) * *(filter);
    img_color += (float)*(img - width) * *(filter + 1);
    img_color += (float)*(img - width + 1) * *(filter + 2);
    img_color += (float)*(img - 1) * *(filter + filter_size);
    img_color += (float)*(img) * *(filter + filter_size + 1);
    img_color += (float)*(img + 1) * *(filter + filter_size + 2);
    img_color += (float)*(img + 1 * width - 1) * *(filter + filter_size * 2);
    img_color += (float)*(img + 1 * width) * *(filter + filter_size * 2 + 1);
    img_color += (float)*(img + 1 * width + 1) * *(filter + filter_size * 2 + 2);
    
    /* for (int i = 0; i < filter_size; ++i) {
        for (int j = 0; j < filter_size; ++j) {
            int offset = (i - 1) * width + (j - 1);
            img_color += (float)*(img + (i - 1) * width + (j - 1)) * *(filter + filter_size * i + j);
        }
    } */
    *p_out = (unsigned char)max(min((float)255., img_color), (float)0.);
}

__device__ void filtering5(unsigned char *img, float *filter, int filter_size, unsigned char *p_out, int width) {
    float img_color = 0.;
    img_color = 0.;
    
    img_color += (float)*(img + (0 - 1) * width + (0 - 1)) * *(filter + filter_size * 0 + 0);
    img_color += (float)*(img + (0 - 1) * width + (1 - 1)) * *(filter + filter_size * 0 + 1);
    img_color += (float)*(img + (0 - 1) * width + (2 - 1)) * *(filter + filter_size * 0 + 2);
    img_color += (float)*(img + (0 - 1) * width + (3 - 1)) * *(filter + filter_size * 0 + 3);
    img_color += (float)*(img + (0 - 1) * width + (4 - 1)) * *(filter + filter_size * 0 + 4);
    img_color += (float)*(img + (1 - 1) * width + (0 - 1)) * *(filter + filter_size * 1 + 0);
    img_color += (float)*(img + (1 - 1) * width + (1 - 1)) * *(filter + filter_size * 1 + 1);
    img_color += (float)*(img + (1 - 1) * width + (2 - 1)) * *(filter + filter_size * 1 + 2);
    img_color += (float)*(img + (1 - 1) * width + (3 - 1)) * *(filter + filter_size * 1 + 3);
    img_color += (float)*(img + (1 - 1) * width + (4 - 1)) * *(filter + filter_size * 1 + 4);
    img_color += (float)*(img + (2 - 1) * width + (0 - 1)) * *(filter + filter_size * 2 + 0);
    img_color += (float)*(img + (2 - 1) * width + (1 - 1)) * *(filter + filter_size * 2 + 1);
    img_color += (float)*(img + (2 - 1) * width + (2 - 1)) * *(filter + filter_size * 2 + 2);
    img_color += (float)*(img + (2 - 1) * width + (3 - 1)) * *(filter + filter_size * 2 + 3);
    img_color += (float)*(img + (2 - 1) * width + (4 - 1)) * *(filter + filter_size * 2 + 4);
    img_color += (float)*(img + (3 - 1) * width + (0 - 1)) * *(filter + filter_size * 3 + 0);
    img_color += (float)*(img + (3 - 1) * width + (1 - 1)) * *(filter + filter_size * 3 + 1);
    img_color += (float)*(img + (3 - 1) * width + (2 - 1)) * *(filter + filter_size * 3 + 2);
    img_color += (float)*(img + (3 - 1) * width + (3 - 1)) * *(filter + filter_size * 3 + 3);
    img_color += (float)*(img + (3 - 1) * width + (4 - 1)) * *(filter + filter_size * 3 + 4);
    img_color += (float)*(img + (4 - 1) * width + (0 - 1)) * *(filter + filter_size * 4 + 0);
    img_color += (float)*(img + (4 - 1) * width + (1 - 1)) * *(filter + filter_size * 4 + 1);
    img_color += (float)*(img + (4 - 1) * width + (2 - 1)) * *(filter + filter_size * 4 + 2);
    img_color += (float)*(img + (4 - 1) * width + (3 - 1)) * *(filter + filter_size * 4 + 3);
    img_color += (float)*(img + (4 - 1) * width + (4 - 1)) * *(filter + filter_size * 4 + 4);
    /*for (int i = 0; i < filter_size; ++i) {
        for (int j = 0; j < filter_size; ++j) {
            int img_x_curr = j - 1;
            int img_y_curr = i - 1;
            int offset = img_y_curr * width + img_x_curr;
            img_color += (float)*(img + offset) * *(filter + filter_size * i + j);
        }
    } */
    *p_out = (unsigned char)max(min((float)255., img_color), (float)0.);
}

__device__ void shared_mem_init(unsigned char *temp, unsigned char *img, int filter_size, int thread_in_block_index, int block_size,
int temp_x_size, int main_temp_index, int img_main_coord, int width) {
    for (int i = 0; i < filter_size; ++i) {
        int offset_temp = (i - filter_size / 2) * temp_x_size;
        int offset_img = (i - filter_size / 2) * width;
        temp[main_temp_index + offset_temp] = img[img_main_coord + offset_img];
    }
    if (thread_in_block_index == 0) {
        for (int i = 0; i < filter_size; ++i) {
            int offset_temp = (i - filter_size / 2) * temp_x_size;
            int offset_img = (i - filter_size / 2) * width;
            for (int j = 0; j < filter_size / 2; ++j) {
                temp[main_temp_index + offset_temp - filter_size / 2 + j] = img[img_main_coord + offset_img - filter_size / 2 + j];
            }
        }
    } else if (thread_in_block_index == (block_size - 1)) {
        for (int i = 0; i < filter_size; ++i) {
            int offset_temp = (i - filter_size / 2) * temp_x_size;
            int offset_img = (i - filter_size / 2) * width;
            for (int j = 0; j < filter_size / 2; ++j) {
                temp[main_temp_index + offset_temp + filter_size / 2 - j] = img[img_main_coord + offset_img + filter_size / 2 - j];
            }
        }
    }
}

#define MEM_SIZE 10000

//rgbchannels warpsize sizing for biggest block == 1024
__global__ void cuda_filter_rgb(unsigned char *img_red, unsigned char *img_green, unsigned char *img_blue,
     unsigned char *img_res_red, unsigned char *img_res_green, unsigned char *img_res_blue, 
     int filter_size, int height, int width) {


    //расчет координат

    //dim3 blockDims(WARP_SIZE, WARP_SIZE, 1);
    //dim3 gridDims(RGB_CHANNELS, ceil((width * height) / block_size), 1);
    int img_main_coord = blockIdx.y * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int img_x = img_main_coord % width;
    int img_y = img_main_coord / height;
    if (!(img_x > filter_size / 2 - 1 && img_x < width - filter_size / 2 && img_y > filter_size / 2 - 1 && img_y < height - filter_size / 2)) {
        return;
    } 

    int block_size = blockDim.x * blockDim.y * blockDim.z;
    //__shared__ unsigned char temp_red[block_size * filter_size + (filter_size - 1) * filter_size];

    __shared__ unsigned char temp_red[MEM_SIZE];
    __shared__ unsigned char temp_green[MEM_SIZE];
    __shared__ unsigned char temp_blue[MEM_SIZE];
    int temp_x_size = block_size + filter_size - 1;
    int thread_in_block_index = threadIdx.y * blockDim.x + threadIdx.x;
    int main_temp_index = temp_x_size * (filter_size / 2) + (filter_size / 2) * (filter_size / 2) * 2 + filter_size / 2 + thread_in_block_index;

    shared_mem_init(temp_red, img_red, filter_size, thread_in_block_index, block_size, temp_x_size, main_temp_index, img_main_coord, width);
    shared_mem_init(temp_green, img_green, filter_size, thread_in_block_index, block_size, temp_x_size, main_temp_index, img_main_coord, width);
    shared_mem_init(temp_blue, img_blue, filter_size, thread_in_block_index, block_size, temp_x_size, main_temp_index, img_main_coord, width);
    //transfer to shared memory
    __syncthreads();

    //apply stencil

    unsigned char *p_out;

    //red
    //p_out = img_res_red + img_main_coord;
    //filtering(temp_red + main_temp_index, filter, filter_size, p_out, temp_x_size);
    //green
    //p_out = img_res_green + img_main_coord;
    //filtering(temp_green + main_temp_index, filter, filter_size, p_out, temp_x_size);
    //blue
    //p_out = img_res_blue + img_main_coord;
    //filtering(temp_blue + main_temp_index, filter, filter_size, p_out, temp_x_size);

    int color = blockIdx.x;
    //if (img_main_coord < width * height)
    switch (color) {
        case 0:
            //red color
            p_out = img_res_red + img_main_coord;
            switch (filter_size) {
                case 3:
                    filtering3(temp_red + main_temp_index, filter3, filter_size, p_out, temp_x_size);
                    break;
                case 5:
                    filtering5(temp_red + main_temp_index, filter5, filter_size, p_out, temp_x_size);
                    break;
                default:
                    break;
            }
            break;
        case 1:
            //green color
            p_out = img_res_green + img_main_coord;
            switch (filter_size) {
                case 3:
                    filtering3(temp_green + main_temp_index, filter3, filter_size, p_out, temp_x_size);
                    break;
                case 5:
                    filtering5(temp_green + main_temp_index, filter5, filter_size, p_out, temp_x_size);
                    break;
                default:
                    break;
            }
            break;
        case 2:
            //blue color
            p_out = img_res_blue + img_main_coord;
            switch (filter_size) {
                case 3:
                    filtering3(temp_blue + main_temp_index, filter3, filter_size, p_out, temp_x_size);
                    break;
                case 5:
                    filtering5(temp_blue + main_temp_index, filter5, filter_size, p_out, temp_x_size);
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }
}

//реализуем разделение на три массива
//развертка цикла ядра
/*void read_img(unsigned char *img, unsigned char *img_red, unsigned char *img_green, unsigned char *img_blue, 
const char *img_path, int &width, int &height, int &channels) {

    img = stbi_load(img_path, &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading image\n");
        exit(1);
    }
    printf("Loaded image with\n width:%d\n height:%d\n channels:%d\n\n", width, height, channels);
} */

void img_rgb_init(unsigned char *img, unsigned char *img_red, unsigned char *img_green,
     unsigned char *img_blue, int &width, int &height, int &channels) {
    //разделение на три компоненты rgb
    int img_rgb_size = width * height;
    if (channels == RGB_CHANNELS) {
        for (int i = 0; i < img_rgb_size; ++i) {
            img_red[i] = img[i * RGB_CHANNELS];
            img_green[i] = img[i * RGB_CHANNELS + 1];
            img_blue[i] = img[i * RGB_CHANNELS + 2];
        }
    } else if (channels == RGBA_CHANNELS) {
        for (int i = 0; i < img_rgb_size; ++i) {
            img_red[i] = img[i * RGBA_CHANNELS];
            img_green[i] = img[i * RGBA_CHANNELS + 1];
            img_blue[i] = img[i * RGBA_CHANNELS + 2];
        }
    }
}

void save_img(unsigned char *img_res_red, unsigned char *img_res_green, 
unsigned char *img_res_blue, const char *img_path, int &width, int &height, const char *img_save_path) {
    int img_res_size = width * height * RGB_CHANNELS;
    unsigned char *img_res = (unsigned char *)malloc(width * height * RGB_CHANNELS * sizeof(*img_res));
    if (img_res == NULL) {
        printf("Unable to allocate memory to img_res_size\n");
        exit(1);
    }
    for (int i = 0; i < img_res_size; i++) {
        int res_color = i % RGB_CHANNELS;
        switch (res_color) {
            case 0:
                //red
                img_res[i] = img_res_red[i / RGB_CHANNELS];
                break;
            case 1:
                //green
                img_res[i] = img_res_green[i / RGB_CHANNELS];
                break;
            case 2:
                //blue
                img_res[i] = img_res_blue[i / RGB_CHANNELS];
                break;
            default:
                break;
        }
    }
    stbi_write_png(img_save_path, width, height, RGB_CHANNELS, img_res, width * RGB_CHANNELS);
    printf("Image written!\n");
    free(img_res);
}

float host_to_device(unsigned char *img_red, unsigned char *img_green, unsigned char *img_blue, float *filter, 
unsigned char *device_img_red, unsigned char *device_img_green, unsigned char *device_img_blue, float *device_filter, 
size_t img_color_byte_size, int filter_size) {

    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    float temp_time = 0.;

    SAFE_CALL(cudaEventRecord(start));
    SAFE_CALL(cudaMemcpy(device_img_red, img_red, img_color_byte_size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_img_green, img_green, img_color_byte_size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_img_blue, img_blue, img_color_byte_size, cudaMemcpyHostToDevice));

    switch (filter_size) {
        case 3:
            SAFE_CALL(cudaMemcpyToSymbol(filter3, filter, sizeof(*filter) * KERNEL3_SIZE));
            break;
        case 5:
            SAFE_CALL(cudaMemcpyToSymbol(filter5, filter, sizeof(*filter) * KERNEL5_SIZE));
            break;
        default:
            break;
    }

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    SAFE_CALL(cudaEventElapsedTime(&temp_time, start, stop));
    return temp_time;
}

float device_to_host(unsigned char *img_res_red, unsigned char *img_res_green, unsigned char *img_res_blue, 
unsigned char *device_img_res_red, unsigned char *device_img_res_green, unsigned char *device_img_res_blue,
size_t img_color_byte_size) {

    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    float temp_time = 0.;

    SAFE_CALL(cudaEventRecord(start));

    SAFE_CALL(cudaMemcpy(img_res_red, device_img_res_red, img_color_byte_size, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(img_res_green, device_img_res_green, img_color_byte_size, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(img_res_blue, device_img_res_blue, img_color_byte_size, cudaMemcpyDeviceToHost));

    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    SAFE_CALL(cudaEventElapsedTime(&temp_time, start, stop));
    return temp_time;
}

float kernel_apply(unsigned char *device_img_red, unsigned char *device_img_green, unsigned char *device_img_blue, 
unsigned char *device_img_res_red, unsigned char *device_img_res_green, unsigned char *device_img_res_blue, float *device_filter, 
int width, int height, int filter_size) {

    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    float temp_time = 0.;

    cudaDeviceProp device_prop;
    SAFE_CALL(cudaGetDeviceProperties(&device_prop, 0));
    int threads_in_block = floor(device_prop.maxThreadsPerBlock);
    
    // запуск вычислений на GPU
    int blockdim_y = threads_in_block / (RGB_CHANNELS * WARP_SIZE);
    //32 32 1 - blockDims
    dim3 blockDims(WARP_SIZE, WARP_SIZE, 1);
    float block_size = WARP_SIZE * WARP_SIZE * 1.;
    //3 x 88 x 1 - griddims for 300x300
    dim3 gridDims(RGB_CHANNELS, ceil((width * height) / block_size), 1);
    //dim3 gridDims(1, ceil((width * height) / block_size), 1);
    std::cout << "blockDim_y : " << blockdim_y << std::endl;
    std::cout << "gridDim_y : " << ceil((width * height) / block_size)<< std::endl;

    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((cuda_filter_rgb<<<gridDims, blockDims>>>(device_img_red, device_img_green, device_img_blue, 
    device_img_res_red, device_img_res_green, device_img_res_blue, filter_size, height, width)));

    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    SAFE_CALL(cudaEventElapsedTime(&temp_time, start, stop));
    return temp_time;
}

/*void mem_alloc(unsigned char *img_red, unsigned char *img_green, unsigned char *img_blue,
unsigned char *img_res_red, unsigned char *img_res_green, unsigned char *img_res_blue,
unsigned char *device_img_red, unsigned char *device_img_green, unsigned char *device_img_blue,
unsigned char *device_img_res_red, unsigned char *device_img_res_green, unsigned char *device_img_res_blue,
float *filter, const float *device_filter, size_t img_color_byte_size, size_t filter_byte_size) {

    img_red = (unsigned char *)malloc(img_color_byte_size);
    img_green = (unsigned char *)malloc(img_color_byte_size);
    img_blue = (unsigned char *)malloc(img_color_byte_size);

    img_res_red = (unsigned char *)malloc(img_color_byte_size);
    img_res_green = (unsigned char *)malloc(img_color_byte_size);
    img_res_blue = (unsigned char *)malloc(img_color_byte_size);

    SAFE_CALL(cudaMalloc(&device_img_red, img_color_byte_size));
    SAFE_CALL(cudaMalloc(&device_img_green, img_color_byte_size));
    SAFE_CALL(cudaMalloc(&device_img_blue, img_color_byte_size));

    SAFE_CALL(cudaMalloc(&device_img_res_red, img_color_byte_size));
    SAFE_CALL(cudaMalloc(&device_img_res_green, img_color_byte_size));
    SAFE_CALL(cudaMalloc(&device_img_res_blue, img_color_byte_size));

    SAFE_CALL(cudaMalloc(&device_filter, filter_byte_size));
} */


void mem_free(unsigned char *img_red, unsigned char *img_green, unsigned char *img_blue,
unsigned char *img_res_red, unsigned char *img_res_green, unsigned char *img_res_blue,
unsigned char *device_img_red, unsigned char *device_img_green, unsigned char *device_img_blue,
unsigned char *device_img_res_red, unsigned char *device_img_res_green, unsigned char *device_img_res_blue) {

    free(img_red);
    free(img_green);
    free(img_blue);

    free(img_res_red);
    free(img_res_green);
    free(img_res_blue);

    SAFE_CALL(cudaFree(device_img_red));
    SAFE_CALL(cudaFree(device_img_green));
    SAFE_CALL(cudaFree(device_img_blue));

    SAFE_CALL(cudaFree(device_img_res_red));
    SAFE_CALL(cudaFree(device_img_res_green));
    SAFE_CALL(cudaFree(device_img_res_blue));
}

std::vector<char *> get_filenames(const char *dir_path) {
    DIR *dir; struct dirent *diread;
    std::vector<char *> files;

    if ((dir = opendir(dir_path)) != NULL) {
        while ((diread = readdir(dir)) != NULL) {
            files.push_back(diread->d_name);
        }
        closedir (dir);
    } else {
        perror ("opendir");
        std::cout << "error dir" << std::endl;
        exit(1);
    }

    for (int i = 0; i < files.size(); i++) {
        std::cout << files[i] << "| ";
    }
    std::cout << std::endl;
    return files;
}

//filter_type 0 - edge detection v1, 1 - edge detection v2, 2 - gaussian blur, image_size (0 - small image, 1 - big image)
int main(int argc, char **argv) {
    int filter_type, image_size;
    sscanf(argv[1], "%d", &filter_type);
    sscanf(argv[2], "%d", &image_size);

    const char *img_path;
    if (image_size == 0) {
        img_path = "images/small.png";
    } else {
        img_path = "images/big.png";
    }

    //filter creation
    int filter_size;
    switch (filter_type) {
        case 0:
            filter_size = 3;
            break;
        case 1:
            filter_size = 3;
            break;
        case 2:
            filter_size = 5;
            break;
        default:
            break;
    }
    float *filter = (float *)malloc(filter_size * filter_size * sizeof(*filter));
    switch (filter_type) {
        case 0:
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    *(filter + filter_size * i + j) = edge_detection[i][j];
                }
            }
            break;
        case 1:
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    *(filter + filter_size * i + j) = sharpen[i][j];
                }
            }
            break;
        case 2:
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    *(filter + filter_size * i + j) = gaussian_blur[i][j] / 256.;
                }
            }
            break;
        default:
            break;
    }

    //filtering image

    unsigned char *img;

    unsigned char *img_red;
    unsigned char *img_green;
    unsigned char *img_blue;

    unsigned char *img_res_red;
    unsigned char *img_res_green;
    unsigned char *img_res_blue;

    unsigned char *device_img_red;
    unsigned char *device_img_green;
    unsigned char *device_img_blue;

    unsigned char *device_img_res_red;
    unsigned char *device_img_res_green;
    unsigned char *device_img_res_blue;

    unsigned char *img1;

    unsigned char *img_red1;
    unsigned char *img_green1;
    unsigned char *img_blue1;

    unsigned char *img_res_red1;
    unsigned char *img_res_green1;
    unsigned char *img_res_blue1;

    unsigned char *device_img_red1;
    unsigned char *device_img_green1;
    unsigned char *device_img_blue1;

    unsigned char *device_img_res_red1;
    unsigned char *device_img_res_green1;
    unsigned char *device_img_res_blue1;

    unsigned char *img2;

    unsigned char *img_red2;
    unsigned char *img_green2;
    unsigned char *img_blue2;

    unsigned char *img_res_red2;
    unsigned char *img_res_green2;
    unsigned char *img_res_blue2;

    unsigned char *device_img_red2;
    unsigned char *device_img_green2;
    unsigned char *device_img_blue2;

    unsigned char *device_img_res_red2;
    unsigned char *device_img_res_green2;
    unsigned char *device_img_res_blue2;

    size_t filter_byte_size = filter_size * filter_size * sizeof(*filter);
    int width, height, channels;

    float kernel_time = 0., transmit_time = 0., load_time = 0.;
    float total_time = 0.;

    float *device_filter;

    if (image_size == 1) { //big image
        //read_img(img, img_red, img_green, img_blue, img_path, width, height, channels);
        img = stbi_load(img_path, &width, &height, &channels, 0);
        if (img == NULL) {
            printf("Error in loading image\n");
            exit(1);
        }
        printf("Loaded image with\n width:%d\n height:%d\n channels:%d\n\n", width, height, channels);
        size_t img_color_byte_size = width * height * sizeof(*img);
        /*mem_alloc(img_red, img_green, img_blue, img_res_red, img_res_green, img_res_blue, device_img_red,
        device_img_green, device_img_blue, device_img_res_red, device_img_res_green,
        device_img_res_blue, filter, device_filter, img_color_byte_size, filter_byte_size); */

        img_red = (unsigned char *)malloc(img_color_byte_size);
        img_green = (unsigned char *)malloc(img_color_byte_size);
        img_blue = (unsigned char *)malloc(img_color_byte_size);

        img_res_red = (unsigned char *)malloc(img_color_byte_size);
        img_res_green = (unsigned char *)malloc(img_color_byte_size);
        img_res_blue = (unsigned char *)malloc(img_color_byte_size);

        SAFE_CALL(cudaMalloc(&device_img_red, img_color_byte_size));
        SAFE_CALL(cudaMalloc(&device_img_green, img_color_byte_size));
        SAFE_CALL(cudaMalloc(&device_img_blue, img_color_byte_size));

        SAFE_CALL(cudaMalloc(&device_img_res_red, img_color_byte_size));
        SAFE_CALL(cudaMalloc(&device_img_res_green, img_color_byte_size));
        SAFE_CALL(cudaMalloc(&device_img_res_blue, img_color_byte_size));

        SAFE_CALL(cudaMalloc(&device_filter, filter_byte_size));

        img_rgb_init(img, img_red, img_green, img_blue, width, height, channels);
        free(img);
        transmit_time += host_to_device(img_red, img_green, img_blue, filter, 
        device_img_red, device_img_green, device_img_blue, device_filter, img_color_byte_size, filter_size);
        kernel_time += kernel_apply(device_img_red, device_img_green, device_img_blue, 
        device_img_res_red, device_img_res_green, device_img_res_blue, device_filter, width, height, filter_size);
        transmit_time += device_to_host(img_res_red, img_res_green, img_res_blue,
        device_img_res_red, device_img_res_green, device_img_res_blue, img_color_byte_size);
        
        const char *img_save_path = "images/output/image.png";
        save_img(img_res_red, img_res_green, img_res_blue, img_path, width, height, img_save_path);
        mem_free(img_red, img_green, img_blue, img_res_red, img_res_green, img_res_blue,
        device_img_red, device_img_green, device_img_blue, device_img_res_red, device_img_res_green, device_img_res_blue);
        free(filter);
        SAFE_CALL(cudaFree(device_filter));
        return 0;
    } else if (image_size == 0) { //small images
        //обработка двух изображений за раз
        const char *dir_path = "images/small/";
        const char *dir_save_path = "images/output/";
        std::string dir_path_str(dir_path);
        std::string dir_save_path_str(dir_save_path);

        std::vector<char *> filenames = get_filenames(dir_path);
        filenames.erase(filenames.begin()); // delete .
        filenames.erase(filenames.begin()); // delete ..
        int number_of_images = filenames.size();
        int flag = 1;

        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        for (int i = 0; i < number_of_images / 2; ++i) {
            //read_img(img, img_red, img_green, img_blue, img_path, width, height, channels);
            std::vector<char *> filenames = get_filenames(dir_path);
            filenames.erase(filenames.begin()); // delete .
            filenames.erase(filenames.begin()); // delete ..

            std::string curr_filename1 = filenames[i * 2];
            std::cout << "filename1: " << curr_filename1 << std::endl;
            std::string img_load_path1 = dir_path_str + curr_filename1;
            std::string curr_filename2 = filenames[i * 2 + 1];
            std::cout << "filename2: " << curr_filename2 << std::endl;
            std::string img_load_path2 = dir_path_str + curr_filename2;

            img1 = stbi_load(img_load_path1.c_str(), &width, &height, &channels, 0);
            img2 = stbi_load(img_load_path2.c_str(), &width, &height, &channels, 0);
            if (img1 == NULL || img2 == NULL) {
                printf("Error in loading image\n");
                exit(1);
            }

            printf("Loaded image with\n width:%d\n height:%d\n channels:%d\n\n", width, height, channels);
            size_t img_color_byte_size = width * height * sizeof(*img);
            /*mem_alloc(img_red, img_green, img_blue, img_res_red, img_res_green, img_res_blue, device_img_red,
            device_img_green, device_img_blue, device_img_res_red, device_img_res_green,
            device_img_res_blue, filter, device_filter, img_color_byte_size, filter_byte_size); */
            
            if (flag) {

                cudaEvent_t start1, stop1;
                SAFE_CALL(cudaEventCreate(&start1));
                SAFE_CALL(cudaEventCreate(&stop1));
                
                SAFE_CALL(cudaEventRecord(start1));

                img_red1 = (unsigned char *)malloc(img_color_byte_size);
                img_green1 = (unsigned char *)malloc(img_color_byte_size);
                img_blue1 = (unsigned char *)malloc(img_color_byte_size);

                img_res_red1 = (unsigned char *)malloc(img_color_byte_size);
                img_res_green1 = (unsigned char *)malloc(img_color_byte_size);
                img_res_blue1 = (unsigned char *)malloc(img_color_byte_size);

                img_red2 = (unsigned char *)malloc(img_color_byte_size);
                img_green2 = (unsigned char *)malloc(img_color_byte_size);
                img_blue2 = (unsigned char *)malloc(img_color_byte_size);

                img_res_red2 = (unsigned char *)malloc(img_color_byte_size);
                img_res_green2 = (unsigned char *)malloc(img_color_byte_size);
                img_res_blue2 = (unsigned char *)malloc(img_color_byte_size);

                SAFE_CALL(cudaMalloc(&device_img_red1, img_color_byte_size));
                SAFE_CALL(cudaMalloc(&device_img_green1, img_color_byte_size));
                SAFE_CALL(cudaMalloc(&device_img_blue1, img_color_byte_size));

                SAFE_CALL(cudaMalloc(&device_img_res_red1, img_color_byte_size));
                SAFE_CALL(cudaMalloc(&device_img_res_green1, img_color_byte_size));
                SAFE_CALL(cudaMalloc(&device_img_res_blue1, img_color_byte_size));

                SAFE_CALL(cudaMalloc(&device_img_red2, img_color_byte_size));
                SAFE_CALL(cudaMalloc(&device_img_green2, img_color_byte_size));
                SAFE_CALL(cudaMalloc(&device_img_blue2, img_color_byte_size));

                SAFE_CALL(cudaMalloc(&device_img_res_red2, img_color_byte_size));
                SAFE_CALL(cudaMalloc(&device_img_res_green2, img_color_byte_size));
                SAFE_CALL(cudaMalloc(&device_img_res_blue2, img_color_byte_size));

                SAFE_CALL(cudaMalloc(&device_filter, filter_byte_size));

                SAFE_CALL(cudaDeviceSynchronize());
                SAFE_CALL(cudaEventRecord(stop1));
                SAFE_CALL(cudaEventSynchronize(stop1));
                SAFE_CALL(cudaEventElapsedTime(&load_time, start1, stop1));
            }
            img_rgb_init(img1, img_red1, img_green1, img_blue1, width, height, channels);
            img_rgb_init(img2, img_red2, img_green2, img_blue2, width, height, channels);
            free(img1);
            free(img2);
            
            cudaEvent_t start, stop;
            SAFE_CALL(cudaEventCreate(&start));
            SAFE_CALL(cudaEventCreate(&stop));
            float temp_time = 0.;

            SAFE_CALL(cudaEventRecord(start));
            SAFE_CALL(cudaMemcpyAsync(device_img_red1, img_red1, img_color_byte_size, cudaMemcpyHostToDevice, stream1));
            SAFE_CALL(cudaMemcpyAsync(device_img_green1, img_green1, img_color_byte_size, cudaMemcpyHostToDevice, stream1));
            SAFE_CALL(cudaMemcpyAsync(device_img_blue1, img_blue1, img_color_byte_size, cudaMemcpyHostToDevice, stream1));


            SAFE_CALL(cudaMemcpyAsync(device_img_red2, img_red2, img_color_byte_size, cudaMemcpyHostToDevice, stream2));
            SAFE_CALL(cudaMemcpyAsync(device_img_green2, img_green2, img_color_byte_size, cudaMemcpyHostToDevice, stream2));
            SAFE_CALL(cudaMemcpyAsync(device_img_blue2, img_blue2, img_color_byte_size, cudaMemcpyHostToDevice, stream2));

            switch (filter_size) {
                case 3:
                    SAFE_CALL(cudaMemcpyToSymbol(filter3, filter, sizeof(*filter) * KERNEL3_SIZE));
                    break;
                case 5:
                    SAFE_CALL(cudaMemcpyToSymbol(filter5, filter, sizeof(*filter) * KERNEL5_SIZE));
                    break;
                default:
                    break;
            }


            cudaEvent_t kernel_start, kernel_stop;
            float temp_kernel_time = 0.;
            SAFE_CALL(cudaEventCreate(&kernel_start));
            SAFE_CALL(cudaEventCreate(&kernel_stop));

            cudaDeviceProp device_prop;
            SAFE_CALL(cudaGetDeviceProperties(&device_prop, 0));
            int threads_in_block = floor(device_prop.maxThreadsPerBlock);
            int blockdim_y = threads_in_block / (RGB_CHANNELS * WARP_SIZE);
            dim3 blockDims(WARP_SIZE, WARP_SIZE, 1);
            float block_size = WARP_SIZE * WARP_SIZE * 1.;
            dim3 gridDims(RGB_CHANNELS, ceil((width * height) / block_size), 1);
            std::cout << "blockDim_y : " << blockdim_y << std::endl;
            std::cout << "gridDim_y : " << ceil((width * height) / block_size)<< std::endl;

            SAFE_CALL(cudaEventRecord(kernel_start));

            SAFE_KERNEL_CALL((cuda_filter_rgb<<<gridDims, blockDims, 0, stream1>>>(device_img_red1, device_img_green1, device_img_blue1, 
            device_img_res_red1, device_img_res_green1, device_img_res_blue1, filter_size, height, width)));
            SAFE_KERNEL_CALL((cuda_filter_rgb<<<gridDims, blockDims, 0, stream2>>>(device_img_red2, device_img_green2, device_img_blue2, 
            device_img_res_red2, device_img_res_green2, device_img_res_blue2, filter_size, height, width)));

            SAFE_CALL(cudaDeviceSynchronize());
            SAFE_CALL(cudaEventRecord(kernel_stop));
            SAFE_CALL(cudaEventSynchronize(kernel_stop));
            SAFE_CALL(cudaEventElapsedTime(&temp_kernel_time, kernel_start, kernel_stop));
            kernel_time += temp_kernel_time;

            SAFE_CALL(cudaMemcpyAsync(img_res_red1, device_img_res_red1, img_color_byte_size, cudaMemcpyDeviceToHost, stream1));
            SAFE_CALL(cudaMemcpyAsync(img_res_green1, device_img_res_green1, img_color_byte_size, cudaMemcpyDeviceToHost, stream1));
            SAFE_CALL(cudaMemcpyAsync(img_res_blue1, device_img_res_blue1, img_color_byte_size, cudaMemcpyDeviceToHost, stream1));

            SAFE_CALL(cudaMemcpyAsync(img_res_red2, device_img_res_red2, img_color_byte_size, cudaMemcpyDeviceToHost, stream2));
            SAFE_CALL(cudaMemcpyAsync(img_res_green2, device_img_res_green2, img_color_byte_size, cudaMemcpyDeviceToHost, stream2));
            SAFE_CALL(cudaMemcpyAsync(img_res_blue2, device_img_res_blue2, img_color_byte_size, cudaMemcpyDeviceToHost, stream2));

            SAFE_CALL(cudaDeviceSynchronize());
            SAFE_CALL(cudaEventRecord(stop));
            SAFE_CALL(cudaEventSynchronize(stop));
            SAFE_CALL(cudaEventElapsedTime(&temp_time, start, stop));

            total_time += temp_time;
            std::string img_save_path_str1 = dir_save_path_str + curr_filename1;
            std::string img_save_path_str2 = dir_save_path_str + curr_filename2;
            save_img(img_res_red1, img_res_green1, img_res_blue1, img_path, width, height, img_save_path_str1.c_str());
            save_img(img_res_red2, img_res_green2, img_res_blue2, img_path, width, height, img_save_path_str2.c_str());
            flag = 0;
        }
        mem_free(img_red1, img_green1, img_blue1, img_res_red1, img_res_green1, img_res_blue1,
        device_img_red1, device_img_green1, device_img_blue1, device_img_res_red1, device_img_res_green1, device_img_res_blue1);

        mem_free(img_red2, img_green2, img_blue2, img_res_red2, img_res_green2, img_res_blue2,
        device_img_red2, device_img_green2, device_img_blue2, device_img_res_red2, device_img_res_green2, device_img_res_blue2);

        free(filter);
        SAFE_CALL(cudaFree(device_filter));
        
    }
    std::cout << "KERNEL TIME: " << kernel_time << std::endl;
    std::cout << "TRANSMIT + KERNEL TIME: " << total_time + load_time << std::endl;
    return 0;
}
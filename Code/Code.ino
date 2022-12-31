/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Includes ---------------------------------------------------------------- */
#include <Driver_Drowsiness_Detection_inferencing.h>

#include "edge-impulse-sdk/dsp/image/image.hpp"

#include "camera.h"
#include "gc2145.h"
#include <ea_malloc.h>

/* Constant defines -------------------------------------------------------- */
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS           320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS           240
#define EI_CAMERA_RAW_FRAME_BYTE_SIZE             2
#define ALIGN_PTR(p,a)   ((p & (a-1)) ?(((uintptr_t)p + a) & ~(uintptr_t)(a-1)) : p)

/* Edge Impulse ------------------------------------------------------------- */

typedef struct {
    size_t width;
    size_t height;
} ei_device_resize_resolutions_t;

int ei_get_serial_available(void) {
    return Serial.available();
}


char ei_get_serial_byte(void) {
    return Serial.read();
}

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static bool is_initialised = false;
static bool is_ll_initialised = false;

GC2145 galaxyCore;
Camera cam(galaxyCore);
FrameBuffer fb;

static uint8_t *ei_camera_capture_out = NULL;
static uint8_t *ei_camera_frame_mem;
static uint8_t *ei_camera_frame_buffer; // 32-byte aligned

/* Function definitions ------------------------------------------------------- */
bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) ;
int calculate_resize_dimensions(uint32_t out_width, uint32_t out_height, uint32_t *resize_col_sz, uint32_t *resize_row_sz, bool *do_resize);

byte count = 0;

void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");

    // initialise M4 RAM
    // Arduino Nicla Vision has 512KB of RAM allocated for M7 core
    // and additional 244k (sic!) on the M4 address space
    // allocating 288 kB as in the line below was
    // advised by a member of Arduino team
    malloc_addblock((void*)0x30000000, 288 * 1024);

    if (ei_camera_init() == false) {
        ei_printf("Failed to initialize Camera!\r\n");
    }
    else {
        ei_printf("Camera initialized\r\n");
    }

    pinMode(PF_3, OUTPUT);         // Set LED and Buzzer as output
}

void loop()
{   
  if(count > 4){
      for(byte i=0; i < 10; i++)
       {       
        digitalWrite(PF_3,HIGH);
        delay(100);
        digitalWrite(PF_3,LOW);
        delay(100);
      }
      count = 0;
    }
    ei_printf("\nStarting inferencing in 2 seconds...\n");

    // instead of wait_ms, we'll wait on the signal, this allows threads to cancel us...
    if (ei_sleep(1000) != EI_IMPULSE_OK) {
        return;
    }

    ei_printf("Taking photo...\n");

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data;

    if (ei_camera_capture((size_t)EI_CLASSIFIER_INPUT_WIDTH, (size_t)EI_CLASSIFIER_INPUT_HEIGHT, NULL) == false) {
        ei_printf("Failed to capture image\r\n");
        return;
    }

    // Run the classifier
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        return;
    }

    // print the predictions
    ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    bool bb_found = result.bounding_boxes[0].value > 0;
    for (size_t ix = 0; ix < result.bounding_boxes_count; ix++) {
        auto bb = result.bounding_boxes[ix];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("    %s (", bb.label);
        ei_printf_float(bb.value);
        ei_printf(") [ x: %u, y: %u, width: %u, height: %u ]\n", bb.x, bb.y, bb.width, bb.height);

        if (bb.label == "Closed_Eye"){
          count += 1;
        }
    }

    if (!bb_found) {
        ei_printf("    No objects found\n");
    }
#else
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: ", result.classification[ix].label);
        ei_printf_float(result.classification[ix].value);
        ei_printf("\n");
        if (result.classification[ix].label == "Closed_Eye"){
          count += 1;
        }
    }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: ");
    ei_printf_float(result.anomaly);
    ei_printf("\n");
#endif
#endif
}


bool ei_camera_init(void) {
    if (is_initialised) return true;

    if (is_ll_initialised == false) {
        if (!cam.begin(CAMERA_R320x240, CAMERA_RGB565, -1)) {
            ei_printf("ERR: Failed to initialise camera\r\n");
            return false;
        }

    // initialize frame buffer
    ei_camera_frame_mem = (uint8_t *) ei_malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_RAW_FRAME_BYTE_SIZE + 32 /*alignment*/);
    if(ei_camera_frame_mem == NULL) {
        ei_printf("failed to create ei_camera_frame_mem\r\n");
        return false;
    }
    ei_camera_frame_buffer = (uint8_t *)ALIGN_PTR((uintptr_t)ei_camera_frame_mem, 32);

    fb.setBuffer(ei_camera_frame_buffer);
    is_initialised = true;
    }
    
    return true;
}


void ei_camera_deinit(void) {

    ei_free(ei_camera_frame_mem);
    ei_camera_frame_mem = NULL;
    ei_camera_frame_buffer = NULL;
    is_initialised = false;
}

bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    bool do_resize = false;
    bool do_crop = false;

    ei_camera_capture_out = (uint8_t*)ea_malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * 3 + 32);
    ei_camera_capture_out = (uint8_t *)ALIGN_PTR((uintptr_t)ei_camera_capture_out, 32);

    if (!is_initialised) {
        ei_printf("ERR: Camera is not initialized\r\n");
        return false;
    }

    int snapshot_response = cam.grabFrame(fb, 100);
    if (snapshot_response != 0) {
        ei_printf("ERR: Failed to get snapshot (%d)\r\n", snapshot_response);
        return false;
    }

    bool converted = RBG565ToRGB888(ei_camera_frame_buffer, ei_camera_capture_out, cam.frameSize());

    if(!converted){
        ei_printf("ERR: Conversion failed\n");
        ei_free(ei_camera_frame_mem);
        return false;
    }

    uint32_t resize_col_sz;
    uint32_t resize_row_sz;
    // choose resize dimensions
    int res = calculate_resize_dimensions(img_width, img_height, &resize_col_sz, &resize_row_sz, &do_resize);
    if (res) {
        ei_printf("ERR: Failed to calculate resize dimensions (%d)\r\n", res);
        return false;
    }

    if ((img_width != resize_col_sz)
        || (img_height != resize_row_sz)) {
        do_crop = true;
    }

    if (do_resize) {

          ei::image::processing::crop_and_interpolate_rgb888(
          ei_camera_capture_out,
          EI_CAMERA_RAW_FRAME_BUFFER_COLS,
          EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
          ei_camera_capture_out,
          resize_col_sz,
          resize_row_sz);
    }

    ea_free(ei_camera_capture_out);
    return true;
}


bool RBG565ToRGB888(uint8_t *src_buf, uint8_t *dst_buf, uint32_t src_len)
{
    uint8_t hb, lb;
    uint32_t pix_count = src_len / 2;

    for(uint32_t i = 0; i < pix_count; i ++) {
        hb = *src_buf++;
        lb = *src_buf++;

        *dst_buf++ = hb & 0xF8;
        *dst_buf++ = (hb & 0x07) << 5 | (lb & 0xE0) >> 3;
        *dst_buf++ = (lb & 0x1F) << 3;
    }

    return true;
}

static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr)
{
    // we already have a RGB888 buffer, so recalculate offset into pixel index
    size_t pixel_ix = offset * 3;
    size_t pixels_left = length;
    size_t out_ptr_ix = 0;

    while (pixels_left != 0) {
        out_ptr[out_ptr_ix] = (ei_camera_capture_out[pixel_ix] << 16) + (ei_camera_capture_out[pixel_ix + 1] << 8) + ei_camera_capture_out[pixel_ix + 2];

        // go to the next pixel
        out_ptr_ix++;
        pixel_ix+=3;
        pixels_left--;
    }

    // and done!
    return 0;
}

int calculate_resize_dimensions(uint32_t out_width, uint32_t out_height, uint32_t *resize_col_sz, uint32_t *resize_row_sz, bool *do_resize)
{
    size_t list_size = 6;
    const ei_device_resize_resolutions_t list[list_size] = {
        {64, 64},
        {96, 96},
        {160, 120},
        {160, 160},
        {320, 240},
    };

    // (default) conditions
    *resize_col_sz = EI_CAMERA_RAW_FRAME_BUFFER_COLS;
    *resize_row_sz = EI_CAMERA_RAW_FRAME_BUFFER_ROWS;
    *do_resize = false;

    for (size_t ix = 0; ix < list_size; ix++) {
        if ((out_width <= list[ix].width) && (out_height <= list[ix].height)) {
            *resize_col_sz = list[ix].width;
            *resize_row_sz = list[ix].height;
            *do_resize = true;
            break;
        }
    }

    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor"
#endif

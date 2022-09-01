#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#if defined(__GNUC__)
#define CGYM_API __attribute__((__dllexport__))
#else
#define CGYM_API __declspec(dllexport)
#endif
#else
#if defined(__GNUC__)
#define CGYM_API __attribute__((__visibility__("default")))
#else
#define CGYM_API
#endif
#endif

#define CGYM_VERSION 1

// Poissible value types
enum cgym_value_type {
    // Primitive values
    CGYM_VALUE_TYPE_INT = 0,
    CGYM_VALUE_TYPE_FLOAT = 1,
    CGYM_VALUE_TYPE_DOUBLE = 2,
    CGYM_VALUE_TYPE_BYTE = 3,

    // Spaces
    CGYM_SPACE_TYPE_BOX = 4,
    CGYM_SPACE_TYPE_MULTI_DISCRETE = 5
};

// Generic value type
typedef union {
    int32_t i;
    float f;
    double d;
    uint8_t b;
} cgym_value;

// Generic buffer type
typedef union {
    int32_t* i;
    float* f;
    double* d;
    uint8_t* b;
} cgym_value_buffer;

typedef struct {
    // Key is a short string
    char* key;

    // Tag and union
    cgym_value_type value_type;
    int32_t value_buffer_size;
    cgym_value_buffer value_buffer;
} cgym_key_value;

// Options for setup
typedef struct {
    char* name;

    // Tag and union
    cgym_value_type value_type;
    cgym_value value;
} cgym_option;

// Returned by make
typedef struct {
    int32_t observation_spaces_size;
    cgym_key_value* observation_spaces;

    int32_t action_spaces_size;
    cgym_key_value* action_spaces;
} cgym_make_data;

// Returned by reset
typedef struct {
    int32_t observations_size;
    cgym_key_value* observations;

    int32_t infos_size;
    cgym_key_value* infos;
} cgym_reset_data;

// Returned by step
typedef struct {
    int32_t observations_size;
    cgym_key_value* observations;

    cgym_value reward;
    bool terminated;
    bool truncated;

    int32_t infos_size;
    cgym_key_value* infos;
} cgym_step_data;

// Returned by render
typedef struct {
    // Type of image
    cgym_value_type value_type;

    // Dimensions of image
    int32_t value_buffer_width;
    int32_t value_buffer_height;
    int32_t value_buffer_channels;
    
    // Image buffer
    cgym_value_buffer value_buffer; // Size height * width * channels, addressed like: channel_index + channels * (x + width * y)
} cgym_frame;

// C ENV DEVELOPERS: IMPLEMENT THESE IN YOUR ENV!
CGYM_API int32_t cgym_get_env_version(); // Version of environment
CGYM_API cgym_make_data* cgym_make(char* render_mode, cgym_option* options, int32_t options_size); // Make the environment
CGYM_API cgym_reset_data* cgym_reset(int32_t seed, cgym_option* options, int32_t options_size); // Reset the environment
CGYM_API cgym_step_data* step(cgym_key_value* actions, int32_t actions_size); // Step (update) the environment
CGYM_API cgym_frame* render(); // Render the environment to a frame
CGYM_API void cgym_close(); // Close (delete) the environment (shutdown)

#ifdef __cplusplus
}
#endif

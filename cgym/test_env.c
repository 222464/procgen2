#include "cgym.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Global stuff that sticks around
int32_t observations_size;
cgym_key_value* observations;

cgym_make_data make_data;
cgym_reset_data reset_data;
cgym_step_data step_data;
cgym_frame frame;

float t; // Timer

int32_t cgym_get_env_version() {
    return 123;
}

cgym_make_data* cgym_make(char* render_mode, cgym_option* options, int32_t options_size) {
    // Allocate make data
    make_data.observation_spaces_size = 1;
    make_data.observation_spaces = (cgym_key_value*)malloc(sizeof(cgym_key_value));

    make_data.observation_spaces[0].key = "obs1";
    make_data.observation_spaces[0].value_type = CGYM_SPACE_TYPE_BOX;
    make_data.observation_spaces[0].value_buffer_size = 20; // Low and high, both 10 in size

    make_data.observation_spaces[0].value_buffer.f = (float*)malloc(make_data.observation_spaces[0].value_buffer_size * sizeof(float));

    // Low
    for (int i = 0; i < 10; i++)
        make_data.observation_spaces[0].value_buffer.f[i] = -1.0f;

    // High
    for (int i = 10; i < 20; i++)
        make_data.observation_spaces[0].value_buffer.f[i] = 1.0f;

    make_data.action_spaces_size = 1;
    make_data.action_spaces = (cgym_key_value*)malloc(sizeof(cgym_key_value));

    make_data.action_spaces[0].key = "act1";
    make_data.action_spaces[0].value_type = CGYM_SPACE_TYPE_MULTI_DISCRETE;
    make_data.action_spaces[0].value_buffer_size = 1;

    make_data.action_spaces[0].value_buffer.i = (int32_t*)malloc(sizeof(int32_t));

    // Allocate observations once and re-use (doesn't resize dynamically)
    observations_size = 10;
    observations = (cgym_key_value*)malloc(observations_size * sizeof(int32_t));

    // Reset data
    reset_data.observations_size = observations_size;
    reset_data.observations = observations;
    reset_data.infos_size = 0;
    reset_data.infos = NULL;

    // Step data
    step_data.observations_size = observations_size;
    step_data.observations = observations;
    step_data.reward.f = 0.0f;
    step_data.terminated = false;
    step_data.truncated = false;
    step_data.infos_size = 0;
    step_data.infos = NULL;

    // Frame
    frame.value_type = CGYM_VALUE_TYPE_BYTE;
    frame.value_buffer_height = 8;
    frame.value_buffer_width = 8;
    frame.value_buffer_channels = 3;
    frame.value_buffer.b = (uint8_t*)malloc(8 * 8 * 3 * sizeof(uint8_t));

    // Game
    t = 0.0f;

    return &make_data;
}

cgym_reset_data* cgym_reset(int32_t seed, cgym_option* options, int32_t options_size) {
    t = 0.0f;

    return &reset_data;
}

cgym_step_data* cgym_step(cgym_key_value* actions, int32_t actions_size) {
    step_data.reward.f = sinf(t);

    return &step_data;
}

cgym_frame* cgym_render() {
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            frame.value_buffer.b[0 + 3 * (x + 8 * y)] = 64;
        }

    return &frame;
}

void cgym_close() {
    // Dealloc make data
    for (int i = 0; i < make_data.observation_spaces_size; i++)
        free(make_data.observation_spaces[i].value_buffer.f);

    free(make_data.observation_spaces);

    for (int i = 0; i < make_data.action_spaces_size; i++)
        free(make_data.action_spaces[i].value_buffer.i);

    free(make_data.action_spaces);

    // Observations
    free(observations);

    // Frame
    free(frame.value_buffer.b);
}
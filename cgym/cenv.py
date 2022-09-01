import gym
from gym import Env
import collections
import copy
import os
import platform
import numpy as np
from ctypes import *
import struct

# Types
CGYM_VALUE_TYPE_INT = 0
CGYM_VALUE_TYPE_FLOAT = 1
CGYM_VALUE_TYPE_DOUBLE = 2
CGYM_VALUE_TYPE_BYTE = 3

CGYM_VALUE_TYPE_BOX = 4
CGYM_VALUE_TYPE_MULTI_DISCRETE = 5

CGYM_VALUE_TYPE_TO_CTYPE = [
    c_int32,
    c_float,
    c_double,
    c_byte,

    # Space types
    c_float,
    c_int32
]

CGYM_PYTHON_TYPE_TO_VALUE_TYPE = {
    int: 0,
    float: 1
}

CGYM_NUMPY_DTYPE_TO_VALUE_TYPE = {
    np.int32: 0,
    np.float32: 1,
    np.float64: 2,
    np.uint8: 3
}

class CGym_Value(Union):
    _fields_ = [("i", c_int32),
                ("f", c_float),
                ("d", c_double),
                ("b", c_byte)]

class CGym_Value_Buffer(Union):
    _fields_ = [("i", pointer(c_int32)),
                ("f", pointer(c_float)),
                ("d", pointer(c_double)),
                ("b", pointer(c_byte))]

class CGym_Key_Value(Structure):
    _fields_ = [("key", c_char_p),
                ("value_type", c_int32),
                ("value_buffer_size", cl_int32),
                ("value_buffer", CGym_Value_Buffer)]

class CGym_Option(Structure):
    _fields_ = [("name", c_char_p),
                ("value_type", c_int32),
                ("value", CGym_Value)]

class CGym_Make_Data(Structure):
    _fields_ = [("observation_spaces_size", cl_int32),
                ("observation_spaces", pointer(CGym_Key_Value)),
                ("info_size", c_int32),
                ("infos", pointer(CGym_Key_Value))] 

class CGym_Reset_Data(Structure):
    _fields_ = [("observation_size", cl_int32),
                ("observations", pointer(CGym_Key_Value)),
                ("info_size", c_int32),
                ("infos", pointer(CGym_Key_Value))] 

class CGym_Step_Data(Structure):
    _fields_ = [("observation_size", cl_int32),
                ("observations", pointer(CGym_Key_Value)),
                ("reward", CGym_Value),
                ("terminated", c_bool),
                ("truncated", c_bool),
                ("info_size", c_int32),
                ("infos", pointer(CGym_Key_Value))] 

class CGym_Frame(Structure):
    _fields_ = [("value_type", c_int32),
                ("value_buffer_width", c_int32),
                ("value_buffer_height", c_int32),
                ("value_buffer_channels", c_int32),
                ("value_buffer", CGym_Value_Buffer)]

class CEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"]}

    def __init__(self, lib_file_path: str, render_mode: Optional[str] = None, options: Optional[Dict[str, Any]] = None):
        # Load library
        self.lib = CDLL(lib_file_path)

        # Set up functions for Python
        self.lib.cgym_get_env_version.argtypes = []
        self.lib.cgym_get_env_version.restype = c_int32

        self.lib.cgym_make.argtypes = [pointer(CGym_Option), c_int32]
        self.lib.cgym_make.restype = pointer(CGym_Make_Data)

        self.lib.cgym_reset.argtypes = [c_int32, pointer(CGym_Option), c_int32]
        self.lib.cgym_reset.restype = pointer(CGym_Reset_Data)

        self.lib.cgym_step.argtypes = [pointer(CGym_Key_Value), c_int32]
        self.lib.cgym_step.restype = pointer(CGym_Step_Data)

        self.lib.cgym_frame.argtypes = []
        self.lib.cgym_frame.restype = pointer(CGym_Frame)

        self.lib.cgym_close.argtypes = []
        self.lib.cgym_close.restype = None

        c_make_data_p = None

        if options == None:
            c_make_data_p = self.lib.cgym_make("" if render_mode == None else render_mode, None, c_int32(0))
        else:
            # Make environment
            c_options = CGym_Option * len(options)

            for k, v in options.items():
                c_options[i].name = k

                value_type = CGYM_PYTHON_TYPE_TO_VALUE_TYPE[type(v)]

                c_options[i].value_type = c_int32(value_type)
                c_options[i].value = CGym_Value(GYM_VALUE_TYPE_TO_CTYPE[value_type](v))

            c_make_data_p = self.lib.cgym_make("" if render_mode == None else render_mode, c_options, c_int32(len(options)))

        self.observation_space = {}
        
        for i in range(c_make_data_p.observation_spaces_size):
            value_type = int(c_make_data_p.observation_spaces[i].value_type)
            value_buffer_size = int(c_make_data_p.observation_spaces[i].value_buffer_size)
            c_buffer_p = c_make_data_p.observation_spaces[i].value_buffer

            arr = np.ctypeslib.as_array((CGYM_VALUE_TYPE_TO_CTYPE[value_type] * value_buffer_size).from_address(c_buffer_p))

            space = None

            if value_type == CGYM_VALUE_TYPE_MULTI_DISCRETE:
                space = gym.spaces.MultiDiscrete(arr)
            else:
                space = gym.spaces.Box(arr[:len(arr) // 2], arr[len(arr) // 2:])

            self.observation_space[c_make_data_p.observation_spaces[i].key] = space
        
        self.action_space = {}
        
        for i in range(c_make_data_p.action_spaces_size):
            value_type = int(c_make_data_p.action_spaces[i].value_type)
            value_buffer_size = int(c_make_data_p.action_spaces[i].value_buffer_size)
            c_buffer_p = c_make_data_p.action_spaces[i].value_buffer

            arr = np.ctypeslib.as_array((CGYM_VALUE_TYPE_TO_CTYPE[value_type] * value_buffer_size).from_address(c_buffer_p))

            space = None

            if value_type == CGYM_VALUE_TYPE_MULTI_DISCRETE:
                space = gym.spaces.MultiDiscrete(arr)
            else:
                space = gym.spaces.Box(arr[:len(arr) // 2], arr[len(arr) // 2:])

            self.action_space[c_make_data_p.action_spaces[i].key] = space

    def step(self, action: gym.core.ActType) -> Tuple[gym.core.ObsType, float, bool, bool, dict]:
        c_actions = None
        num_actions = 1

        if action is int:
            c_action = c_int32(action)
            c_actions = CGym_Key_Value("action", c_int32(CGYM_VALUE_TYPE_INT), c_int32(1), byref(c_action))
        elif action is np.array:
            action = np.ascontiguousarray(action)

            c_actions = CGym_Key_Value("action", c_int32(CGYM_NUMPY_DTYPE_TO_VALUE_TYPE[action.dtype]),
                    c_int32(len(action)), byref(action.data))
        elif action is dict:
            num_actions = len(action)
            
            c_actions = CGym_Key_Value * num_actions

            for k, v in action.items():
                c_actions[i].key = k
                c_actions[i].value_type = c_int32(CGYM_NUMPY_DTYPE_TO_VALUE_TYPE[v.dtype])
                c_actions[i].value_buffer_size = c_int32(len(k))
                c_actions[i].value_buffer = byref(k.data)

        else:
            raise(Exception("Unrecognized action type! Supported are: int, np.array, Dict[np.array]"))
            
        c_step_data_p = self.lib.cgym_step(c_actions, c_int32(num_actions))

        # Create observation
        observation = {}

        for i in range(c_step_data_p.observations_size):
            value_type = int(c_step_data_p.observations[i].value_type)
            value_buffer_size = int(c_step_data_p.observations[i].value_buffer_size)
            c_buffer_p = c_step_data_p.observations[i].value_buffer

            arr = np.ctypeslib.as_array((CGYM_VALUE_TYPE_TO_CTYPE[value_type] * value_buffer_size).from_address(c_buffer_p))

            observation[c_step_data_p.observations[i].key] = arr
        
        info = {}

        for i in range(c_step_data_p.infos_size):
            value_type = int(c_step_data_p.infos[i].value_type)
            value_buffer_size = int(c_step_data_p.infos[i].value_buffer_size)
            c_buffer_p = c_step_data_p.infos[i].value_buffer

            arr = np.ctypeslib.as_array((CGYM_VALUE_TYPE_TO_CTYPE[value_type] * value_buffer_size).from_address(c_buffer_p))

            info[c_step_data_p.infos[i].key] = arr

        reward = float(c_step_data_p.reward)
        terminated = bool(c_step_data_p.terminated)
        truncated = bool(c_step_data_p.truncated)

        return (observation, reward, terminated, truncated, info)

    def reset(self, seed: Optional[int] = None, options: Optional[List[Any]] = None) -> Tuple[gym.core.ObsType, dict]:
        if seed == None:
            seed = 0

        c_options = None

        if options == None:
            c_options = CGym_Option * len(options)

        c_reset_data_p = self.lib.cgym_reset(c_int32(seed), c_options, c_int32(0 if options == None else len(options)))
        
        # Create observation
        observation = {}

        for i in range(c_reset_data_p.observations_size):
            value_type = int(c_reset_data_p.observations[i].value_type)
            value_buffer_size = int(c_reset_data_p.observations[i].value_buffer_size)
            c_buffer_p = c_reset_data_p.observations[i].value_buffer

            arr = np.ctypeslib.as_array((CGYM_VALUE_TYPE_TO_CTYPE[value_type] * value_buffer_size).from_address(c_buffer_p))

            observation[c_reset_data_p.observations[i].key] = arr
        
        info = {}

        for i in range(c_reset_data_p.infos_size):
            value_type = int(c_reset_data_p.infos[i].value_type)
            value_buffer_size = int(c_reset_data_p.infos[i].value_buffer_size)
            c_buffer_p = c_reset_data_p.infos[i].value_buffer

            arr = np.ctypeslib.as_array((CGYM_VALUE_TYPE_TO_CTYPE[value_type] * value_buffer_size).from_address(c_buffer_p))

            info[c_reset_data_p.infos[i].key] = arr

        return (observation, info)

    def render(self) -> gym.core.RenderFrame:
        c_frame_p = self.lib.cgym_render()

        value_type = c_frame_p.value_type
        value_buffer_size = c_frame_p.value_buffer_height * c_frame_p.value_buffer_width * c_frame_p.value_buffer_channels
        c_buffer_p = c_frame_p.value_buffer

        arr = np.ctypeslib.as_array((CGYM_VALUE_TYPE_TO_CTYPE[value_type] * value_buffer_size).from_address(c_buffer_p))

        return arr.reshape(c_frame_p.value_buffer_height, c_frame_p.value_buffer_width, c_frame_p.value_buffer_channels)

    def close(self):
        self.lib.cgym_close()

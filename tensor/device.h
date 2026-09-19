#pragma once

enum class DeviceType {
    CPU,
    GPU
};

struct Device {
    DeviceType type;
    int index = 0;
};
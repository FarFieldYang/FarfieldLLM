#pragma once

class GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);
};

class NoGradGuard {
private:
    bool previous_;

public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};
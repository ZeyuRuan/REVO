#pragma once
#include <chrono>
using namespace std::chrono;
class Timer
{
public:
    Timer();
    static inline high_resolution_clock::time_point getTime()
    {
        return high_resolution_clock::now();
    }
    //returns the time in microseconds
    static inline int getTimeDiffMiS(const high_resolution_clock::time_point& start, const high_resolution_clock::time_point& end)
    {
        return duration_cast<microseconds>(end - start).count();
    }
    //returns the time in milliseconds
    static inline int getTimeDiffMs(const high_resolution_clock::time_point& start, const high_resolution_clock::time_point& end)
    {
        return duration_cast<milliseconds>(end - start).count();
    }
};


#ifndef __TIMER_CUH__
#define __TIMER_CUH__
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <string>
#include <numeric>
#include <iostream>

class Timer {
public:
    Timer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
    };
    ~Timer() {
        if(start_event) cudaEventDestroy(start_event);
        if(end_event) cudaEventDestroy(end_event);
    };

    void start() {
        cudaEventRecord(start_event, nullptr);
    };
    void end() {
        cudaEventRecord(end_event, nullptr);
        cudaEventSynchronize(end_event);
        float time;
        cudaEventElapsedTime(&time, start_event, end_event);
        times.push_back(time / 1000.0); // convert to seconds
    };
    void clear() {
        times.clear();
    };
    // returns the average time in seconds
    // metric: https://en.wikipedia.org/wiki/Mean
    [[nodiscard]] double average() const {
        return std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    };
    // returns the standard deviation in seconds
    // metric: https://en.wikipedia.org/wiki/Standard_deviation
    [[nodiscard]] double stddev() const {
        double avg = average();
        double sdev = static_cast<double>(0.);
        for (auto t : times) {
            sdev += (t - avg) * (t - avg);
        }
        return std::sqrt(sdev / static_cast<double>(times.size()));
    };
    // record mops(milliseconds operations per second), avg time in milliseconds, and standard deviation in milliseconds
    void report(std::uint32_t keys) const {
        double avg = average();
        double mops = keys / 1e6 / avg;
        double sdev = stddev();
        printf("%-12.6f%-12.6f%-12.6f\n", mops, avg * 1000, 1000 * stddev()); 
    };
    [[nodiscard]] std::string to_string(std::uint32_t keys) const {
        double avg = average();
        double mops = keys / 1e6 / avg;
        double sdev = stddev();
        return std::to_string(mops) + " " + std::to_string(avg * 1000) + " " +
               std::to_string(1000 * stddev()) + '\n';
    };

private: 
    std::vector<double> times;
    cudaEvent_t start_event{};
    cudaEvent_t end_event{};
};
#endif // __TIMER_CUH__
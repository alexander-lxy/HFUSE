#ifndef TIMER_H
#define TIMER_H

#include <cuda_runtime.h>
#include <cstdio>

inline void startTimer(cudaEvent_t& start_event, cudaEvent_t& stop_event) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
}

inline void stopTimerAndPrint(const char* kernel_name, cudaEvent_t& start_event, cudaEvent_t& stop_event) {
    float elapsed_time_ms = 0.0f;

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event);

    printf("Kernel '%s' execution time: %.3f ms\n", kernel_name, elapsed_time_ms);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

#endif // TIMER_H
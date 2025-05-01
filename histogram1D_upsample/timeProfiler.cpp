// timer.cpp
#include "timer.h"

// 定义全局计时变量
cudaEvent_t start_event, stop_event;
float elapsed_time_ms = 0.0f;

void startTimer() {
    // 创建事件
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    // 记录起始时间
    cudaEventRecord(start_event, 0);
}

void stopTimerAndPrint(const char* kernel_name) {
    // 记录结束时间
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    
    // 计算经过的时间
    cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event);
    
    // 打印结果
    printf("Kernel '%s' execution time: %.3f ms\n", kernel_name, elapsed_time_ms);
    
    // 清理事件
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

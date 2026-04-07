#ifndef _PERFORMANCE_MONITOR_GLOBAL_H_
#define _PERFORMANCE_MONITOR_GLOBAL_H_

#include "performance_monitor.h"

namespace sim {

// 全局性能监控器访问接口
class GlobalPerfMonitor {
public:
  static void init(int num_devices);
  static void cleanup();
  static PerformanceMonitor* get();

private:
  static PerformanceMonitor* instance_;
};

} // namespace sim

#endif // _PERFORMANCE_MONITOR_GLOBAL_H_

#include "performance_monitor_global.h"

namespace sim {

PerformanceMonitor* GlobalPerfMonitor::instance_ = nullptr;

void GlobalPerfMonitor::init(int num_devices) {
  if (instance_ == nullptr) {
    instance_ = new PerformanceMonitor(num_devices);
  }
}

void GlobalPerfMonitor::cleanup() {
  if (instance_ != nullptr) {
    delete instance_;
    instance_ = nullptr;
  }
}

PerformanceMonitor* GlobalPerfMonitor::get() {
  return instance_;
}

} // namespace sim

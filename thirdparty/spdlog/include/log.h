#ifdef DEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
// #define SPDLOG_DEBUG_ON
#endif

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

/* basic */
constexpr const char* filename = "logs/log.txt";
/* rotating file config */
constexpr int file_size  = 10*1024*1024; // 10M
constexpr int back_count = 5;

/* async sink config */
constexpr int task_count = 1024 * 8;
constexpr int tp_size    = 1;


static inline void init_logger(){
    // create console_sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::warn);
    
    // create rotating file sink
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(filename, file_size, back_count, false);
#ifdef DEBUG
    file_sink->set_level(spdlog::level::debug);
#else
    file_sink->set_level(spdlog::level::info);
#endif
    // sink's bucket
    spdlog::sinks_init_list sinks{console_sink, file_sink};

    // create async logger, and use global threadpool
    spdlog::init_thread_pool(task_count, tp_size);
    auto logger = std::make_shared<spdlog::async_logger>("Adminlogger", sinks, spdlog::thread_pool());

    // ajust level.
#ifdef DEBUG
    logger->set_level(spdlog::level::debug);
#else
    logger->set_level(spdlog::level::info);
#endif

    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
}

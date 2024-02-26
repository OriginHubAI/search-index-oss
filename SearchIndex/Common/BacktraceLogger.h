#pragma once
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

// Print backtraces of specific functions for debugging purposes

template <typename T, std::size_t N>
struct std::hash<std::array<T, N>>
{
    std::size_t operator()(const std::array<T, N> & arr) const
    {
        auto hashT = std::hash<T>();
        std::size_t h = 0;
        for (const auto & elem : arr)
        {
            h ^= hashT(elem) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

template <typename T, std::size_t N>
struct std::hash<std::pair<std::array<T, N>, std::string>>
{
    std::size_t
    operator()(const std::pair<std::array<T, N>, std::string> & arr) const
    {
        return std::hash<std::array<T, N>>()(arr.first)
            ^ std::hash<std::string>()(arr.second);
    }
};

namespace backward
{
class StackTrace;
}

class BacktraceLogger
{
public:
    static const int MAX_BACKTRACE_SIZE = 32;
    using BacktraceArr = std::array<void *, MAX_BACKTRACE_SIZE>;

    BacktraceLogger(const std::string name_) : name(std::move(name_)) { }

    void logBacktrace(const std::string & tag, size_t size);

    void printBacktraces(int64_t num_entries = MAX_BACKTRACE_SIZE);

    static bool
    log(const std::string & name,
        const std::string & tag = "",
        size_t size = -1)
    {
        auto it = name_to_logger.find(name);
        if (it == name_to_logger.end())
            return false;
        it->second->logBacktrace(tag, size);
        return true;
    }

    static void resetLogger(
        const std::string & name, int default_st_size = MAX_BACKTRACE_SIZE)
    {
        name_to_logger[name].reset(new BacktraceLogger(name));
        name_to_logger[name]->default_stacktrace_size = default_st_size;
    }

    static std::shared_ptr<BacktraceLogger> getLogger(const std::string & name)
    {
        auto it = name_to_logger.find(name);
        return it == name_to_logger.end() ? nullptr : it->second;
    }

    static void printLogger(const std::string & name)
    {
        auto logger = getLogger(name);
        if (logger != nullptr)
        {
            printf("## BacktraceLogger %s ##\n", name.c_str());
            logger->printBacktraces();
        }
    }

private:
    std::string name;
    int default_stacktrace_size = MAX_BACKTRACE_SIZE;
    std::unordered_map<std::pair<BacktraceArr, std::string>, uint64_t>
        backtrace_counter;
    std::unordered_map<
        std::pair<BacktraceArr, std::string>,
        std::shared_ptr<backward::StackTrace>>
        backtrace_st;
    inline static std::
        unordered_map<std::string, std::shared_ptr<BacktraceLogger>>
            name_to_logger;
};

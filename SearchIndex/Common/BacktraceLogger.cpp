#include "BacktraceLogger.h"
#include <backward-cpp/backward.hpp>
#include "Utils.h"

void BacktraceLogger::logBacktrace(const std::string & tag, size_t size)
{
    size = size == -1 ? default_stacktrace_size : size;
    void * array[MAX_BACKTRACE_SIZE];
    SI_THROW_IF_NOT(
        size <= MAX_BACKTRACE_SIZE, Search::ErrorCode::LOGICAL_ERROR);
    size_t len = backtrace(array, size);
    BacktraceArr ba;
    std::copy(&array[0], &array[len], &ba[0]);
    std::fill(ba.begin() + len, ba.end(), nullptr);
    auto ba_tag = std::make_pair(ba, tag);
    uint64_t & counter = backtrace_counter[ba_tag];
    counter += 1;
    if (counter == 1)
    {
        // first counter
        backtrace_st[ba_tag].reset(new backward::StackTrace());
        backtrace_st[ba_tag]->load_here(size);
    }
}

void BacktraceLogger::printBacktraces(int64_t num_entries)
{
    auto print_backtrace_addr2line = [this](BacktraceArr & bt)
    {
        size_t size = 0;
        while (size < MAX_BACKTRACE_SIZE && bt[size] != nullptr)
            size += 1;
        char ** strings = backtrace_symbols(bt.data(), size);
        for (int i = 0; i < size; i++)
        {
            printf("\t[bt] #%02d %s\n", i, strings[i]);
            char syscom[256];
            // last parameter is the name of this app
            sprintf(syscom, "addr2line %p -e <prog_name>", bt[i]);
            system(syscom);
        }
        free(strings);
    };

    std::map<uint64_t, std::pair<BacktraceArr, std::string>> c2b;
    for (auto & it : backtrace_counter)
        c2b[it.second] = it.first;
    int idx = 0;
    backward::Printer p;
    p.object = true;
    p.address = true;
    for (auto it = c2b.rbegin(); it != c2b.rend(); ++it)
    {
        // print backtraces ordered by frequency in descending order
        if (idx >= num_entries)
            break;
        printf(
            "[%s] Backtrace #%d, tag=%s, count=%lu\n",
            name.c_str(),
            idx,
            it->second.second.c_str(),
            it->first);
        p.print(*backtrace_st[it->second]);

        // TODO addr2line is too slow
        // print_backtrace_addr2line(it->second);
        ++idx;
    }
}

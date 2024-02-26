#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

namespace Search
{

template <typename T>
class BlockingQueue
{
private:
    std::mutex mut;
    std::queue<T> private_std_queue;
    std::condition_variable cond_not_empty;
    std::condition_variable cond_not_full;
    int count{0}; // Guard with Mutex
    int max_size;

public:
    BlockingQueue(int max_size_) : max_size(max_size_) { }

    void put(T new_value)
    {
        std::unique_lock<std::mutex> lk(mut);
        //Condition takes a unique_lock and waits given the false condition
        cond_not_full.wait(lk, [this] { return count < max_size; });
        private_std_queue.push(new_value);
        count++;
        cond_not_empty.notify_one();
    }
    void take(T & value)
    {
        std::unique_lock<std::mutex> lk(mut);
        //Condition takes a unique_lock and waits given the false condition
        cond_not_empty.wait(lk, [this] { return !private_std_queue.empty(); });
        value = private_std_queue.front();
        private_std_queue.pop();
        count--;
        cond_not_full.notify_one();
    }
};

}

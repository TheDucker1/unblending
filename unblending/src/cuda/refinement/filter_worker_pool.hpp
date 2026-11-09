#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace twostage {
// Persistent worker threads keep CPU filter timing free of per-loop spawn costs.
class Pool {
    std::vector<std::thread> threads;std::mutex mutex;std::condition_variable ready,finished;
    std::function<void(int,int)> job;std::atomic<int> next{0};int size=0,grain=1,pending=0,generation=0;
    bool stopping=false;std::exception_ptr error;
public:
    explicit Pool(int workers){for(int i=0;i<workers;++i)threads.emplace_back([this]{
        int seen=0;for(;;){std::unique_lock<std::mutex> lock(mutex);
            ready.wait(lock,[&]{return stopping||generation!=seen;});if(stopping)return;
            seen=generation;lock.unlock();
            try{for(;;){int begin=next.fetch_add(grain);if(begin>=size)break;job(begin,std::min(begin+grain,size));}}
            catch(...){std::lock_guard<std::mutex> guard(mutex);if(!error)error=std::current_exception();next.store(size);}
            lock.lock();if(--pending==0)finished.notify_one();
        }
    });}
    ~Pool(){{std::lock_guard<std::mutex> guard(mutex);stopping=true;}ready.notify_all();for(auto& t:threads)t.join();}
    void run(int count,int chunk,const std::function<void(int,int)>& f){
        std::unique_lock<std::mutex> lock(mutex);size=count;grain=chunk;next=0;job=f;error=nullptr;
        pending=threads.size();++generation;ready.notify_all();finished.wait(lock,[&]{return pending==0;});
        if(error)std::rethrow_exception(error);
    }
};
}

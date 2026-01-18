# Task Scheduler with Work Stealing - Implementation Guide

## What is This Project?

Build a high-performance task scheduler for parallel computing with work-stealing queues, similar to Intel TBB or C++20's executors. This project teaches concurrent programming, lock-free data structures, and parallel algorithms used in modern multi-core systems.

## Why Build This?

- Master concurrent programming and synchronization
- Learn lock-free data structures
- Implement work-stealing algorithm for load balancing
- Understand task-based parallelism
- Build the foundation of parallel computing frameworks

---

## Architecture Overview

```
┌──────────────────────────────────────────┐
│       Application (Task Graph)           │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐         │
│  │ T1 │→ │ T2 │→ │ T3 │  │ T4 │         │
│  └────┘  └────┘  └────┘  └────┘         │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│         Task Scheduler                   │
│  ┌──────────────────────────────────┐   │
│  │   Dependency Resolution (DAG)    │   │
│  └──────────────────────────────────┘   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│         Thread Pool                      │
│  ┌────────┐  ┌────────┐  ┌────────┐     │
│  │Worker 1│  │Worker 2│  │Worker 3│     │
│  │[Queue] │  │[Queue] │  │[Queue] │     │
│  └────────┘  └────────┘  └────────┘     │
│      ▲           ▲           ▲          │
│      └───Work Stealing───────┘          │
└──────────────────────────────────────────┘
```

---

## Implementation Hints

### 1. Lock-Free Work-Stealing Queue

**What you need:**
Deque where owner pushes/pops from one end, thieves steal from other end.

**Hint:**
```cpp
template<typename T>
class WorkStealingQueue {
private:
    static constexpr size_t INITIAL_SIZE = 1024;

    struct CircularArray {
        std::atomic<T*> tasks[INITIAL_SIZE];
        size_t size = INITIAL_SIZE;
    };

    std::atomic<int64_t> top;    // Owner pushes/pops here
    std::atomic<int64_t> bottom; // Thieves steal here
    std::atomic<CircularArray*> array;

public:
    WorkStealingQueue() {
        array = new CircularArray();
        top.store(0);
        bottom.store(0);
    }

    // Owner pushes task (LIFO)
    void push(T* task) {
        int64_t b = bottom.load(std::memory_order_relaxed);
        int64_t t = top.load(std::memory_order_acquire);

        CircularArray* a = array.load(std::memory_order_relaxed);

        // Check if resize needed
        if (b - t >= (int64_t)a->size - 1) {
            resize();
            a = array.load(std::memory_order_relaxed);
        }

        a->tasks[b % a->size].store(task, std::memory_order_relaxed);

        // Make task visible before incrementing bottom
        std::atomic_thread_fence(std::memory_order_release);
        bottom.store(b + 1, std::memory_order_relaxed);
    }

    // Owner pops task (LIFO)
    T* pop() {
        int64_t b = bottom.load(std::memory_order_relaxed) - 1;
        CircularArray* a = array.load(std::memory_order_relaxed);

        bottom.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        int64_t t = top.load(std::memory_order_relaxed);

        if (t <= b) {
            // Queue not empty
            T* task = a->tasks[b % a->size].load(std::memory_order_relaxed);

            if (t == b) {
                // Last element, race with thieves
                if (!top.compare_exchange_strong(t, t + 1,
                                                std::memory_order_seq_cst,
                                                std::memory_order_relaxed)) {
                    // Lost race, thief got it
                    task = nullptr;
                }
                bottom.store(b + 1, std::memory_order_relaxed);
            }

            return task;
        } else {
            // Queue empty
            bottom.store(b + 1, std::memory_order_relaxed);
            return nullptr;
        }
    }

    // Thief steals task (FIFO)
    T* steal() {
        int64_t t = top.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = bottom.load(std::memory_order_acquire);

        if (t < b) {
            // Queue not empty
            CircularArray* a = array.load(std::memory_order_consume);
            T* task = a->tasks[t % a->size].load(std::memory_order_relaxed);

            if (!top.compare_exchange_strong(t, t + 1,
                                            std::memory_order_seq_cst,
                                            std::memory_order_relaxed)) {
                // Lost race with another thief
                return nullptr;
            }

            return task;
        }

        return nullptr;
    }

private:
    void resize() {
        CircularArray* old_array = array.load(std::memory_order_relaxed);
        CircularArray* new_array = new CircularArray();
        new_array->size = old_array->size * 2;

        int64_t t = top.load(std::memory_order_relaxed);
        int64_t b = bottom.load(std::memory_order_relaxed);

        for (int64_t i = t; i < b; i++) {
            new_array->tasks[i % new_array->size].store(
                old_array->tasks[i % old_array->size].load(std::memory_order_relaxed),
                std::memory_order_relaxed
            );
        }

        array.store(new_array, std::memory_order_release);
    }
};
```

**Tips:**
- Based on Chase-Lev deque algorithm
- Owner operations are almost lock-free
- Thieves use CAS for synchronization
- Handle ABA problem with tagged pointers

### 2. Thread Pool with Work Stealing

**What you need:**
Worker threads that execute tasks and steal from others when idle.

**Hint:**
```cpp
class TaskScheduler {
private:
    struct Worker {
        std::thread thread;
        WorkStealingQueue<Task> queue;
        std::atomic<bool> is_active{true};
        int worker_id;
    };

    std::vector<std::unique_ptr<Worker>> workers;
    std::atomic<int> num_workers;
    std::atomic<bool> shutdown{false};

public:
    TaskScheduler(int num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }

        num_workers = num_threads;

        for (int i = 0; i < num_threads; i++) {
            auto worker = std::make_unique<Worker>();
            worker->worker_id = i;
            worker->thread = std::thread(&TaskScheduler::workerLoop, this, worker.get());
            workers.push_back(std::move(worker));
        }
    }

    void submit(Task* task) {
        // Get current worker (if called from worker thread)
        Worker* current = getCurrentWorker();

        if (current) {
            // Push to own queue
            current->queue.push(task);
        } else {
            // Push to random worker queue
            int idx = rand() % workers.size();
            workers[idx]->queue.push(task);
        }
    }

    void shutdown() {
        shutdown.store(true);

        for (auto& worker : workers) {
            worker->is_active.store(false);
            worker->thread.join();
        }
    }

private:
    void workerLoop(Worker* worker) {
        // Set thread-local worker pointer
        setCurrentWorker(worker);

        while (worker->is_active.load()) {
            Task* task = nullptr;

            // Try to pop from own queue
            task = worker->queue.pop();

            if (!task) {
                // Own queue empty, try stealing
                task = stealTask(worker);
            }

            if (task) {
                // Execute task
                task->execute();
                delete task;
            } else {
                // No work available, backoff
                std::this_thread::yield();
            }
        }
    }

    Task* stealTask(Worker* thief) {
        // Try to steal from random victims
        int num_attempts = workers.size();

        for (int i = 0; i < num_attempts; i++) {
            int victim_id = rand() % workers.size();

            if (victim_id == thief->worker_id) {
                continue; // Don't steal from self
            }

            Task* task = workers[victim_id]->queue.steal();
            if (task) {
                return task;
            }
        }

        return nullptr;
    }

    // Thread-local storage for current worker
    static thread_local Worker* current_worker;

    Worker* getCurrentWorker() {
        return current_worker;
    }

    void setCurrentWorker(Worker* worker) {
        current_worker = worker;
    }
};

thread_local TaskScheduler::Worker* TaskScheduler::current_worker = nullptr;
```

**Tips:**
- Use exponential backoff when idle
- Implement NUMA-aware scheduling
- Add task priority support
- Monitor queue depths for load balancing

### 3. Task with Dependencies (DAG)

**What you need:**
Tasks that depend on other tasks completing first.

**Hint:**
```cpp
class Task {
public:
    virtual void execute() = 0;
    virtual ~Task() = default;

    void addDependency(Task* task) {
        dependencies.push_back(task);
        dependency_count.fetch_add(1);
    }

    void notifyCompleted() {
        // Decrement dependency count of dependent tasks
        for (Task* dependent : dependents) {
            if (dependent->dependency_count.fetch_sub(1) == 1) {
                // Last dependency satisfied, task is ready
                scheduler->submit(dependent);
            }
        }
    }

    bool isReady() {
        return dependency_count.load() == 0;
    }

private:
    std::vector<Task*> dependencies;
    std::vector<Task*> dependents;
    std::atomic<int> dependency_count{0};
    TaskScheduler* scheduler;

    friend class TaskGraph;
};

class TaskGraph {
private:
    std::vector<std::unique_ptr<Task>> tasks;

public:
    void addTask(std::unique_ptr<Task> task) {
        tasks.push_back(std::move(task));
    }

    void addDependency(Task* from, Task* to) {
        to->addDependency(from);
        from->dependents.push_back(to);
    }

    void execute(TaskScheduler& scheduler) {
        // Find tasks with no dependencies
        for (auto& task : tasks) {
            task->scheduler = &scheduler;
            if (task->isReady()) {
                scheduler.submit(task.get());
            }
        }

        // Wait for all tasks to complete
        waitForCompletion();
    }

private:
    void waitForCompletion() {
        // Use condition variable or busy wait
        while (true) {
            bool all_done = true;
            for (auto& task : tasks) {
                if (!task->is_completed) {
                    all_done = false;
                    break;
                }
            }

            if (all_done) break;

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};
```

**Tips:**
- Use topological sort for dependency ordering
- Detect cycles in task graph
- Implement wait/notify for synchronization
- Add task cancellation support

### 4. Fork-Join Parallelism

**What you need:**
Pattern where task spawns subtasks and waits for them.

**Hint:**
```cpp
template<typename T>
class Future {
private:
    std::shared_ptr<std::atomic<bool>> ready;
    std::shared_ptr<T> value;
    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<std::condition_variable> cv;

public:
    Future() :
        ready(std::make_shared<std::atomic<bool>>(false)),
        value(std::make_shared<T>()),
        mutex(std::make_shared<std::mutex>()),
        cv(std::make_shared<std::condition_variable>()) {}

    void set(const T& val) {
        *value = val;
        ready->store(true);
        cv->notify_all();
    }

    T get() {
        // If called from worker, help execute tasks while waiting
        while (!ready->load()) {
            Task* task = scheduler->tryGetTask();
            if (task) {
                task->execute();
            } else {
                std::unique_lock<std::mutex> lock(*mutex);
                cv->wait_for(lock, std::chrono::milliseconds(1));
            }
        }

        return *value;
    }
};

template<typename Func>
Future<typename std::result_of<Func()>::type> async(Func&& f) {
    using ReturnType = typename std::result_of<Func()>::type;

    Future<ReturnType> future;

    auto task = new LambdaTask([f, future]() mutable {
        auto result = f();
        future.set(result);
    });

    scheduler->submit(task);

    return future;
}

// Example: Parallel Fibonacci
int fib(int n) {
    if (n < 2) return n;

    auto f1 = async([n] { return fib(n - 1); });
    auto f2 = async([n] { return fib(n - 2); });

    return f1.get() + f2.get();
}
```

**Tips:**
- Implement continuation passing style
- Add when_all / when_any combinators
- Use promise/future pattern
- Avoid creating tasks for small work

### 5. Parallel Algorithms

**What you need:**
High-level parallel versions of common algorithms.

**Hint:**
```cpp
template<typename Iterator, typename Func>
void parallel_for_each(Iterator begin, Iterator end, Func func) {
    size_t n = std::distance(begin, end);
    size_t chunk_size = 1000; // Minimum work per task

    if (n <= chunk_size) {
        // Sequential fallback
        std::for_each(begin, end, func);
        return;
    }

    size_t num_chunks = (n + chunk_size - 1) / chunk_size;
    std::atomic<size_t> completed{0};

    for (size_t i = 0; i < num_chunks; i++) {
        Iterator chunk_begin = begin + i * chunk_size;
        Iterator chunk_end = std::min(chunk_begin + chunk_size, end);

        auto task = new LambdaTask([chunk_begin, chunk_end, func, &completed] {
            std::for_each(chunk_begin, chunk_end, func);
            completed.fetch_add(1);
        });

        scheduler->submit(task);
    }

    // Wait for completion
    while (completed.load() < num_chunks) {
        std::this_thread::yield();
    }
}

template<typename Iterator, typename T, typename BinaryOp>
T parallel_reduce(Iterator begin, Iterator end, T init, BinaryOp op) {
    size_t n = std::distance(begin, end);
    size_t chunk_size = 1000;

    if (n <= chunk_size) {
        return std::accumulate(begin, end, init, op);
    }

    size_t num_chunks = (n + chunk_size - 1) / chunk_size;
    std::vector<Future<T>> futures;

    for (size_t i = 0; i < num_chunks; i++) {
        Iterator chunk_begin = begin + i * chunk_size;
        Iterator chunk_end = std::min(chunk_begin + chunk_size, end);

        auto future = async([chunk_begin, chunk_end, init, op] {
            return std::accumulate(chunk_begin, chunk_end, init, op);
        });

        futures.push_back(future);
    }

    // Reduce partial results
    T result = init;
    for (auto& future : futures) {
        result = op(result, future.get());
    }

    return result;
}
```

**Tips:**
- Tune chunk size based on work complexity
- Implement parallel sort, scan, filter
- Add SIMD optimization hints
- Support custom execution policies

### 6. Coroutines for Lightweight Tasks

**What you need:**
Suspend/resume tasks without OS threads (C++20 coroutines).

**Hint:**
```cpp
#include <coroutine>

struct Task {
    struct promise_type {
        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() {}
    };

    std::coroutine_handle<promise_type> handle;

    Task(std::coroutine_handle<promise_type> h) : handle(h) {}

    ~Task() {
        if (handle) handle.destroy();
    }

    void resume() {
        if (handle && !handle.done()) {
            handle.resume();
        }
    }

    bool done() {
        return handle.done();
    }
};

struct Awaitable {
    bool await_ready() { return false; }

    void await_suspend(std::coroutine_handle<> handle) {
        // Schedule continuation
        scheduler->submit(new ResumeTask(handle));
    }

    void await_resume() {}
};

// Example coroutine
Task processData() {
    // Do some work
    co_await Awaitable{}; // Suspend point

    // Continue after resume
    co_await Awaitable{}; // Another suspend point

    // Done
}
```

**Tips:**
- Use coroutines for I/O-bound tasks
- Implement async/await pattern
- Add coroutine scheduling to work queues
- Support cancellation tokens

---

## Project Structure

```
10_task_scheduler/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── queue/
│   │   └── work_stealing_queue.cpp
│   ├── scheduler/
│   │   ├── task_scheduler.cpp
│   │   ├── worker.cpp
│   │   └── task.cpp
│   ├── graph/
│   │   ├── task_graph.cpp
│   │   └── dag.cpp
│   ├── parallel/
│   │   ├── parallel_for.cpp
│   │   ├── parallel_reduce.cpp
│   │   └── parallel_sort.cpp
│   └── coroutine/
│       └── coroutine_scheduler.cpp
├── tests/
│   ├── test_queue.cpp
│   ├── test_scheduler.cpp
│   └── test_parallel.cpp
├── benchmarks/
│   ├── fibonacci_bench.cpp
│   ├── parallel_sort_bench.cpp
│   └── throughput_bench.cpp
└── examples/
    ├── quicksort.cpp
    ├── matrix_multiply.cpp
    └── raytracer.cpp
```

---

## Performance Goals

- **Task Throughput**: 1M+ tasks/sec
- **Latency**: < 100ns to submit task
- **Scalability**: Linear up to 16 cores
- **Overhead**: < 5% vs manual threading

---

## Testing

```cpp
// Benchmark against Intel TBB
tbb::parallel_for(0, n, [&](int i) {
    data[i] = process(data[i]);
});

// Your scheduler
parallel_for_each(data.begin(), data.end(), [](auto& x) {
    x = process(x);
});

// Compare execution time
```

---

## Resources

- Paper: "Dynamic Circular Work-Stealing Deque" (Chase & Lev)
- [Intel TBB Documentation](https://oneapi-src.github.io/oneTBB/)
- [C++ Concurrency in Action](https://www.manning.com/books/c-plus-plus-concurrency-in-action-second-edition)
- [Cilk Work-Stealing Scheduler](http://supertech.csail.mit.edu/cilk/)
- [C++20 Coroutines](https://en.cppreference.com/w/cpp/language/coroutines)

Good luck building your task scheduler!

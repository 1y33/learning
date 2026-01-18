# Container Runtime (Docker-like) - Implementation Guide

## What is This Project?

Build a lightweight container runtime using Linux kernel features like namespaces and cgroups. This project teaches operating systems concepts, process isolation, resource management, and container technologies that power modern cloud infrastructure.

## Why Build This?

- Understand how containers actually work under the hood
- Master Linux kernel features (namespaces, cgroups, capabilities)
- Learn process isolation and sandboxing
- Implement overlay filesystems
- Build the foundation of cloud-native applications

---

## Architecture Overview

```
┌──────────────────────────────────────┐
│        Container CLI Tool            │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│    Container Runtime Manager         │
│  ┌──────────┐  ┌─────────────────┐  │
│  │ Lifecycle│  │  Image Manager  │  │
│  └──────────┘  └─────────────────┘  │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│      Linux Kernel Features           │
│  ┌────────┐ ┌────────┐ ┌─────────┐  │
│  │Namespcs│ │Cgroups │ │CapabilX │  │
│  └────────┘ └────────┘ └─────────┘  │
└──────────────────────────────────────┘
```

---

## Implementation Hints

### 1. Process Namespaces for Isolation

**What you need:**
Isolate processes using PID, Mount, Network, UTS, IPC, and User namespaces.

**Hint:**
```cpp
#include <sched.h>
#include <unistd.h>
#include <sys/wait.h>

class NamespaceManager {
public:
    struct ContainerConfig {
        bool new_pid_ns = true;
        bool new_mnt_ns = true;
        bool new_net_ns = true;
        bool new_uts_ns = true;
        bool new_ipc_ns = true;
        bool new_user_ns = false;
    };

    pid_t createContainer(const ContainerConfig& config,
                          std::function<int()> container_init) {
        int flags = SIGCHLD;

        if (config.new_pid_ns)  flags |= CLONE_NEWPID;
        if (config.new_mnt_ns)  flags |= CLONE_NEWNS;
        if (config.new_net_ns)  flags |= CLONE_NEWNET;
        if (config.new_uts_ns)  flags |= CLONE_NEWUTS;
        if (config.new_ipc_ns)  flags |= CLONE_NEWIPC;
        if (config.new_user_ns) flags |= CLONE_NEWUSER;

        // Stack for child process
        const size_t STACK_SIZE = 1024 * 1024; // 1MB
        char* stack = new char[STACK_SIZE];
        char* stack_top = stack + STACK_SIZE;

        // Clone with namespaces
        pid_t child_pid = clone(
            childFunction,
            stack_top,
            flags,
            &container_init
        );

        if (child_pid == -1) {
            perror("clone");
            return -1;
        }

        return child_pid;
    }

private:
    static int childFunction(void* arg) {
        auto* init_func = static_cast<std::function<int()>*>(arg);
        return (*init_func)();
    }
};

// Usage example
void runContainer() {
    NamespaceManager::ContainerConfig config;

    auto init = []() -> int {
        // This code runs inside container
        std::cout << "Container PID: " << getpid() << std::endl; // Will be 1

        // Set hostname (isolated due to UTS namespace)
        sethostname("mycontainer", 11);

        // Execute shell
        execl("/bin/sh", "/bin/sh", nullptr);
        return 0;
    };

    pid_t container_pid = NamespaceManager().createContainer(config, init);
    waitpid(container_pid, nullptr, 0);
}
```

**Tips:**
- PID namespace: Process sees itself as PID 1
- Mount namespace: Separate filesystem view
- Network namespace: Isolated network stack
- UTS namespace: Separate hostname
- User namespace: Map UIDs/GIDs for security
- Use `/proc/<pid>/ns/` to inspect namespaces

### 2. cgroups for Resource Limiting

**What you need:**
Limit CPU, memory, I/O usage for containers.

**Hint:**
```cpp
class CgroupManager {
private:
    std::string cgroup_name;
    std::string cgroup_path;

public:
    CgroupManager(const std::string& name) : cgroup_name(name) {
        cgroup_path = "/sys/fs/cgroup/myruntime/" + name;
    }

    bool createCgroup() {
        // Create cgroup directory
        if (mkdir(cgroup_path.c_str(), 0755) != 0) {
            if (errno != EEXIST) {
                perror("mkdir cgroup");
                return false;
            }
        }
        return true;
    }

    bool setMemoryLimit(size_t bytes) {
        std::string mem_limit_path = cgroup_path + "/memory.max";
        std::ofstream file(mem_limit_path);
        if (!file.is_open()) return false;

        file << bytes;
        return true;
    }

    bool setCPULimit(int cpu_shares) {
        // cpu.weight: 1-10000 (default 100)
        std::string cpu_weight_path = cgroup_path + "/cpu.weight";
        std::ofstream file(cpu_weight_path);
        if (!file.is_open()) return false;

        file << cpu_shares;
        return true;
    }

    bool setCPUQuota(int period_us, int quota_us) {
        // Limit CPU time: quota_us per period_us
        // Example: 50000/100000 = 50% of one CPU

        std::ofstream max_file(cgroup_path + "/cpu.max");
        if (!max_file.is_open()) return false;

        max_file << quota_us << " " << period_us;
        return true;
    }

    bool setPidsLimit(int max_pids) {
        std::string pids_max_path = cgroup_path + "/pids.max";
        std::ofstream file(pids_max_path);
        if (!file.is_open()) return false;

        file << max_pids;
        return true;
    }

    bool attachProcess(pid_t pid) {
        // Add process to cgroup
        std::string procs_path = cgroup_path + "/cgroup.procs";
        std::ofstream file(procs_path);
        if (!file.is_open()) return false;

        file << pid;
        return true;
    }

    void removeCgroup() {
        rmdir(cgroup_path.c_str());
    }
};

// Usage
void limitContainer(pid_t container_pid) {
    CgroupManager cgroup("container_" + std::to_string(container_pid));

    cgroup.createCgroup();
    cgroup.setMemoryLimit(512 * 1024 * 1024); // 512MB
    cgroup.setCPUQuota(100000, 50000);        // 50% of 1 CPU
    cgroup.setPidsLimit(100);                  // Max 100 processes

    cgroup.attachProcess(container_pid);
}
```

**Tips:**
- cgroups v2 is the modern interface
- Memory: `memory.max`, `memory.high`, `memory.swap.max`
- CPU: `cpu.max`, `cpu.weight`
- I/O: `io.max`, `io.weight`
- Monitor usage via `memory.current`, `cpu.stat`

### 3. Overlay Filesystem for Images

**What you need:**
Layered filesystem where image layers are read-only, container has writable layer.

**Hint:**
```cpp
class OverlayFS {
private:
    std::string lower_dirs;  // Image layers (read-only)
    std::string upper_dir;   // Container layer (read-write)
    std::string work_dir;    // OverlayFS work directory
    std::string merged_dir;  // Final mounted view

public:
    bool mount() {
        // Create directories
        std::filesystem::create_directories(upper_dir);
        std::filesystem::create_directories(work_dir);
        std::filesystem::create_directories(merged_dir);

        // Mount overlay
        std::string mount_opts =
            "lowerdir=" + lower_dirs + ","
            "upperdir=" + upper_dir + ","
            "workdir=" + work_dir;

        if (::mount("overlay", merged_dir.c_str(), "overlay", 0,
                    mount_opts.c_str()) != 0) {
            perror("mount overlay");
            return false;
        }

        return true;
    }

    bool unmount() {
        if (umount(merged_dir.c_str()) != 0) {
            perror("umount overlay");
            return false;
        }
        return true;
    }
};

// Example: 3-layer image
void setupContainerFS(const std::string& container_id) {
    OverlayFS overlay;
    overlay.lower_dirs = "/var/lib/images/ubuntu/layer1:"
                        "/var/lib/images/ubuntu/layer2:"
                        "/var/lib/images/ubuntu/layer3";
    overlay.upper_dir = "/var/lib/containers/" + container_id + "/upper";
    overlay.work_dir = "/var/lib/containers/" + container_id + "/work";
    overlay.merged_dir = "/var/lib/containers/" + container_id + "/merged";

    overlay.mount();
}
```

**Tips:**
- Lower layers are shared between containers
- Upper layer is unique per container
- Use `rsync` or `tar` to extract image layers
- Implement copy-on-write efficiently

### 4. Container Lifecycle Management

**What you need:**
Create, start, stop, kill, delete containers.

**Hint:**
```cpp
enum class ContainerState {
    CREATED,
    RUNNING,
    PAUSED,
    STOPPED,
    DELETED
};

struct Container {
    std::string id;
    std::string image;
    pid_t pid;
    ContainerState state;
    std::string rootfs_path;
    std::unique_ptr<CgroupManager> cgroup;
};

class ContainerRuntime {
private:
    std::map<std::string, Container> containers;

public:
    std::string createContainer(const std::string& image,
                                const std::vector<std::string>& cmd) {
        std::string container_id = generateID();

        Container container;
        container.id = container_id;
        container.image = image;
        container.state = ContainerState::CREATED;
        container.rootfs_path = "/var/lib/containers/" + container_id + "/merged";

        // Setup filesystem
        setupContainerFS(container_id);

        // Setup cgroup
        container.cgroup = std::make_unique<CgroupManager>(container_id);
        container.cgroup->createCgroup();

        containers[container_id] = std::move(container);

        return container_id;
    }

    bool startContainer(const std::string& container_id) {
        auto it = containers.find(container_id);
        if (it == containers.end()) return false;

        auto& container = it->second;

        // Create namespaces and run container
        pid_t pid = fork();
        if (pid == 0) {
            // Child process: set up and exec

            // Unshare namespaces
            unshare(CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWNET |
                   CLONE_NEWUTS | CLONE_NEWIPC);

            // Change root
            chroot(container.rootfs_path.c_str());
            chdir("/");

            // Mount proc
            mount("proc", "/proc", "proc", 0, nullptr);

            // Execute command
            execl("/bin/sh", "/bin/sh", nullptr);
            exit(1);
        }

        // Parent process
        container.pid = pid;
        container.state = ContainerState::RUNNING;

        // Attach to cgroup
        container.cgroup->attachProcess(pid);

        return true;
    }

    bool stopContainer(const std::string& container_id) {
        auto it = containers.find(container_id);
        if (it == containers.end()) return false;

        auto& container = it->second;

        // Send SIGTERM
        kill(container.pid, SIGTERM);

        // Wait for graceful shutdown (10 seconds)
        for (int i = 0; i < 10; i++) {
            if (waitpid(container.pid, nullptr, WNOHANG) > 0) {
                container.state = ContainerState::STOPPED;
                return true;
            }
            sleep(1);
        }

        // Force kill
        kill(container.pid, SIGKILL);
        waitpid(container.pid, nullptr, 0);

        container.state = ContainerState::STOPPED;
        return true;
    }

    bool deleteContainer(const std::string& container_id) {
        auto it = containers.find(container_id);
        if (it == containers.end()) return false;

        auto& container = it->second;

        if (container.state == ContainerState::RUNNING) {
            stopContainer(container_id);
        }

        // Cleanup filesystem
        OverlayFS overlay;
        // ... unmount and delete directories

        // Cleanup cgroup
        container.cgroup->removeCgroup();

        containers.erase(it);
        return true;
    }

private:
    std::string generateID() {
        // Generate random container ID
        return "c" + std::to_string(rand());
    }
};
```

**Tips:**
- Store container metadata in JSON files
- Implement pause/unpause with `SIGSTOP`/`SIGCONT`
- Add container logs (capture stdout/stderr)
- Support exec into running containers

### 5. Network Bridge for Container Networking

**What you need:**
Virtual network bridge to connect containers.

**Hint:**
```cpp
class NetworkManager {
public:
    bool createBridge(const std::string& bridge_name) {
        // Create bridge using ip command
        std::string cmd = "ip link add " + bridge_name + " type bridge";
        system(cmd.c_str());

        cmd = "ip link set " + bridge_name + " up";
        system(cmd.c_str());

        // Assign IP to bridge
        cmd = "ip addr add 172.17.0.1/16 dev " + bridge_name;
        system(cmd.c_str());

        return true;
    }

    bool setupContainerNetwork(const std::string& container_id, pid_t container_pid) {
        std::string veth_host = "veth" + container_id.substr(0, 8);
        std::string veth_container = "veth_c";

        // Create veth pair
        std::string cmd = "ip link add " + veth_host +
                         " type veth peer name " + veth_container;
        system(cmd.c_str());

        // Move container end to container's network namespace
        cmd = "ip link set " + veth_container +
              " netns /proc/" + std::to_string(container_pid) + "/ns/net";
        system(cmd.c_str());

        // Attach host end to bridge
        cmd = "ip link set " + veth_host + " master docker0";
        system(cmd.c_str());

        cmd = "ip link set " + veth_host + " up";
        system(cmd.c_str());

        // Inside container namespace: configure interface
        cmd = "nsenter -t " + std::to_string(container_pid) + " -n " +
              "ip addr add 172.17.0.2/16 dev " + veth_container;
        system(cmd.c_str());

        cmd = "nsenter -t " + std::to_string(container_pid) + " -n " +
              "ip link set " + veth_container + " up";
        system(cmd.c_str());

        cmd = "nsenter -t " + std::to_string(container_pid) + " -n " +
              "ip route add default via 172.17.0.1";
        system(cmd.c_str());

        return true;
    }

    bool enableNAT() {
        // Enable IP forwarding
        system("echo 1 > /proc/sys/net/ipv4/ip_forward");

        // Set up iptables NAT
        system("iptables -t nat -A POSTROUTING -s 172.17.0.0/16 -j MASQUERADE");

        return true;
    }
};
```

**Tips:**
- Use veth pairs for container-to-host networking
- Implement port forwarding with iptables
- Support custom networks and DNS
- Add network isolation between containers

### 6. Image Management

**What you need:**
Pull, store, and manage container images.

**Hint:**
```cpp
struct ImageLayer {
    std::string digest;  // SHA256 hash
    std::string path;
    size_t size;
};

struct Image {
    std::string name;
    std::string tag;
    std::vector<ImageLayer> layers;
    std::string config_digest;
};

class ImageManager {
private:
    std::string image_store_path = "/var/lib/images";
    std::map<std::string, Image> images;

public:
    bool pullImage(const std::string& image_name) {
        // 1. Parse image name (e.g., "ubuntu:20.04")
        // 2. Download manifest from registry
        // 3. Download each layer (if not cached)
        // 4. Extract layers to storage

        // Simplified: Download tarball and extract
        std::string url = "https://registry.hub.docker.com/...";
        std::string tarball = "/tmp/image.tar";

        downloadFile(url, tarball);
        extractLayers(tarball, image_name);

        return true;
    }

private:
    void extractLayers(const std::string& tarball,
                      const std::string& image_name) {
        std::string layer_dir = image_store_path + "/" + image_name;
        std::filesystem::create_directories(layer_dir);

        // Extract with tar
        std::string cmd = "tar -xf " + tarball + " -C " + layer_dir;
        system(cmd.c_str());

        // Parse manifest.json to get layer order
        // Store layer metadata
    }
};
```

**Tips:**
- Implement Docker Registry v2 API
- Use content-addressable storage (SHA256 digests)
- Support image caching and deduplication
- Implement Dockerfile parser and builder

---

## Project Structure

```
06_container_runtime/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── runtime/
│   │   ├── container.cpp
│   │   ├── namespace_manager.cpp
│   │   └── cgroup_manager.cpp
│   ├── storage/
│   │   ├── overlay_fs.cpp
│   │   └── image_manager.cpp
│   ├── network/
│   │   ├── bridge.cpp
│   │   └── veth.cpp
│   └── cli/
│       └── commands.cpp
├── tests/
│   ├── test_namespace.cpp
│   ├── test_cgroup.cpp
│   └── test_overlay.cpp
└── examples/
    └── run_alpine.cpp
```

---

## Testing

Requires Linux with kernel 4.8+ and root privileges:

```bash
# Test namespace isolation
sudo ./container run alpine /bin/sh

# Test resource limits
sudo ./container run --memory 100m --cpus 0.5 alpine stress

# Test networking
sudo ./container run -p 8080:80 nginx
```

---

## Resources

- [Linux Namespaces Man Page](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [Cgroups v2 Documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
- [OCI Runtime Specification](https://github.com/opencontainers/runtime-spec)
- [Docker Source Code](https://github.com/moby/moby)
- Book: "Container Security" by Liz Rice

Good luck building your container runtime!

# BitTorrent Client - Implementation Guide

## What is This Project?

Build a peer-to-peer file sharing client that implements the BitTorrent protocol. This project teaches distributed systems, network programming, cryptographic hashing, and P2P architectures used by millions for file distribution.

## Why Build This?

- Learn peer-to-peer networking
- Master the BitTorrent protocol
- Implement distributed algorithms (piece selection, choking)
- Work with binary protocols and bencode
- Build something that can download real torrents

---

## Architecture Overview

```
┌──────────────────────────────────────────┐
│          Torrent File (.torrent)         │
│  ┌────────────┐  ┌──────────────────┐   │
│  │ Metainfo   │  │   Tracker URL    │   │
│  └────────────┘  └──────────────────┘   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│         Tracker Communication            │
│  ┌────────────┐  ┌──────────────────┐   │
│  │ HTTP/UDP   │  │  Peer Discovery  │   │
│  └────────────┘  └──────────────────┘   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│        Peer Wire Protocol                │
│  ┌────────────┐  ┌──────────────────┐   │
│  │ Handshake  │  │  Message Exchange│   │
│  └────────────┘  └──────────────────┘   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│      Piece Selection & Download          │
│  ┌────────────┐  ┌──────────────────┐   │
│  │ Rarest     │  │   Choking Algo   │   │
│  │ First      │  │                  │   │
│  └────────────┘  └──────────────────┘   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│         File Assembly & Verification     │
└──────────────────────────────────────────┘
```

---

## Implementation Hints

### 1. Torrent File Parser (Bencode)

**What you need:**
Parse `.torrent` files which use bencode encoding.

**Hint:**
```cpp
// Bencode types: integers, strings, lists, dictionaries
// Examples:
// i42e → 42
// 4:spam → "spam"
// l4:spam4:eggse → ["spam", "eggs"]
// d3:cow3:moo4:spam4:eggse → {"cow": "moo", "spam": "eggs"}

class BencodeParser {
public:
    struct Value {
        enum Type { INTEGER, STRING, LIST, DICT };
        Type type;

        int64_t integer;
        std::string string;
        std::vector<Value> list;
        std::map<std::string, Value> dict;
    };

    Value parse(const std::string& data) {
        size_t pos = 0;
        return parseValue(data, pos);
    }

private:
    Value parseValue(const std::string& data, size_t& pos) {
        char ch = data[pos];

        if (ch == 'i') {
            return parseInt(data, pos);
        } else if (ch == 'l') {
            return parseList(data, pos);
        } else if (ch == 'd') {
            return parseDict(data, pos);
        } else if (isdigit(ch)) {
            return parseString(data, pos);
        }

        throw std::runtime_error("Invalid bencode");
    }

    Value parseInt(const std::string& data, size_t& pos) {
        pos++; // Skip 'i'
        size_t end = data.find('e', pos);
        int64_t value = std::stoll(data.substr(pos, end - pos));
        pos = end + 1;

        Value v;
        v.type = Value::INTEGER;
        v.integer = value;
        return v;
    }

    Value parseString(const std::string& data, size_t& pos) {
        size_t colon = data.find(':', pos);
        int length = std::stoi(data.substr(pos, colon - pos));
        pos = colon + 1;

        std::string str = data.substr(pos, length);
        pos += length;

        Value v;
        v.type = Value::STRING;
        v.string = str;
        return v;
    }

    Value parseList(const std::string& data, size_t& pos) {
        pos++; // Skip 'l'

        Value v;
        v.type = Value::LIST;

        while (data[pos] != 'e') {
            v.list.push_back(parseValue(data, pos));
        }
        pos++; // Skip 'e'

        return v;
    }

    Value parseDict(const std::string& data, size_t& pos) {
        pos++; // Skip 'd'

        Value v;
        v.type = Value::DICT;

        while (data[pos] != 'e') {
            Value key = parseString(data, pos);
            Value value = parseValue(data, pos);
            v.dict[key.string] = value;
        }
        pos++; // Skip 'e'

        return v;
    }
};

struct TorrentInfo {
    std::string announce;      // Tracker URL
    std::string name;          // File/directory name
    int64_t piece_length;      // Bytes per piece
    std::vector<uint8_t> pieces; // SHA1 hashes (20 bytes each)
    int64_t length;            // Total size (single file)
    std::vector<FileInfo> files; // Multi-file torrents

    struct FileInfo {
        int64_t length;
        std::vector<std::string> path;
    };

    std::vector<uint8_t> info_hash; // SHA1 of info dict
};

TorrentInfo parseTorrent(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());

    BencodeParser parser;
    auto root = parser.parse(data);

    TorrentInfo info;
    info.announce = root.dict["announce"].string;

    auto& info_dict = root.dict["info"];
    info.name = info_dict.dict["name"].string;
    info.piece_length = info_dict.dict["piece length"].integer;
    info.pieces = std::vector<uint8_t>(
        info_dict.dict["pieces"].string.begin(),
        info_dict.dict["pieces"].string.end()
    );

    // Compute info_hash (SHA1 of bencoded info dict)
    // ... extract and hash info dictionary

    return info;
}
```

**Tips:**
- Use SHA1 for info_hash computation
- Handle both single-file and multi-file torrents
- Support announce-list for multiple trackers
- Validate piece hashes during download

### 2. Tracker Communication

**What you need:**
Contact tracker to get list of peers.

**Hint:**
```cpp
struct TrackerRequest {
    std::vector<uint8_t> info_hash;
    std::vector<uint8_t> peer_id;    // 20 random bytes
    int64_t uploaded = 0;
    int64_t downloaded = 0;
    int64_t left;
    std::string event = "started"; // started, stopped, completed
    uint16_t port = 6881;
};

struct Peer {
    std::string ip;
    uint16_t port;
    std::vector<uint8_t> peer_id;
};

struct TrackerResponse {
    int interval;              // Seconds to wait before next request
    int min_interval;
    std::vector<Peer> peers;
    int complete;              // Number of seeders
    int incomplete;            // Number of leechers
};

class TrackerClient {
public:
    TrackerResponse announce(const std::string& tracker_url,
                            const TrackerRequest& req) {
        // Build HTTP GET request
        std::string url = tracker_url + "?"
            "info_hash=" + urlEncode(req.info_hash) + "&"
            "peer_id=" + urlEncode(req.peer_id) + "&"
            "port=" + std::to_string(req.port) + "&"
            "uploaded=" + std::to_string(req.uploaded) + "&"
            "downloaded=" + std::to_string(req.downloaded) + "&"
            "left=" + std::to_string(req.left) + "&"
            "event=" + req.event + "&"
            "compact=1"; // Request compact peer list

        // Send HTTP request
        std::string response = httpGet(url);

        // Parse bencode response
        BencodeParser parser;
        auto data = parser.parse(response);

        TrackerResponse resp;
        resp.interval = data.dict["interval"].integer;

        // Parse compact peer list (6 bytes per peer)
        std::string peers_data = data.dict["peers"].string;
        for (size_t i = 0; i < peers_data.size(); i += 6) {
            Peer peer;
            uint8_t* bytes = (uint8_t*)&peers_data[i];

            peer.ip = std::to_string(bytes[0]) + "." +
                     std::to_string(bytes[1]) + "." +
                     std::to_string(bytes[2]) + "." +
                     std::to_string(bytes[3]);

            peer.port = (bytes[4] << 8) | bytes[5];

            resp.peers.push_back(peer);
        }

        return resp;
    }

private:
    std::string urlEncode(const std::vector<uint8_t>& data) {
        std::ostringstream escaped;
        escaped << std::hex;

        for (uint8_t byte : data) {
            escaped << '%' << std::setw(2) << std::setfill('0')
                   << (int)byte;
        }

        return escaped.str();
    }
};
```

**Tips:**
- Support both HTTP and UDP trackers
- Implement periodic tracker announces
- Handle tracker failures gracefully
- Add DHT for decentralized peer discovery

### 3. Peer Wire Protocol

**What you need:**
Communicate with peers using BitTorrent message protocol.

**Hint:**
```cpp
enum class MessageType : uint8_t {
    CHOKE = 0,
    UNCHOKE = 1,
    INTERESTED = 2,
    NOT_INTERESTED = 3,
    HAVE = 4,
    BITFIELD = 5,
    REQUEST = 6,
    PIECE = 7,
    CANCEL = 8
};

struct Message {
    MessageType type;
    std::vector<uint8_t> payload;
};

class PeerConnection {
private:
    int socket_fd;
    std::vector<uint8_t> info_hash;
    std::vector<uint8_t> peer_id;

    bool am_choking = true;
    bool am_interested = false;
    bool peer_choking = true;
    bool peer_interested = false;

    std::vector<bool> peer_pieces; // Which pieces peer has

public:
    bool connect(const std::string& ip, uint16_t port) {
        socket_fd = socket(AF_INET, SOCK_STREAM, 0);

        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

        if (::connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            return false;
        }

        return handshake();
    }

    bool handshake() {
        // BitTorrent handshake:
        // <pstrlen><pstr><reserved><info_hash><peer_id>
        // pstrlen = 19, pstr = "BitTorrent protocol"

        std::vector<uint8_t> handshake_msg(68);
        handshake_msg[0] = 19;
        memcpy(&handshake_msg[1], "BitTorrent protocol", 19);
        // handshake_msg[20..27] = reserved (zeros)
        memcpy(&handshake_msg[28], info_hash.data(), 20);
        memcpy(&handshake_msg[48], peer_id.data(), 20);

        send(socket_fd, handshake_msg.data(), 68, 0);

        // Receive handshake response
        uint8_t response[68];
        if (recv(socket_fd, response, 68, 0) != 68) {
            return false;
        }

        // Verify info_hash matches
        if (memcmp(&response[28], info_hash.data(), 20) != 0) {
            return false;
        }

        return true;
    }

    Message receiveMessage() {
        // Read length prefix (4 bytes)
        uint32_t length;
        recv(socket_fd, &length, 4, 0);
        length = ntohl(length);

        if (length == 0) {
            // Keep-alive message
            return Message{};
        }

        // Read message type
        uint8_t type;
        recv(socket_fd, &type, 1, 0);

        // Read payload
        std::vector<uint8_t> payload(length - 1);
        recv(socket_fd, payload.data(), length - 1, 0);

        return Message{static_cast<MessageType>(type), payload};
    }

    void sendMessage(const Message& msg) {
        uint32_t length = htonl(1 + msg.payload.size());
        send(socket_fd, &length, 4, 0);
        send(socket_fd, &msg.type, 1, 0);
        send(socket_fd, msg.payload.data(), msg.payload.size(), 0);
    }

    void sendInterested() {
        am_interested = true;
        sendMessage({MessageType::INTERESTED, {}});
    }

    void requestPiece(uint32_t index, uint32_t begin, uint32_t length) {
        std::vector<uint8_t> payload(12);
        *reinterpret_cast<uint32_t*>(&payload[0]) = htonl(index);
        *reinterpret_cast<uint32_t*>(&payload[4]) = htonl(begin);
        *reinterpret_cast<uint32_t*>(&payload[8]) = htonl(length);

        sendMessage({MessageType::REQUEST, payload});
    }
};
```

**Tips:**
- Handle message framing carefully
- Implement pipelining (multiple requests in-flight)
- Add timeout for slow peers
- Support Fast Extension (BEP 6)

### 4. Piece Selection Algorithm (Rarest First)

**What you need:**
Decide which pieces to download for optimal swarm health.

**Hint:**
```cpp
class PieceSelector {
private:
    int num_pieces;
    std::vector<bool> have_pieces;
    std::map<int, int> piece_frequency; // piece_index → count of peers with it

public:
    int selectPiece(const std::vector<int>& available_pieces) {
        // Rarest first: Choose piece that fewest peers have

        int rarest_piece = -1;
        int min_frequency = INT_MAX;

        for (int piece : available_pieces) {
            if (!have_pieces[piece]) {
                int freq = piece_frequency[piece];
                if (freq < min_frequency) {
                    min_frequency = freq;
                    rarest_piece = piece;
                }
            }
        }

        return rarest_piece;
    }

    int selectPieceEndgame(const std::vector<int>& in_progress) {
        // Endgame mode: Request all remaining pieces from all peers
        // to finish quickly
        for (int i = 0; i < num_pieces; i++) {
            if (!have_pieces[i] &&
                std::find(in_progress.begin(), in_progress.end(), i) == in_progress.end()) {
                return i;
            }
        }
        return -1;
    }

    void updatePeerBitfield(const std::vector<bool>& peer_pieces) {
        for (int i = 0; i < num_pieces; i++) {
            if (peer_pieces[i]) {
                piece_frequency[i]++;
            }
        }
    }
};
```

**Tips:**
- Start with random piece for new downloads
- Use strict priority for partial pieces
- Implement endgame mode for final pieces
- Add sequential mode for video streaming

### 5. Choking Algorithm

**What you need:**
Decide which peers to upload to (incentivize reciprocation).

**Hint:**
```cpp
class ChokingManager {
private:
    struct PeerStats {
        int peer_id;
        uint64_t downloaded_from;
        uint64_t uploaded_to;
        bool is_interested;
        bool is_choked;
    };

    std::vector<PeerStats> peers;
    std::chrono::time_point<std::chrono::steady_clock> last_optimistic_unchoke;

public:
    void updateChoking() {
        // Unchoke 4 best uploaders every 10 seconds
        // + 1 random peer (optimistic unchoke) every 30 seconds

        // Sort by download rate
        std::sort(peers.begin(), peers.end(), [](const auto& a, const auto& b) {
            return a.downloaded_from > b.downloaded_from;
        });

        // Unchoke top 4
        for (int i = 0; i < std::min(4, (int)peers.size()); i++) {
            if (peers[i].is_interested) {
                unchokePeer(peers[i].peer_id);
                peers[i].is_choked = false;
            }
        }

        // Choke the rest
        for (int i = 4; i < peers.size(); i++) {
            chokePeer(peers[i].peer_id);
            peers[i].is_choked = true;
        }

        // Optimistic unchoke
        auto now = std::chrono::steady_clock::now();
        if (now - last_optimistic_unchoke > std::chrono::seconds(30)) {
            int random_idx = rand() % peers.size();
            unchokePeer(peers[random_idx].peer_id);
            last_optimistic_unchoke = now;
        }
    }
};
```

**Tips:**
- Seeders use different algorithm (upload rate)
- Implement anti-snubbing
- Track peer performance metrics
- Use super-seeding for rare content

### 6. SHA1 Verification

**What you need:**
Verify downloaded pieces match expected hash.

**Hint:**
```cpp
#include <openssl/sha.h>

bool verifyPiece(const std::vector<uint8_t>& piece_data,
                const std::vector<uint8_t>& expected_hash) {
    uint8_t computed_hash[SHA_DIGEST_LENGTH];
    SHA1(piece_data.data(), piece_data.size(), computed_hash);

    return memcmp(computed_hash, expected_hash.data(), SHA_DIGEST_LENGTH) == 0;
}
```

**Tips:**
- Verify pieces immediately after download
- Discard and re-request invalid pieces
- Ban peers that send bad data repeatedly
- Track piece verification failures

---

## Project Structure

```
08_bittorrent_client/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── bencode/
│   │   ├── parser.cpp
│   │   └── encoder.cpp
│   ├── tracker/
│   │   ├── http_tracker.cpp
│   │   └── udp_tracker.cpp
│   ├── peer/
│   │   ├── peer_connection.cpp
│   │   ├── peer_manager.cpp
│   │   └── protocol.cpp
│   ├── piece/
│   │   ├── piece_selector.cpp
│   │   └── piece_manager.cpp
│   ├── choking/
│   │   └── choking_manager.cpp
│   └── file/
│       ├── file_manager.cpp
│       └── disk_io.cpp
├── tests/
│   ├── test_bencode.cpp
│   ├── test_protocol.cpp
│   └── test_piece_selection.cpp
└── examples/
    └── download_ubuntu.cpp
```

---

## Testing

```bash
# Download a legal torrent (e.g., Ubuntu ISO)
./bittorrent ubuntu-20.04.torrent

# Monitor progress
./bittorrent status

# Test with local tracker
./tracker --port 8080
./bittorrent --tracker http://localhost:8080 test.torrent
```

---

## Resources

- [BitTorrent Protocol Specification (BEP 3)](http://www.bittorrent.org/beps/bep_0003.html)
- [BitTorrent Enhancement Proposals](http://www.bittorrent.org/beps/bep_0000.html)
- [Unofficial BitTorrent Specification](https://wiki.theory.org/BitTorrentSpecification)
- [libtorrent Documentation](https://www.libtorrent.org/)

Good luck building your BitTorrent client!

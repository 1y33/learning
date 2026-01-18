# Database Design Patterns - Complete Guide

## Table of Contents
1. [Database Fundamentals](#database-fundamentals)
2. [Sharding Patterns](#sharding-patterns)
3. [Replication Patterns](#replication-patterns)
4. [Partitioning Strategies](#partitioning-strategies)
5. [Indexing Strategies](#indexing-strategies)
6. [Denormalization Patterns](#denormalization-patterns)
7. [Materialized Views](#materialized-views)
8. [Change Data Capture (CDC)](#change-data-capture-cdc)
9. [Multi-Tenancy Patterns](#multi-tenancy-patterns)
10. [Database Scaling Patterns](#database-scaling-patterns)
11. [Consistency Patterns](#consistency-patterns)
12. [Transaction Patterns](#transaction-patterns)
13. [Database Selection Matrix](#database-selection-matrix)
14. [Complete Implementations](#complete-implementations)

---

## Database Fundamentals

### ACID vs BASE

**ACID (Traditional Databases)**
- **Atomicity** - All or nothing
- **Consistency** - Valid state always
- **Isolation** - Concurrent transactions don't interfere
- **Durability** - Committed data persists

**BASE (NoSQL Databases)**
- **Basically Available** - System appears to work most of the time
- **Soft state** - State may change without input
- **Eventual consistency** - System will become consistent eventually

### CAP Theorem

```
┌─────────────────────────────────────┐
│           CAP Theorem               │
│  (Can only choose 2 out of 3)       │
├─────────────────────────────────────┤
│                                     │
│         Consistency (C)             │
│              /\                     │
│             /  \                    │
│            /    \                   │
│           /  CA  \                  │
│          /________\                 │
│         /          \                │
│        /   CP   AP  \               │
│       /              \              │
│   Availability    Partition         │
│       (A)         Tolerance (P)     │
└─────────────────────────────────────┘

CA: RDBMS (MySQL, PostgreSQL) - No partition tolerance
CP: MongoDB, Redis, HBase - Sacrifice availability
AP: Cassandra, DynamoDB, Couchbase - Sacrifice consistency
```

---

## Sharding Patterns

### What is Sharding?

**Sharding** = Horizontal partitioning across multiple database servers

### Benefits
✅ **Scalability** - Distribute data across machines
✅ **Performance** - Parallel queries
✅ **Availability** - One shard failure doesn't bring down entire system

### Challenges
❌ **Complexity** - Routing, cross-shard queries
❌ **Rebalancing** - When adding/removing shards
❌ **Transactions** - Hard to maintain ACID across shards

---

### 1. Hash-Based Sharding

**Concept:** Use hash function to determine shard

```python
# hash_sharding.py
import hashlib

class HashSharding:
    """
    Hash-Based Sharding
    - Pros: Even distribution
    - Cons: Adding shards requires rehashing
    """

    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        self.shards = [DatabaseConnection(f"shard_{i}") for i in range(num_shards)]

    def get_shard_id(self, key: str) -> int:
        """
        Calculate shard ID using hash function
        """
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_id = hash_value % self.num_shards
        return shard_id

    def get_shard(self, key: str):
        """
        Get database connection for a key
        """
        shard_id = self.get_shard_id(key)
        return self.shards[shard_id]

    def insert_user(self, user_id: int, data: dict):
        """
        Insert user into appropriate shard
        """
        shard = self.get_shard(str(user_id))
        shard.execute(
            "INSERT INTO users (id, name, email) VALUES (%s, %s, %s)",
            (user_id, data['name'], data['email'])
        )

    def get_user(self, user_id: int):
        """
        Get user from appropriate shard
        """
        shard = self.get_shard(str(user_id))
        result = shard.query(
            "SELECT * FROM users WHERE id = %s",
            (user_id,)
        )
        return result

# Usage
sharding = HashSharding(num_shards=4)

# User 123 goes to shard: hash(123) % 4 = shard_2
sharding.insert_user(123, {'name': 'John', 'email': 'john@example.com'})

# Always goes to same shard
user = sharding.get_user(123)
```

**Problems with simple hash sharding:**

```python
# Problem: Adding a new shard
# Before: 4 shards
# After: 5 shards

# User 123: hash(123) % 4 = 3  (was on shard 3)
# User 123: hash(123) % 5 = 0  (now on shard 0!)

# Solution: Consistent Hashing
```

---

### 2. Consistent Hashing

**Concept:** Minimize reshuffling when adding/removing shards

```python
# consistent_hashing.py
import hashlib
import bisect

class ConsistentHashing:
    """
    Consistent Hashing
    - Minimize data movement when adding/removing shards
    - Use virtual nodes for better distribution
    """

    def __init__(self, num_shards: int, virtual_nodes: int = 150):
        self.num_shards = num_shards
        self.virtual_nodes = virtual_nodes
        self.ring = []  # Sorted list of hash positions
        self.hash_to_shard = {}  # Map hash -> shard

        self._initialize_ring()

    def _hash(self, key: str) -> int:
        """Generate hash for a key"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _initialize_ring(self):
        """
        Initialize consistent hashing ring with virtual nodes
        """
        for shard_id in range(self.num_shards):
            for vnode_id in range(self.virtual_nodes):
                # Create virtual node
                vnode_key = f"shard_{shard_id}_vnode_{vnode_id}"
                hash_value = self._hash(vnode_key)

                # Add to ring
                self.ring.append(hash_value)
                self.hash_to_shard[hash_value] = shard_id

        # Sort ring
        self.ring.sort()

    def get_shard_id(self, key: str) -> int:
        """
        Find shard for a key using consistent hashing
        """
        if not self.ring:
            return 0

        key_hash = self._hash(key)

        # Find first position >= key_hash
        idx = bisect.bisect_right(self.ring, key_hash)

        # Wrap around if at end
        if idx == len(self.ring):
            idx = 0

        ring_position = self.ring[idx]
        shard_id = self.hash_to_shard[ring_position]

        return shard_id

    def add_shard(self, new_shard_id: int):
        """
        Add new shard to ring
        Only ~1/N keys need to move!
        """
        for vnode_id in range(self.virtual_nodes):
            vnode_key = f"shard_{new_shard_id}_vnode_{vnode_id}"
            hash_value = self._hash(vnode_key)

            # Insert into sorted ring
            bisect.insort(self.ring, hash_value)
            self.hash_to_shard[hash_value] = new_shard_id

        self.num_shards += 1

    def remove_shard(self, shard_id: int):
        """
        Remove shard from ring
        """
        # Remove all virtual nodes for this shard
        positions_to_remove = [
            pos for pos, sid in self.hash_to_shard.items()
            if sid == shard_id
        ]

        for pos in positions_to_remove:
            self.ring.remove(pos)
            del self.hash_to_shard[pos]

        self.num_shards -= 1

# Demo: Consistent hashing minimizes data movement
ch = ConsistentHashing(num_shards=4)

# Distribute 10000 users
shard_distribution = {i: 0 for i in range(4)}
for user_id in range(10000):
    shard_id = ch.get_shard_id(f"user_{user_id}")
    shard_distribution[shard_id] += 1

print("Distribution with 4 shards:", shard_distribution)
# Output: {0: 2487, 1: 2513, 2: 2491, 3: 2509} - Pretty balanced!

# Add 5th shard
ch.add_shard(4)

# Redistribute
new_distribution = {i: 0 for i in range(5)}
for user_id in range(10000):
    shard_id = ch.get_shard_id(f"user_{user_id}")
    new_distribution[shard_id] += 1

print("Distribution with 5 shards:", new_distribution)
# Output: {0: 1989, 1: 2011, 2: 1995, 3: 2007, 4: 1998}
# Only ~20% of data moved! (instead of 100% with simple hashing)
```

**Advantages:**
✅ Adding/removing shards only moves ~1/N data
✅ Virtual nodes ensure even distribution
✅ Widely used (Cassandra, DynamoDB, Memcached)

---

### 3. Range-Based Sharding

**Concept:** Partition by ranges (e.g., user_id 0-999999 → shard 0)

```python
# range_sharding.py
class RangeSharding:
    """
    Range-Based Sharding
    - Pros: Range queries are efficient
    - Cons: Uneven distribution (hotspots)
    """

    def __init__(self):
        # Define ranges for each shard
        self.ranges = [
            (0, 999999, DatabaseConnection("shard_0")),
            (1000000, 1999999, DatabaseConnection("shard_1")),
            (2000000, 2999999, DatabaseConnection("shard_2")),
            (3000000, float('inf'), DatabaseConnection("shard_3")),
        ]

    def get_shard(self, user_id: int):
        """
        Find shard based on range
        """
        for start, end, shard in self.ranges:
            if start <= user_id <= end:
                return shard

        raise ValueError(f"No shard found for user_id {user_id}")

    def insert_user(self, user_id: int, data: dict):
        shard = self.get_shard(user_id)
        shard.execute(
            "INSERT INTO users (id, name) VALUES (%s, %s)",
            (user_id, data['name'])
        )

    def range_query(self, start_id: int, end_id: int):
        """
        Efficient range queries!
        """
        results = []

        for range_start, range_end, shard in self.ranges:
            # Check if range overlaps
            if range_end >= start_id and range_start <= end_id:
                # Query this shard
                shard_results = shard.query(
                    "SELECT * FROM users WHERE id BETWEEN %s AND %s",
                    (max(start_id, range_start), min(end_id, range_end))
                )
                results.extend(shard_results)

        return results

# Usage
sharding = RangeSharding()

# Efficient: Get users 500000-600000
users = sharding.range_query(500000, 600000)  # Only queries shard_0!
```

**Use Cases:**
✅ Time-series data (partition by date)
✅ Geographic data (partition by region)
✅ Sequential IDs

**Problems:**
❌ Hotspots (newest users all on last shard)
❌ Requires planning ranges upfront

---

### 4. Geographic Sharding

**Concept:** Shard by location for low latency

```python
# geographic_sharding.py
class GeographicSharding:
    """
    Geographic Sharding
    - Pros: Low latency (data near users)
    - Cons: Uneven distribution
    """

    def __init__(self):
        self.region_shards = {
            'US-EAST': DatabaseConnection("us-east-db"),
            'US-WEST': DatabaseConnection("us-west-db"),
            'EU': DatabaseConnection("eu-db"),
            'ASIA': DatabaseConnection("asia-db"),
        }

    def get_shard(self, region: str):
        """Get shard based on user's region"""
        return self.region_shards.get(region, self.region_shards['US-EAST'])

    def insert_user(self, user_id: int, data: dict, region: str):
        shard = self.get_shard(region)
        shard.execute(
            "INSERT INTO users (id, name, region) VALUES (%s, %s, %s)",
            (user_id, data['name'], region)
        )

    def get_user(self, user_id: int, region: str):
        """
        Fast: Data is in same region as user
        """
        shard = self.get_shard(region)
        return shard.query(
            "SELECT * FROM users WHERE id = %s",
            (user_id,)
        )

# Usage: EU users → EU database (low latency!)
sharding = GeographicSharding()
sharding.insert_user(123, {'name': 'Hans'}, region='EU')
```

---

### 5. Composite Sharding

**Concept:** Combine multiple sharding strategies

```python
# composite_sharding.py
class CompositeSharding:
    """
    Composite Sharding: Geographic + Hash
    - First shard by region (low latency)
    - Then shard by hash within region (scalability)
    """

    def __init__(self):
        self.regions = {
            'US': ConsistentHashing(num_shards=10),  # 10 shards in US
            'EU': ConsistentHashing(num_shards=8),   # 8 shards in EU
            'ASIA': ConsistentHashing(num_shards=12), # 12 shards in Asia
        }

    def get_shard(self, user_id: int, region: str):
        """
        Two-level sharding
        1. Find region
        2. Hash within region
        """
        region_sharder = self.regions.get(region)
        if not region_sharder:
            raise ValueError(f"Unknown region: {region}")

        shard_id = region_sharder.get_shard_id(str(user_id))

        # Shard name: US_7, EU_3, ASIA_9
        return f"{region}_{shard_id}"

# Usage
sharding = CompositeSharding()

# US user 123 → US_7
us_shard = sharding.get_shard(user_id=123, region='US')

# EU user 123 → EU_3 (different shard!)
eu_shard = sharding.get_shard(user_id=123, region='EU')
```

---

## Replication Patterns

### Why Replicate?

✅ **High Availability** - If master fails, promote replica
✅ **Read Scaling** - Distribute reads across replicas
✅ **Disaster Recovery** - Backups in different regions
✅ **Low Latency** - Replicas closer to users

---

### 1. Master-Slave Replication

**Concept:** One master (writes), multiple slaves (reads)

```python
# master_slave_replication.py
import random

class MasterSlaveReplication:
    """
    Master-Slave Replication
    - All writes → Master
    - Reads → Slaves (load balanced)
    """

    def __init__(self):
        self.master = DatabaseConnection("master-db")
        self.slaves = [
            DatabaseConnection("slave-1"),
            DatabaseConnection("slave-2"),
            DatabaseConnection("slave-3"),
        ]

    def write(self, query: str, params: tuple):
        """
        All writes go to master
        """
        result = self.master.execute(query, params)

        # Master replicates to slaves asynchronously
        # (handled by database, not application)

        return result

    def read(self, query: str, params: tuple):
        """
        Reads from random slave (load balancing)
        """
        slave = random.choice(self.slaves)
        return slave.query(query, params)

    def read_from_master(self, query: str, params: tuple):
        """
        Force read from master (for consistency)
        """
        return self.master.query(query, params)

# Usage
db = MasterSlaveReplication()

# Write to master
db.write("INSERT INTO users (name) VALUES (%s)", ('John',))

# Read from slave (might be slightly stale)
users = db.read("SELECT * FROM users", ())

# Read from master (guaranteed fresh)
fresh_users = db.read_from_master("SELECT * FROM users", ())
```

**Configuration (PostgreSQL):**

```sql
-- On Master
-- postgresql.conf
wal_level = replica
max_wal_senders = 10
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/archive/%f'

-- On Slave
-- recovery.conf
standby_mode = 'on'
primary_conninfo = 'host=master-db port=5432 user=replication password=xxx'
```

**Replication Lag:**

```python
def check_replication_lag():
    """
    Monitor replication lag
    """
    # Query master
    master_lsn = master_db.query("SELECT pg_current_wal_lsn()")[0]

    # Query slave
    slave_lsn = slave_db.query("SELECT pg_last_wal_replay_lsn()")[0]

    # Calculate lag (in bytes)
    lag = master_lsn - slave_lsn

    if lag > 1000000:  # 1MB
        print(f"⚠️ Replication lag is {lag} bytes!")

    return lag
```

---

### 2. Multi-Master Replication

**Concept:** Multiple masters, all accept writes

```python
# multi_master_replication.py
class MultiMasterReplication:
    """
    Multi-Master Replication
    - Pros: High availability, write scaling
    - Cons: Conflict resolution needed
    """

    def __init__(self):
        self.masters = [
            DatabaseConnection("master-us"),
            DatabaseConnection("master-eu"),
            DatabaseConnection("master-asia"),
        ]

    def write_to_nearest_master(self, user_region: str, query: str, params: tuple):
        """
        Write to geographically closest master
        """
        master = self._get_nearest_master(user_region)
        result = master.execute(query, params)

        # Changes replicate to other masters
        # Conflict resolution handled by database

        return result

    def _get_nearest_master(self, region: str):
        region_mapping = {
            'US': self.masters[0],
            'EU': self.masters[1],
            'ASIA': self.masters[2],
        }
        return region_mapping.get(region, self.masters[0])

# Conflict Resolution Strategies
class ConflictResolution:
    """
    How to handle conflicts in multi-master replication
    """

    @staticmethod
    def last_write_wins(versions: list):
        """
        Strategy 1: Last Write Wins (LWW)
        - Use timestamp
        - Simple but can lose data
        """
        latest = max(versions, key=lambda v: v['timestamp'])
        return latest

    @staticmethod
    def vector_clock(versions: list):
        """
        Strategy 2: Vector Clocks
        - Track causality
        - Used by DynamoDB, Riak
        """
        # Simplified example
        pass

    @staticmethod
    def crdt_merge(versions: list):
        """
        Strategy 3: CRDTs (Conflict-free Replicated Data Types)
        - Mathematically guaranteed merge
        - Used by Redis, Riak
        """
        # Example: Counter CRDT
        total = sum(v['counter'] for v in versions)
        return {'counter': total}
```

**PostgreSQL BDR (Bi-Directional Replication):**

```sql
-- Setup multi-master replication with PostgreSQL BDR
CREATE EXTENSION bdr;

-- On Master 1 (US)
SELECT bdr.bdr_group_create(
    local_node_name := 'us-node',
    node_external_dsn := 'host=us-master port=5432 dbname=mydb'
);

-- On Master 2 (EU)
SELECT bdr.bdr_group_join(
    local_node_name := 'eu-node',
    node_external_dsn := 'host=eu-master port=5432 dbname=mydb',
    join_using_dsn := 'host=us-master port=5432 dbname=mydb'
);

-- Conflict resolution
ALTER TABLE users SET (bdr.conflict_resolution = 'last_update_wins');
```

---

### 3. Master-Master with Failover

**Concept:** Two masters, one active, one standby

```python
# master_master_failover.py
import time
from enum import Enum

class NodeState(Enum):
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"

class MasterMasterFailover:
    """
    Active-Standby with automatic failover
    """

    def __init__(self):
        self.master1 = DatabaseConnection("master-1")
        self.master2 = DatabaseConnection("master-2")

        self.active = self.master1
        self.standby = self.master2

        self.state = {
            'master-1': NodeState.ACTIVE,
            'master-2': NodeState.STANDBY
        }

    def write(self, query: str, params: tuple):
        """
        Write to active master
        """
        try:
            result = self.active.execute(query, params)
            return result
        except Exception as e:
            print(f"⚠️ Master failed! Initiating failover...")
            self.failover()

            # Retry on new master
            return self.active.execute(query, params)

    def failover(self):
        """
        Promote standby to active
        """
        print(f"Promoting standby to active...")

        # Swap active/standby
        self.active, self.standby = self.standby, self.active

        # Update state
        self.state['master-1'] = NodeState.STANDBY if self.active == self.master1 else NodeState.FAILED
        self.state['master-2'] = NodeState.ACTIVE if self.active == self.master2 else NodeState.STANDBY

        print(f"Failover complete. Active: {self.active.host}")

    def health_check(self):
        """
        Continuous health checking
        """
        while True:
            try:
                self.active.query("SELECT 1")
            except Exception:
                print("Active master is down! Failing over...")
                self.failover()

            time.sleep(5)  # Check every 5 seconds
```

**Using Patroni (PostgreSQL HA):**

```yaml
# patroni.yml
scope: postgres-cluster
name: postgres-1

restapi:
  listen: 0.0.0.0:8008
  connect_address: postgres-1:8008

etcd:
  host: etcd:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576

postgresql:
  listen: 0.0.0.0:5432
  connect_address: postgres-1:5432
  data_dir: /var/lib/postgresql/data
  parameters:
    max_connections: 100
    shared_buffers: 256MB
```

---

## Partitioning Strategies

### Horizontal Partitioning (Sharding)

**Already covered in Sharding section**

### Vertical Partitioning

**Concept:** Split table by columns

```python
# vertical_partitioning.py
"""
Example: User table split into multiple tables

Before (Single Table):
+----+-------+-------+-----------+--------+-----+
| id | name  | email | password  | bio    | ... |
+----+-------+-------+-----------+--------+-----+

After (Vertical Partitioning):
users_core:              users_auth:          users_profile:
+----+-------+-------+  +----+-----------+    +----+-----+-----+
| id | name  | email |  | id | password  |    | id | bio | ... |
+----+-------+-------+  +----+-----------+    +----+-----+-----+

Why?
- users_core: Accessed frequently (cache)
- users_auth: Security-sensitive (encryption)
- users_profile: Large data (separate storage)
"""

class VerticalPartitioning:
    def __init__(self):
        self.users_core = DatabaseConnection("users_core_db")
        self.users_auth = DatabaseConnection("users_auth_db")
        self.users_profile = DatabaseConnection("users_profile_db")

    def create_user(self, user_data: dict):
        """
        Insert into multiple partitions
        """
        user_id = user_data['id']

        # Core data
        self.users_core.execute(
            "INSERT INTO users (id, name, email) VALUES (%s, %s, %s)",
            (user_id, user_data['name'], user_data['email'])
        )

        # Auth data (encrypted)
        self.users_auth.execute(
            "INSERT INTO user_auth (user_id, password_hash) VALUES (%s, %s)",
            (user_id, user_data['password_hash'])
        )

        # Profile data
        self.users_profile.execute(
            "INSERT INTO user_profiles (user_id, bio, avatar_url) VALUES (%s, %s, %s)",
            (user_id, user_data['bio'], user_data['avatar_url'])
        )

    def get_user_for_login(self, email: str):
        """
        Login: Only need core + auth data
        """
        # Query only necessary partitions
        core = self.users_core.query(
            "SELECT id, name, email FROM users WHERE email = %s",
            (email,)
        )

        if core:
            auth = self.users_auth.query(
                "SELECT password_hash FROM user_auth WHERE user_id = %s",
                (core['id'],)
            )
            return {**core, **auth}

        return None

    def get_full_user_profile(self, user_id: int):
        """
        Profile page: Need all data
        Join from multiple partitions
        """
        core = self.users_core.query("SELECT * FROM users WHERE id = %s", (user_id,))
        profile = self.users_profile.query("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))

        return {**core, **profile}
```

**Benefits:**
✅ Optimize each partition separately
✅ Different storage engines
✅ Security isolation
✅ Reduce I/O for common queries

---

## Indexing Strategies

### B-Tree Index (Default)

```sql
-- B-Tree: Good for equality and range queries
CREATE INDEX idx_users_email ON users(email);

-- Query uses index
SELECT * FROM users WHERE email = 'john@example.com';

-- Range query also uses index
SELECT * FROM users WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';
```

### Hash Index

```sql
-- Hash: Only for equality (=), faster than B-Tree
CREATE INDEX idx_users_email_hash ON users USING HASH (email);

-- Good
SELECT * FROM users WHERE email = 'john@example.com';

-- Bad: Can't use hash index
SELECT * FROM users WHERE email LIKE 'john%';
```

### Composite Index

```python
# composite_index.py
"""
Composite Index: Multiple columns

Index on (country, city, age)
"""

# Create composite index
CREATE INDEX idx_location_age ON users(country, city, age);

# Queries that use the index:

# ✅ Uses index (leftmost prefix)
SELECT * FROM users WHERE country = 'US';

# ✅ Uses index
SELECT * FROM users WHERE country = 'US' AND city = 'NYC';

# ✅ Uses index (full)
SELECT * FROM users WHERE country = 'US' AND city = 'NYC' AND age = 25;

# ❌ Doesn't use index (missing leftmost)
SELECT * FROM users WHERE city = 'NYC';

# ❌ Doesn't use index
SELECT * FROM users WHERE age = 25;

# ✅ Uses index for country, scans for age
SELECT * FROM users WHERE country = 'US' AND age = 25;
```

**Index Column Order Matters:**

```python
class IndexOptimization:
    """
    Rule: Most selective column first
    """

    @staticmethod
    def analyze_selectivity(table: str, column: str):
        """
        Calculate selectivity: # distinct values / # rows
        Higher = more selective = should be first in index
        """
        query = f"""
        SELECT
            COUNT(DISTINCT {column}) as distinct_values,
            COUNT(*) as total_rows,
            COUNT(DISTINCT {column})::float / COUNT(*) as selectivity
        FROM {table}
        """

        # Example results:
        # country: 200 distinct / 10M rows = 0.00002 (low selectivity)
        # email: 10M distinct / 10M rows = 1.0 (high selectivity)

        # Good: CREATE INDEX idx ON users(email, country)
        # Bad: CREATE INDEX idx ON users(country, email)
```

### Covering Index

```sql
-- Covering Index: Includes all columns needed by query
-- Query doesn't need to access table!

CREATE INDEX idx_users_covering ON users(email) INCLUDE (name, created_at);

-- This query uses ONLY the index (no table access)
SELECT email, name, created_at
FROM users
WHERE email = 'john@example.com';

-- Index-only scan (very fast!)
```

### Partial Index

```sql
-- Partial Index: Index only rows that meet condition
-- Smaller index, faster queries

-- Only index active users (90% of queries are for active users)
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- This uses the partial index
SELECT * FROM users WHERE email = 'john@example.com' AND status = 'active';

-- This doesn't (status != 'active')
SELECT * FROM users WHERE email = 'john@example.com' AND status = 'deleted';
```

### Expression Index

```sql
-- Expression Index: Index on computed value

-- Index on lowercase email
CREATE INDEX idx_email_lower ON users(LOWER(email));

-- This uses the index
SELECT * FROM users WHERE LOWER(email) = 'john@example.com';

-- Index on JSON field
CREATE INDEX idx_metadata_country ON users((metadata->>'country'));

SELECT * FROM users WHERE metadata->>'country' = 'US';
```

### Full-Text Search Index

```sql
-- PostgreSQL: Full-text search
CREATE INDEX idx_posts_fulltext ON posts USING GIN(to_tsvector('english', title || ' ' || body));

-- Search
SELECT * FROM posts
WHERE to_tsvector('english', title || ' ' || body) @@ to_tsquery('english', 'database & design');

-- Ranking
SELECT *, ts_rank(to_tsvector('english', title || ' ' || body), to_tsquery('english', 'database')) as rank
FROM posts
WHERE to_tsvector('english', title || ' ' || body) @@ to_tsquery('english', 'database')
ORDER BY rank DESC;
```

---

## Denormalization Patterns

### When to Denormalize?

**Normalize (Default):**
- Avoid data redundancy
- Easier updates
- Data consistency

**Denormalize (Performance):**
- Reduce JOINs
- Faster reads
- Acceptable for read-heavy systems

---

### Pattern 1: Duplicate Data

```python
# denormalization_duplicate.py
"""
Example: E-commerce orders

Normalized (3 tables):
orders
+----------+---------+-------------+
| order_id | user_id | total_price |
+----------+---------+-------------+

order_items
+----------+---------+----------+-------+
| order_id | product_id | quantity | price |
+----------+---------+----------+-------+

products
+------------+------+-------+
| product_id | name | price |
+------------+------+-------+

Problem: Get order details requires 3 JOINs!

Denormalized (1 table):
orders
+----------+---------+-------------+---------------------------+
| order_id | user_id | total_price | items (JSONB)             |
+----------+---------+-------------+---------------------------+
| 1        | 123     | 150.00      | [{"product_name": "...",  |
|          |         |             |   "price": 50.00,         |
|          |         |             |   "quantity": 3}]         |
+----------+---------+-------------+---------------------------+

Benefit: Single query, no JOINs!
Trade-off: Product name/price duplicated (if product changes, orders keep old data)
"""

class DenormalizedOrderStorage:
    def create_order(self, user_id: int, items: list):
        """
        Store order with denormalized product data
        """
        # Get product details
        product_ids = [item['product_id'] for item in items]
        products = db.query(
            "SELECT id, name, price FROM products WHERE id IN %s",
            (tuple(product_ids),)
        )

        # Denormalize: Include product name and current price in order
        denormalized_items = []
        total_price = 0

        for item in items:
            product = next(p for p in products if p['id'] == item['product_id'])

            denormalized_items.append({
                'product_id': product['id'],
                'product_name': product['name'],  # Denormalized!
                'price': product['price'],         # Denormalized!
                'quantity': item['quantity']
            })

            total_price += product['price'] * item['quantity']

        # Insert order
        db.execute(
            "INSERT INTO orders (user_id, total_price, items) VALUES (%s, %s, %s)",
            (user_id, total_price, json.dumps(denormalized_items))
        )

    def get_order(self, order_id: int):
        """
        Single query, no JOINs!
        """
        order = db.query(
            "SELECT * FROM orders WHERE order_id = %s",
            (order_id,)
        )

        # All data is here, no JOINs needed
        return order
```

---

### Pattern 2: Computed Aggregates

```python
# denormalization_aggregates.py
"""
Example: User post counts

Normalized:
users               posts
+----+-------+     +----+---------+-------+
| id | name  |     | id | user_id | title |
+----+-------+     +----+---------+-------+

To get post count: SELECT user_id, COUNT(*) FROM posts GROUP BY user_id

Denormalized:
users
+----+-------+------------+
| id | name  | post_count |  ← Denormalized!
+----+-------+------------+

Benefit: No aggregation needed!
"""

class DenormalizedUserStats:
    def create_post(self, user_id: int, title: str, content: str):
        """
        Create post and update denormalized counter
        """
        # Insert post
        db.execute(
            "INSERT INTO posts (user_id, title, content) VALUES (%s, %s, %s)",
            (user_id, title, content)
        )

        # Update denormalized counter
        db.execute(
            "UPDATE users SET post_count = post_count + 1 WHERE id = %s",
            (user_id,)
        )

    def delete_post(self, post_id: int):
        """
        Delete post and update counter
        """
        # Get user_id before deleting
        post = db.query("SELECT user_id FROM posts WHERE id = %s", (post_id,))
        user_id = post['user_id']

        # Delete post
        db.execute("DELETE FROM posts WHERE id = %s", (post_id,))

        # Update counter
        db.execute(
            "UPDATE users SET post_count = post_count - 1 WHERE id = %s",
            (user_id,)
        )

    def get_top_contributors(self):
        """
        Super fast! No aggregation needed
        """
        return db.query(
            "SELECT id, name, post_count FROM users ORDER BY post_count DESC LIMIT 10"
        )

# Alternative: Use triggers to keep in sync
CREATE TRIGGER update_post_count_on_insert
AFTER INSERT ON posts
FOR EACH ROW
EXECUTE FUNCTION increment_user_post_count();

CREATE FUNCTION increment_user_post_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users SET post_count = post_count + 1 WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

## Materialized Views

**Concept:** Pre-computed query results stored as table

```sql
-- Regular View (Virtual)
CREATE VIEW active_users AS
SELECT id, name, email
FROM users
WHERE status = 'active';

-- Querying view runs the query every time!
SELECT * FROM active_users;  -- Scans users table

-- Materialized View (Physical)
CREATE MATERIALIZED VIEW active_users_mv AS
SELECT id, name, email
FROM users
WHERE status = 'active';

-- First time: Slow (computes)
SELECT * FROM active_users_mv;

-- Subsequent queries: Fast! (reads pre-computed data)
SELECT * FROM active_users_mv;

-- Refresh when data changes
REFRESH MATERIALIZED VIEW active_users_mv;

-- Concurrent refresh (doesn't block reads)
REFRESH MATERIALIZED VIEW CONCURRENTLY active_users_mv;
```

**Advanced Example: Analytics Dashboard**

```python
# materialized_views.py
"""
Example: Analytics dashboard with complex aggregations
"""

# Create materialized view
CREATE MATERIALIZED VIEW daily_revenue_mv AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as order_count,
    SUM(total_price) as revenue,
    AVG(total_price) as avg_order_value,
    COUNT(DISTINCT user_id) as unique_customers
FROM orders
GROUP BY DATE(created_at);

CREATE INDEX idx_daily_revenue_date ON daily_revenue_mv(date);

class AnalyticsDashboard:
    def get_revenue_trend(self, days: int = 30):
        """
        Get revenue trend (very fast!)
        """
        # Without materialized view: Would scan millions of orders
        # With materialized view: Reads 30 rows
        return db.query("""
            SELECT * FROM daily_revenue_mv
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC
        """, (days,))

    def refresh_dashboard(self):
        """
        Refresh materialized view (run every hour via cron)
        """
        db.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY daily_revenue_mv")

# Automatic refresh with trigger
class AutoRefreshMaterializedView:
    """
    Auto-refresh materialized view when base table changes
    """

    @staticmethod
    def setup_auto_refresh():
        # Create trigger
        db.execute("""
            CREATE OR REPLACE FUNCTION refresh_daily_revenue()
            RETURNS TRIGGER AS $$
            BEGIN
                REFRESH MATERIALIZED VIEW CONCURRENTLY daily_revenue_mv;
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;

            CREATE TRIGGER orders_refresh_trigger
            AFTER INSERT OR UPDATE OR DELETE ON orders
            FOR EACH STATEMENT
            EXECUTE FUNCTION refresh_daily_revenue();
        """)
```

---

## Change Data Capture (CDC)

**Concept:** Track changes to database and stream them

```python
# cdc_pattern.py
"""
CDC: Capture database changes and propagate to other systems

Use cases:
- Sync to Elasticsearch for search
- Update cache (Redis)
- Stream to data warehouse
- Event-driven architecture
"""

# Method 1: Database Triggers
CREATE TABLE user_changes_log (
    id SERIAL PRIMARY KEY,
    user_id INT,
    operation VARCHAR(10),  -- INSERT, UPDATE, DELETE
    old_data JSONB,
    new_data JSONB,
    changed_at TIMESTAMP DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION log_user_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO user_changes_log (user_id, operation, new_data)
        VALUES (NEW.id, 'INSERT', row_to_json(NEW));
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO user_changes_log (user_id, operation, old_data, new_data)
        VALUES (NEW.id, 'UPDATE', row_to_json(OLD), row_to_json(NEW));
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO user_changes_log (user_id, operation, old_data)
        VALUES (OLD.id, 'DELETE', row_to_json(OLD));
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_changes_trigger
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION log_user_changes();

# Method 2: Debezium (Production-grade CDC)
from kafka import KafkaConsumer
import json

class DebeziumCDCConsumer:
    """
    Consume database changes from Debezium
    """

    def __init__(self):
        self.consumer = KafkaConsumer(
            'dbserver1.public.users',  # Topic: <server>.<schema>.<table>
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    def process_changes(self):
        """
        Process database changes
        """
        for message in self.consumer:
            change = message.value

            operation = change['payload']['op']  # c=create, u=update, d=delete
            data = change['payload']['after'] if operation != 'd' else change['payload']['before']

            if operation == 'c':
                self.handle_insert(data)
            elif operation == 'u':
                self.handle_update(data)
            elif operation == 'd':
                self.handle_delete(data)

    def handle_insert(self, data: dict):
        """Sync new user to Elasticsearch"""
        es.index(index='users', id=data['id'], document=data)

    def handle_update(self, data: dict):
        """Update user in Elasticsearch"""
        es.update(index='users', id=data['id'], doc=data)

    def handle_delete(self, data: dict):
        """Remove user from Elasticsearch"""
        es.delete(index='users', id=data['id'])

# docker-compose.yml for Debezium
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092

  postgres:
    image: debezium/postgres:14
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres

  debezium:
    image: debezium/connect:latest
    depends_on:
      - kafka
      - postgres
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: 1
      CONFIG_STORAGE_TOPIC: debezium_configs
      OFFSET_STORAGE_TOPIC: debezium_offsets
      STATUS_STORAGE_TOPIC: debezium_statuses
```

---

## Multi-Tenancy Patterns

### Pattern 1: Shared Database, Shared Schema

```python
# multi_tenancy_shared.py
"""
All tenants share same tables, filtered by tenant_id
"""

# Table design
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    tenant_id INT NOT NULL,  -- Discriminator
    name VARCHAR(100),
    email VARCHAR(100),
    UNIQUE(tenant_id, email)
);

CREATE INDEX idx_users_tenant ON users(tenant_id);

class SharedDatabaseMultiTenancy:
    def __init__(self, tenant_id: int):
        self.tenant_id = tenant_id
        self.db = DatabaseConnection()

    def get_users(self):
        """
        Always filter by tenant_id
        """
        return self.db.query(
            "SELECT * FROM users WHERE tenant_id = %s",
            (self.tenant_id,)
        )

    def create_user(self, name: str, email: str):
        """
        Automatically include tenant_id
        """
        return self.db.execute(
            "INSERT INTO users (tenant_id, name, email) VALUES (%s, %s, %s)",
            (self.tenant_id, name, email)
        )

# Row-Level Security (PostgreSQL)
CREATE POLICY tenant_isolation ON users
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.tenant_id')::int);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

# Set tenant_id for session
SET app.tenant_id = '123';

# Now queries automatically filtered!
SELECT * FROM users;  -- Only returns users for tenant 123
```

**Pros:**
✅ Simple, easy to manage
✅ Cost-effective

**Cons:**
❌ Security risk (tenant_id must always be filtered)
❌ Hard to isolate one tenant's data
❌ Performance impact with many tenants

---

### Pattern 2: Shared Database, Separate Schema

```python
# multi_tenancy_schema.py
"""
Each tenant gets own schema
"""

CREATE SCHEMA tenant_123;
CREATE SCHEMA tenant_456;

CREATE TABLE tenant_123.users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE tenant_456.users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

class SchemaPerTenantMultiTenancy:
    def __init__(self, tenant_id: int):
        self.schema = f"tenant_{tenant_id}"
        self.db = DatabaseConnection()

    def get_users(self):
        """
        Query from tenant-specific schema
        """
        return self.db.query(f"SELECT * FROM {self.schema}.users")

    def create_user(self, name: str, email: str):
        return self.db.execute(
            f"INSERT INTO {self.schema}.users (name, email) VALUES (%s, %s)",
            (name, email)
        )

# Auto-create schema for new tenant
def onboard_new_tenant(tenant_id: int):
    schema = f"tenant_{tenant_id}"

    # Create schema
    db.execute(f"CREATE SCHEMA {schema}")

    # Create tables (from template)
    db.execute(f"CREATE TABLE {schema}.users (id SERIAL PRIMARY KEY, name VARCHAR, email VARCHAR)")
    db.execute(f"CREATE TABLE {schema}.products (id SERIAL PRIMARY KEY, name VARCHAR, price DECIMAL)")

    # Create indexes
    db.execute(f"CREATE INDEX idx_users_email ON {schema}.users(email)")
```

**Pros:**
✅ Better isolation
✅ Easier to backup/restore single tenant
✅ Can customize schema per tenant

**Cons:**
❌ More complex
❌ Schema migration headaches

---

### Pattern 3: Separate Database per Tenant

```python
# multi_tenancy_database.py
"""
Each tenant gets completely separate database
"""

class DatabasePerTenantMultiTenancy:
    def __init__(self, tenant_id: int):
        # Connection string points to tenant-specific database
        self.db = DatabaseConnection(f"tenant_{tenant_id}_db")

    def get_users(self):
        # No need to filter by tenant_id!
        return self.db.query("SELECT * FROM users")

# Tenant routing
class TenantRouter:
    def __init__(self):
        self.connections = {}

    def get_connection(self, tenant_id: int):
        if tenant_id not in self.connections:
            self.connections[tenant_id] = DatabaseConnection(
                host='db-cluster',
                database=f'tenant_{tenant_id}'
            )
        return self.connections[tenant_id]

# API middleware
from flask import request, g

@app.before_request
def set_tenant_connection():
    # Get tenant from subdomain (e.g., acme.myapp.com)
    subdomain = request.host.split('.')[0]
    tenant_id = get_tenant_id(subdomain)

    # Set database connection for this tenant
    g.db = tenant_router.get_connection(tenant_id)

@app.route('/users')
def get_users():
    # Uses tenant-specific database
    users = g.db.query("SELECT * FROM users")
    return jsonify(users)
```

**Pros:**
✅ Complete isolation
✅ Easy to scale (distribute databases)
✅ Can offer different SLAs per tenant

**Cons:**
❌ Expensive (many databases)
❌ Complex management
❌ Schema migrations are nightmare

---

## Database Scaling Patterns

### Read Replicas for Read Scaling

```python
# read_replicas.py
from random import choice

class ReadReplicaPool:
    """
    Distribute reads across replicas
    """

    def __init__(self):
        self.master = DatabaseConnection("master")
        self.replicas = [
            DatabaseConnection("replica-1"),
            DatabaseConnection("replica-2"),
            DatabaseConnection("replica-3"),
        ]

    def write(self, query: str, params: tuple):
        """All writes to master"""
        return self.master.execute(query, params)

    def read(self, query: str, params: tuple, from_master: bool = False):
        """
        Reads from replicas (load balanced)
        Set from_master=True for read-after-write consistency
        """
        if from_master:
            return self.master.query(query, params)

        # Round-robin or random selection
        replica = choice(self.replicas)
        return replica.query(query, params)

# Usage
db = ReadReplicaPool()

# Write to master
user_id = db.write("INSERT INTO users (name) VALUES (%s) RETURNING id", ('John',))[0]

# Read from master (avoid replication lag)
user = db.read("SELECT * FROM users WHERE id = %s", (user_id,), from_master=True)

# Subsequent reads from replicas
user = db.read("SELECT * FROM users WHERE id = %s", (user_id,))
```

### Connection Pooling

```python
# connection_pooling.py
from psycopg2.pool import SimpleConnectionPool

class DatabaseConnectionPool:
    """
    Connection pooling to reduce connection overhead
    """

    def __init__(self, minconn=1, maxconn=20):
        self.pool = SimpleConnectionPool(
            minconn,
            maxconn,
            host='localhost',
            database='mydb',
            user='user',
            password='password'
        )

    def execute(self, query: str, params: tuple = None):
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall()
        finally:
            self.pool.putconn(conn)

# PgBouncer (Production)
# pgbouncer.ini
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction  # or session, statement
max_client_conn = 1000
default_pool_size = 25
```

---

## Consistency Patterns

### Strong Consistency

```python
# strong_consistency.py
"""
Strong Consistency: Read always returns latest write
- Use when: Financial transactions, inventory
"""

class StronglyConsistentDB:
    def __init__(self):
        self.db = DatabaseConnection()

    def transfer_money(self, from_account: int, to_account: int, amount: float):
        """
        ACID transaction - all or nothing
        """
        with self.db.transaction():
            # Deduct from source
            self.db.execute(
                "UPDATE accounts SET balance = balance - %s WHERE id = %s",
                (amount, from_account)
            )

            # Add to destination
            self.db.execute(
                "UPDATE accounts SET balance = balance + %s WHERE id = %s",
                (amount, to_account)
            )

            # If any step fails, entire transaction rolls back

        # Read immediately sees changes
        balance = self.db.query("SELECT balance FROM accounts WHERE id = %s", (from_account,))
```

### Eventual Consistency

```python
# eventual_consistency.py
"""
Eventual Consistency: System becomes consistent eventually
- Use when: Social media likes, view counts
"""

class EventuallyConsistentCounter:
    def __init__(self):
        self.redis = redis.Redis()
        self.db = DatabaseConnection()

    def increment_view_count(self, post_id: int):
        """
        Increment in Redis (fast), sync to DB later
        """
        # Increment in cache (immediate)
        self.redis.incr(f"post:{post_id}:views")

        # Async: Sync to database later (eventual)
        # (Batch update every minute via cron job)

    def batch_sync_to_database(self):
        """
        Periodic sync from Redis to PostgreSQL
        """
        for key in self.redis.scan_iter("post:*:views"):
            post_id = int(key.decode().split(':')[1])
            views = int(self.redis.get(key))

            # Update database
            self.db.execute(
                "UPDATE posts SET view_count = %s WHERE id = %s",
                (views, post_id)
            )

        # Optionally clear Redis after sync
```

### Causal Consistency

```python
# causal_consistency.py
"""
Causal Consistency: If A causes B, everyone sees A before B
- Use when: Social media comments, chat messages
"""

class CausallyConsistentDB:
    def __init__(self):
        self.db = DatabaseConnection()

    def create_comment(self, post_id: int, parent_comment_id: int, text: str):
        """
        Ensure parent comment exists before creating reply
        """
        # Check causality
        if parent_comment_id:
            parent = self.db.query(
                "SELECT id FROM comments WHERE id = %s",
                (parent_comment_id,)
            )
            if not parent:
                raise ValueError("Parent comment does not exist yet")

        # Create comment
        self.db.execute(
            "INSERT INTO comments (post_id, parent_id, text) VALUES (%s, %s, %s)",
            (post_id, parent_comment_id, text)
        )
```

---

## Complete Implementation Example

### E-commerce Database Architecture

```python
# ecommerce_db_architecture.py
"""
Complete E-commerce Database Design

Requirements:
- 10M users
- 100M products
- 1B orders
- Global availability
- High read throughput

Architecture:
- Master-Slave replication
- Read replicas in multiple regions
- Sharding by user_id
- Denormalized order data
- Materialized views for analytics
"""

# 1. Schema Design
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    password_hash VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE products (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock INT NOT NULL DEFAULT 0,
    category_id INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Denormalized orders table
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    user_email VARCHAR(255),  -- Denormalized
    total_price DECIMAL(10, 2),
    status VARCHAR(50),
    items JSONB,  -- Denormalized product data
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_orders_created ON orders(created_at);
CREATE INDEX idx_orders_items_gin ON orders USING GIN(items);

-- Partitioning (by month)
CREATE TABLE orders_2024_01 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE orders_2024_02 PARTITION OF orders
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

# 2. Sharding Strategy
class EcommerceDatabase:
    def __init__(self):
        # User data sharded by user_id
        self.user_sharding = ConsistentHashing(num_shards=16)

        # Products: Not sharded (replicated to all regions)
        self.products_db = DatabaseConnection("products_db")

        # Orders: Sharded by user_id (co-located with user data)
        self.order_sharding = self.user_sharding  # Same sharding key

    def create_user(self, email: str, name: str, password: str):
        """Create user in appropriate shard"""
        user_id = generate_id()
        shard = self.user_sharding.get_shard(str(user_id))

        shard.execute(
            "INSERT INTO users (id, email, name, password_hash) VALUES (%s, %s, %s, %s)",
            (user_id, email, name, hash_password(password))
        )

        return user_id

    def create_order(self, user_id: int, items: list):
        """
        Create order with denormalized data
        """
        # Get user data
        shard = self.user_sharding.get_shard(str(user_id))
        user = shard.query("SELECT email FROM users WHERE id = %s", (user_id,))

        # Get product data
        product_ids = [item['product_id'] for item in items]
        products = self.products_db.query(
            "SELECT id, name, price FROM products WHERE id IN %s",
            (tuple(product_ids),)
        )

        # Denormalize: Store product snapshot in order
        denormalized_items = []
        total_price = 0

        for item in items:
            product = next(p for p in products if p['id'] == item['product_id'])

            denormalized_items.append({
                'product_id': product['id'],
                'product_name': product['name'],
                'price': product['price'],
                'quantity': item['quantity']
            })

            total_price += product['price'] * item['quantity']

        # Insert order (in same shard as user)
        order_id = shard.execute(
            """
            INSERT INTO orders (user_id, user_email, total_price, status, items)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (user_id, user['email'], total_price, 'pending', json.dumps(denormalized_items))
        )[0]

        return order_id

# 3. Read Replicas
class EcommerceReadReplicas:
    def __init__(self):
        # Geographic replicas
        self.replicas = {
            'US-EAST': DatabaseConnection("us-east-replica"),
            'US-WEST': DatabaseConnection("us-west-replica"),
            'EU': DatabaseConnection("eu-replica"),
            'ASIA': DatabaseConnection("asia-replica"),
        }

    def get_products(self, region: str, category_id: int):
        """
        Read from nearest replica
        """
        replica = self.replicas.get(region, self.replicas['US-EAST'])

        return replica.query(
            "SELECT * FROM products WHERE category_id = %s LIMIT 100",
            (category_id,)
        )

# 4. Materialized Views for Analytics
CREATE MATERIALIZED VIEW daily_sales_summary AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as order_count,
    SUM(total_price) as total_revenue,
    AVG(total_price) as avg_order_value,
    COUNT(DISTINCT user_id) as unique_customers
FROM orders
GROUP BY DATE(created_at);

CREATE INDEX idx_daily_sales_date ON daily_sales_summary(date);

# Refresh every hour
SELECT cron.schedule('refresh-daily-sales', '0 * * * *',
    'REFRESH MATERIALIZED VIEW CONCURRENTLY daily_sales_summary');

# 5. CDC for Search Index
class ProductSearchSync:
    """
    Sync product changes to Elasticsearch
    """

    def __init__(self):
        self.es = Elasticsearch(['localhost:9200'])

        # Subscribe to product changes
        self.cdc = DebeziumCDCConsumer('products')

    def sync_to_elasticsearch(self):
        for change in self.cdc.process_changes():
            if change['op'] == 'c' or change['op'] == 'u':
                # Index/update product
                product = change['after']
                self.es.index(
                    index='products',
                    id=product['id'],
                    document={
                        'name': product['name'],
                        'description': product['description'],
                        'price': product['price'],
                        'category_id': product['category_id']
                    }
                )

            elif change['op'] == 'd':
                # Delete from index
                self.es.delete(index='products', id=change['before']['id'])
```

---

## Key Takeaways

### Database Design Principles

1. **Start Normalized** - Denormalize only for performance
2. **Index Strategically** - Don't over-index
3. **Shard When Necessary** - Don't premature shard
4. **Replicate for Reads** - Master-slave for read scaling
5. **Monitor Query Performance** - Use EXPLAIN ANALYZE

### Scaling Path

```
1. Single Database
    ↓
2. Add Read Replicas
    ↓
3. Add Caching (Redis)
    ↓
4. Vertical Partitioning (split large tables)
    ↓
5. Horizontal Partitioning (sharding)
    ↓
6. Multi-Region Replication
```

### Common Mistakes

❌ **Sharding too early** - Adds complexity
❌ **Not using indexes** - Slow queries
❌ **Over-denormalization** - Data inconsistency
❌ **Ignoring monitoring** - Can't optimize what you don't measure
❌ **No backup strategy** - Data loss

---

**Remember:** Database design is about trade-offs. Choose the pattern that fits YOUR requirements! 🎯

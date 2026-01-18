# Planet-Scale Architecture - Billions of Requests Per Second

## Table of Contents
1. [Introduction to Planet-Scale](#introduction-to-planet-scale)
2. [Global Infrastructure Patterns](#global-infrastructure-patterns)
3. [CDN Architecture](#cdn-architecture)
4. [Multi-Region Database Strategy](#multi-region-database-strategy)
5. [Load Balancing at Scale](#load-balancing-at-scale)
6. [Caching Strategies](#caching-strategies)
7. [Message Queue at Scale](#message-queue-at-scale)
8. [Real-World Architectures](#real-world-architectures)
9. [Observability & Monitoring](#observability--monitoring)
10. [Disaster Recovery](#disaster-recovery)
11. [Cost Optimization](#cost-optimization)
12. [Complete Implementation Examples](#complete-implementation-examples)

---

## Introduction to Planet-Scale

### What is Planet-Scale?

**Planet-Scale** = Systems that serve **billions of users** across the **entire globe** with:
- **Billions of requests per second**
- **Petabytes of data**
- **Sub-100ms latency worldwide**
- **99.99%+ availability**

### Companies at Planet-Scale

| Company | Daily Active Users | Requests/Second | Data Scale |
|---------|-------------------|-----------------|------------|
| **Google** | 4+ billion | 100+ billion | Exabytes |
| **Facebook** | 3+ billion | 60+ billion | Exabytes |
| **Amazon** | 300+ million | 1+ billion | Petabytes |
| **Netflix** | 250+ million | 500+ million | Petabytes |
| **Twitter** | 200+ million | 500+ million | Petabytes |

### Key Challenges

❌ **Network Latency** - Speed of light limit (100ms NYC ↔ Tokyo)
❌ **Data Consistency** - CAP theorem across continents
❌ **Cost** - Infrastructure costs are enormous
❌ **Complexity** - Thousands of services, millions of servers
❌ **Failure** - Something is always failing

---

## Global Infrastructure Patterns

### 1. Multi-Region Active-Active

**Concept:** All regions serve traffic simultaneously

```
┌─────────────────────────────────────────────────────────────┐
│              Active-Active Multi-Region                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   User in US        User in EU         User in ASIA        │
│       ↓                 ↓                   ↓               │
│   ┌───────┐        ┌───────┐          ┌───────┐            │
│   │US-EAST│        │ EU-W  │          │ASIA-SE│            │
│   │Region │        │Region │          │Region │            │
│   │       │        │       │          │       │            │
│   │ App   │←──────→│ App   │←────────→│ App   │            │
│   │ DB    │        │ DB    │          │ DB    │            │
│   └───────┘        └───────┘          └───────┘            │
│       ↑                ↑                   ↑                │
│       └────────────────┴───────────────────┘                │
│            Cross-Region Replication                         │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# multi_region_active_active.py
from enum import Enum
from typing import Dict, List
import requests

class Region(Enum):
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_SOUTHEAST = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"

class ActiveActiveRouter:
    """
    Route users to nearest region
    """

    def __init__(self):
        self.region_endpoints = {
            Region.US_EAST: "https://us-east.api.example.com",
            Region.US_WEST: "https://us-west.api.example.com",
            Region.EU_WEST: "https://eu-west.api.example.com",
            Region.EU_CENTRAL: "https://eu-central.api.example.com",
            Region.ASIA_SOUTHEAST: "https://asia-se.api.example.com",
            Region.ASIA_NORTHEAST: "https://asia-ne.api.example.com",
        }

        # Latency matrix (ms)
        self.latency_matrix = self._build_latency_matrix()

    def get_nearest_region(self, user_location: tuple) -> Region:
        """
        Find region with lowest latency to user
        Uses GeoDNS or client-side latency measurements
        """
        lat, lon = user_location

        # Simple geographic mapping
        if -180 <= lon < -50:  # Americas
            if lat > 40:
                return Region.US_EAST
            else:
                return Region.US_WEST
        elif -50 <= lon < 40:  # Europe
            if lat > 50:
                return Region.EU_WEST
            else:
                return Region.EU_CENTRAL
        else:  # Asia
            if lat > 30:
                return Region.ASIA_NORTHEAST
            else:
                return Region.ASIA_SOUTHEAST

    def route_request(self, user_location: tuple, request_data: dict):
        """
        Route request to nearest region
        """
        region = self.get_nearest_region(user_location)
        endpoint = self.region_endpoints[region]

        # Send request to nearest region
        response = requests.post(f"{endpoint}/api/data", json=request_data)

        return response.json()

    def _build_latency_matrix(self):
        """
        Pre-measured latency between regions
        """
        return {
            (Region.US_EAST, Region.EU_WEST): 80,
            (Region.US_EAST, Region.ASIA_SOUTHEAST): 220,
            (Region.EU_WEST, Region.ASIA_SOUTHEAST): 180,
            # ... full matrix
        }

# Cross-Region Replication
class CrossRegionReplication:
    """
    Replicate data across regions
    """

    def __init__(self):
        self.regions = [
            Region.US_EAST,
            Region.EU_WEST,
            Region.ASIA_SOUTHEAST
        ]

        self.databases = {
            region: DatabaseConnection(f"{region.value}-db")
            for region in self.regions
        }

    def write_with_replication(self, data: dict, consistency: str = 'eventual'):
        """
        Write to multiple regions

        Consistency modes:
        - 'eventual': Write to local, replicate async
        - 'strong': Write to all regions synchronously (slow!)
        - 'quorum': Write to majority of regions
        """

        if consistency == 'eventual':
            # Write to local region
            local_region = self._get_local_region()
            self.databases[local_region].execute(
                "INSERT INTO data (content) VALUES (%s)",
                (data,)
            )

            # Async replication to other regions
            for region in self.regions:
                if region != local_region:
                    self._async_replicate(region, data)

        elif consistency == 'strong':
            # Write to ALL regions synchronously (slow!)
            for region in self.regions:
                self.databases[region].execute(
                    "INSERT INTO data (content) VALUES (%s)",
                    (data,)
                )

        elif consistency == 'quorum':
            # Write to majority (e.g., 2 out of 3)
            import concurrent.futures

            quorum_size = len(self.regions) // 2 + 1
            successful_writes = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.regions)) as executor:
                futures = {
                    executor.submit(
                        self.databases[region].execute,
                        "INSERT INTO data (content) VALUES (%s)",
                        (data,)
                    ): region
                    for region in self.regions
                }

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                        successful_writes += 1

                        if successful_writes >= quorum_size:
                            # Quorum reached!
                            return True
                    except Exception as e:
                        print(f"Write failed in region {futures[future]}: {e}")

            return successful_writes >= quorum_size

    def _async_replicate(self, region: Region, data: dict):
        """
        Async replication using message queue
        """
        kafka_producer.send(
            topic=f'replication-{region.value}',
            value=data
        )

# Conflict Resolution (for concurrent writes)
class ConflictResolver:
    """
    Resolve conflicts when same data modified in multiple regions
    """

    @staticmethod
    def last_write_wins(versions: List[dict]) -> dict:
        """
        Strategy 1: Last Write Wins
        - Simple, but can lose updates
        """
        return max(versions, key=lambda v: v['timestamp'])

    @staticmethod
    def vector_clock(versions: List[dict]) -> dict:
        """
        Strategy 2: Vector Clocks (used by DynamoDB)
        - Track causality
        - Detect concurrent updates
        """
        # Simplified example
        latest_version = None
        latest_clock = {}

        for version in versions:
            clock = version['vector_clock']

            if not latest_version or ConflictResolver._is_newer(clock, latest_clock):
                latest_version = version
                latest_clock = clock

        return latest_version

    @staticmethod
    def _is_newer(clock1: dict, clock2: dict) -> bool:
        """Check if clock1 is newer than clock2"""
        for key in clock1:
            if clock1.get(key, 0) < clock2.get(key, 0):
                return False
        return True

    @staticmethod
    def crdt_merge(versions: List[dict]) -> dict:
        """
        Strategy 3: CRDTs (Conflict-free Replicated Data Types)
        - Mathematically guaranteed convergence
        - Used by Redis, Riak
        """
        # Example: Grow-only counter
        merged = {'counter': 0}

        for version in versions:
            merged['counter'] = max(merged['counter'], version.get('counter', 0))

        return merged
```

**Benefits:**
✅ Low latency worldwide (data near users)
✅ High availability (region failure doesn't impact others)
✅ Write scalability (writes distributed)

**Challenges:**
❌ Complex conflict resolution
❌ Data consistency is eventual
❌ Higher costs (redundant infrastructure)

---

### 2. Multi-Region Active-Passive

**Concept:** One active region, others are standby

```python
# active_passive.py
class ActivePassiveFailover:
    """
    Active region serves traffic
    Passive regions are hot standbys
    """

    def __init__(self):
        self.active_region = Region.US_EAST
        self.passive_regions = [Region.EU_WEST, Region.ASIA_SOUTHEAST]

        self.health_check_interval = 5  # seconds

    def route_request(self, request_data: dict):
        """
        All traffic goes to active region
        """
        endpoint = self.region_endpoints[self.active_region]

        try:
            response = requests.post(f"{endpoint}/api/data", json=request_data, timeout=5)
            return response.json()
        except Exception as e:
            print(f"Active region failed! Initiating failover...")
            self.failover()
            return self.route_request(request_data)

    def failover(self):
        """
        Promote passive region to active
        """
        # Choose nearest passive region
        new_active = self.passive_regions[0]

        print(f"Failing over from {self.active_region} to {new_active}")

        # Update DNS to point to new region (takes 60 seconds)
        self._update_global_dns(new_active)

        # Update routing
        self.passive_regions.remove(new_active)
        self.passive_regions.append(self.active_region)
        self.active_region = new_active

    def _update_global_dns(self, new_region: Region):
        """
        Update global DNS (Route 53, Cloudflare, etc.)
        """
        # AWS Route 53 example
        route53.change_resource_record_sets(
            HostedZoneId='Z123456',
            ChangeBatch={
                'Changes': [{
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': 'api.example.com',
                        'Type': 'A',
                        'TTL': 60,
                        'ResourceRecords': [
                            {'Value': self.region_ips[new_region]}
                        ]
                    }
                }]
            }
        )
```

**Benefits:**
✅ Simpler (no conflict resolution)
✅ Strong consistency
✅ Lower costs (one active infrastructure)

**Challenges:**
❌ Higher latency for distant users
❌ Failover takes time (DNS propagation)
❌ Passive regions are wasted capacity

---

## CDN Architecture

### What is a CDN?

**Content Delivery Network** = Globally distributed servers that cache content close to users

### How CDNs Work

```
User in Tokyo
    ↓
Edge Server (Tokyo) ─── Cache Hit? → Return Cached Content
    ↓ (Cache Miss)
Origin Server (US) → Return Content → Cache at Edge
```

### CDN Implementation

```python
# cdn_architecture.py
import hashlib
from datetime import datetime, timedelta

class CDNEdgeServer:
    """
    Edge server (Point of Presence)
    """

    def __init__(self, location: str):
        self.location = location
        self.cache = {}  # In-memory cache (in reality: Varnish, Nginx)
        self.origin_url = "https://origin.example.com"

    def handle_request(self, url: str, headers: dict) -> dict:
        """
        Handle request at edge server
        """
        cache_key = self._get_cache_key(url, headers)

        # Check cache
        cached_response = self._get_from_cache(cache_key)

        if cached_response and not self._is_expired(cached_response):
            print(f"✅ Cache HIT at {self.location}")
            return cached_response

        # Cache miss - fetch from origin
        print(f"❌ Cache MISS at {self.location} - fetching from origin")
        response = self._fetch_from_origin(url, headers)

        # Cache the response
        self._store_in_cache(cache_key, response)

        return response

    def _get_cache_key(self, url: str, headers: dict) -> str:
        """
        Generate cache key
        Include query params, headers (for personalization)
        """
        # Simple: Just URL
        # Advanced: URL + User-Agent + Accept-Language + Cookies
        key_data = f"{url}:{headers.get('Accept-Language', '')}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str):
        """Get from cache"""
        return self.cache.get(cache_key)

    def _is_expired(self, cached_response: dict) -> bool:
        """Check if cached response is expired"""
        expires_at = cached_response.get('expires_at')
        return datetime.now() > expires_at

    def _fetch_from_origin(self, url: str, headers: dict) -> dict:
        """Fetch from origin server"""
        response = requests.get(f"{self.origin_url}{url}", headers=headers)

        # Parse Cache-Control headers
        cache_control = response.headers.get('Cache-Control', '')
        max_age = self._parse_max_age(cache_control)

        return {
            'content': response.content,
            'status': response.status_code,
            'headers': dict(response.headers),
            'expires_at': datetime.now() + timedelta(seconds=max_age)
        }

    def _store_in_cache(self, cache_key: str, response: dict):
        """Store in cache"""
        self.cache[cache_key] = response

    def _parse_max_age(self, cache_control: str) -> int:
        """Parse max-age from Cache-Control header"""
        for directive in cache_control.split(','):
            if 'max-age' in directive:
                return int(directive.split('=')[1])
        return 0  # Don't cache if no max-age

# Global CDN Network
class GlobalCDN:
    """
    Global CDN with edge servers worldwide
    """

    def __init__(self):
        self.edge_servers = {
            'tokyo': CDNEdgeServer('tokyo'),
            'singapore': CDNEdgeServer('singapore'),
            'london': CDNEdgeServer('london'),
            'frankfurt': CDNEdgeServer('frankfurt'),
            'new-york': CDNEdgeServer('new-york'),
            'san-francisco': CDNEdgeServer('san-francisco'),
            'sao-paulo': CDNEdgeServer('sao-paulo'),
            'sydney': CDNEdgeServer('sydney'),
        }

    def route_to_nearest_edge(self, user_ip: str) -> CDNEdgeServer:
        """
        Route user to nearest edge server (using GeoDNS)
        """
        # In reality: GeoDNS (Route 53, Cloudflare) does this
        user_location = self._geolocate_ip(user_ip)

        # Find nearest edge server
        nearest_edge = min(
            self.edge_servers.values(),
            key=lambda edge: self._calculate_distance(user_location, edge.location)
        )

        return nearest_edge

# Cache Invalidation
class CDNCacheInvalidation:
    """
    Invalidate cached content across all edge servers
    """

    def __init__(self, edge_servers: dict):
        self.edge_servers = edge_servers

    def purge_url(self, url: str):
        """
        Purge specific URL from all edge servers
        """
        print(f"Purging {url} from all edge servers...")

        for location, edge_server in self.edge_servers.items():
            # Remove from cache
            cache_key = edge_server._get_cache_key(url, {})
            if cache_key in edge_server.cache:
                del edge_server.cache[cache_key]
                print(f"Purged from {location}")

    def purge_pattern(self, pattern: str):
        """
        Purge URLs matching pattern (e.g., /images/*)
        """
        import re

        regex = re.compile(pattern)

        for location, edge_server in self.edge_servers.items():
            keys_to_remove = [
                key for key in edge_server.cache.keys()
                if regex.match(key)
            ]

            for key in keys_to_remove:
                del edge_server.cache[key]

            print(f"Purged {len(keys_to_remove)} items from {location}")

# Advanced: CDN with Tiered Caching
class TieredCDN:
    """
    Multi-tier CDN architecture
    Edge → Regional → Origin
    """

    def __init__(self):
        # Tier 1: Edge servers (many, close to users)
        self.edge_cache = RedisCluster(nodes=['edge1:6379', 'edge2:6379'])

        # Tier 2: Regional cache (fewer, larger)
        self.regional_cache = RedisCluster(nodes=['regional1:6379'])

        # Tier 3: Origin (one, has all data)
        self.origin = "https://origin.example.com"

    def get_content(self, url: str) -> bytes:
        """
        Check edge → regional → origin
        """
        # Tier 1: Edge cache
        content = self.edge_cache.get(url)
        if content:
            print("✅ Hit: Edge cache")
            return content

        # Tier 2: Regional cache
        content = self.regional_cache.get(url)
        if content:
            print("✅ Hit: Regional cache")
            # Populate edge cache
            self.edge_cache.setex(url, 3600, content)
            return content

        # Tier 3: Origin
        print("❌ Miss: Fetching from origin")
        response = requests.get(f"{self.origin}{url}")
        content = response.content

        # Populate caches
        self.regional_cache.setex(url, 86400, content)  # 24 hours
        self.edge_cache.setex(url, 3600, content)       # 1 hour

        return content
```

### CDN Strategies for Different Content Types

```python
# cdn_strategies.py
class CDNStrategies:
    """
    Different caching strategies for different content types
    """

    @staticmethod
    def static_assets():
        """
        Static assets (images, CSS, JS)
        - Long cache time
        - Immutable (versioned URLs)
        """
        return {
            'cache-control': 'public, max-age=31536000, immutable',
            'cdn_ttl': 31536000  # 1 year
        }

    @staticmethod
    def api_responses():
        """
        API responses
        - Short cache time
        - Personalized (vary by user)
        """
        return {
            'cache-control': 'private, max-age=60',
            'vary': 'Authorization, Accept-Language',
            'cdn_ttl': 60  # 1 minute
        }

    @staticmethod
    def user_generated_content():
        """
        User-generated content (profile pics, videos)
        - Medium cache time
        - Public
        """
        return {
            'cache-control': 'public, max-age=3600',
            'cdn_ttl': 3600  # 1 hour
        }

    @staticmethod
    def dynamic_pages():
        """
        Dynamic pages (HTML)
        - Edge Side Includes (ESI)
        - Fragment caching
        """
        return {
            'cache-control': 'public, max-age=300, s-maxage=600',
            'cdn_ttl': 600,  # 10 minutes
            'esi': True  # Enable ESI
        }

# Edge Computing (Cloudflare Workers, Lambda@Edge)
class EdgeComputing:
    """
    Run code at CDN edge servers
    """

    @staticmethod
    def cloudflare_worker_example():
        """
        JavaScript code running at Cloudflare edge
        """
        return """
        addEventListener('fetch', event => {
          event.respondWith(handleRequest(event.request))
        })

        async function handleRequest(request) {
          // A/B testing at edge
          const variant = Math.random() < 0.5 ? 'A' : 'B'

          // Rewrite URL based on variant
          const url = new URL(request.url)
          url.pathname = `/v${variant}${url.pathname}`

          // Fetch from origin
          const response = await fetch(url)

          // Add custom header
          const newResponse = new Response(response.body, response)
          newResponse.headers.set('X-Variant', variant)

          return newResponse
        }
        """

    @staticmethod
    def lambda_edge_example():
        """
        Python code running at AWS Lambda@Edge
        """
        return """
        def handler(event, context):
            request = event['Records'][0]['cf']['request']
            headers = request['headers']

            # Device detection at edge
            user_agent = headers.get('user-agent', [{}])[0].get('value', '')

            if 'Mobile' in user_agent:
                request['uri'] = '/mobile' + request['uri']
            else:
                request['uri'] = '/desktop' + request['uri']

            return request
        """
```

---

## Multi-Region Database Strategy

### Global Database Patterns

#### 1. Multi-Master Replication

```python
# multi_master_db.py
class MultiMasterDatabase:
    """
    Write to any region, replicate everywhere
    Used by: Cassandra, CockroachDB, YugabyteDB
    """

    def __init__(self):
        self.regions = {
            'us-east': DatabaseConnection('us-east-master'),
            'eu-west': DatabaseConnection('eu-west-master'),
            'asia-se': DatabaseConnection('asia-se-master'),
        }

    def write(self, data: dict, user_region: str):
        """
        Write to nearest region
        """
        db = self.regions[user_region]

        # Write locally
        db.execute(
            "INSERT INTO users (id, name, version) VALUES (%s, %s, %s)",
            (data['id'], data['name'], data['version'])
        )

        # Async replication to other regions
        for region, region_db in self.regions.items():
            if region != user_region:
                self._async_replicate(region_db, data)

    def _async_replicate(self, target_db, data: dict):
        """
        Async replication with conflict detection
        """
        try:
            target_db.execute(
                """
                INSERT INTO users (id, name, version)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET name = EXCLUDED.name,
                    version = EXCLUDED.version
                WHERE EXCLUDED.version > users.version
                """,
                (data['id'], data['name'], data['version'])
            )
        except Exception as e:
            # Handle conflict
            self._resolve_conflict(target_db, data)

    def _resolve_conflict(self, db, data: dict):
        """
        Resolve write conflicts (Last Write Wins)
        """
        existing = db.query("SELECT version FROM users WHERE id = %s", (data['id'],))

        if not existing or data['version'] > existing['version']:
            # New version is newer
            db.execute(
                "UPDATE users SET name = %s, version = %s WHERE id = %s",
                (data['name'], data['version'], data['id'])
            )
```

#### 2. Geo-Partitioned Database

```python
# geo_partitioned_db.py
class GeoPartitionedDatabase:
    """
    Partition data by geography
    US users → US database
    EU users → EU database
    Complies with data residency laws (GDPR)
    """

    def __init__(self):
        self.partitions = {
            'US': DatabaseConnection('us-db'),
            'EU': DatabaseConnection('eu-db'),
            'ASIA': DatabaseConnection('asia-db'),
        }

    def get_partition(self, user_id: int) -> str:
        """
        Determine partition from user ID
        """
        # User metadata table (globally replicated)
        user_meta = global_db.query(
            "SELECT region FROM user_metadata WHERE user_id = %s",
            (user_id,)
        )

        return user_meta['region']

    def write(self, user_id: int, data: dict):
        """
        Write to user's home partition
        """
        partition = self.get_partition(user_id)
        db = self.partitions[partition]

        db.execute(
            "INSERT INTO user_data (user_id, data) VALUES (%s, %s)",
            (user_id, data)
        )

    def read(self, user_id: int):
        """
        Read from user's home partition
        """
        partition = self.get_partition(user_id)
        db = self.partitions[partition]

        return db.query(
            "SELECT data FROM user_data WHERE user_id = %s",
            (user_id,)
        )

# CockroachDB: Global Database Example
class CockroachDBGlobal:
    """
    CockroachDB with geo-partitioning
    """

    @staticmethod
    def setup_geo_partitioning():
        """
        Configure CockroachDB for multi-region
        """
        # SQL
        return """
        -- Create multi-region database
        CREATE DATABASE myapp PRIMARY REGION "us-east-1"
            REGIONS "eu-west-1", "ap-southeast-1";

        -- Create table with geo-partitioning
        CREATE TABLE users (
            id UUID PRIMARY KEY,
            email STRING,
            region STRING,
            data JSONB
        ) PARTITION BY LIST (region) (
            PARTITION us VALUES IN ('us-east-1', 'us-west-2'),
            PARTITION eu VALUES IN ('eu-west-1', 'eu-central-1'),
            PARTITION asia VALUES IN ('ap-southeast-1', 'ap-northeast-1')
        );

        -- Pin partitions to regions
        ALTER PARTITION us OF TABLE users
            CONFIGURE ZONE USING constraints = '[+region=us-east-1]';

        ALTER PARTITION eu OF TABLE users
            CONFIGURE ZONE USING constraints = '[+region=eu-west-1]';

        ALTER PARTITION asia OF TABLE users
            CONFIGURE ZONE USING constraints = '[+region=ap-southeast-1]';
        """
```

---

## Load Balancing at Scale

### Global Load Balancing

```python
# global_load_balancing.py
class GlobalLoadBalancer:
    """
    Multi-layer load balancing
    """

    def __init__(self):
        # Layer 1: GeoDNS (routes to nearest region)
        self.geodns = GeoDNS()

        # Layer 2: Regional load balancer (distributes within region)
        self.regional_lb = {
            'us-east': LoadBalancer(['server1', 'server2', 'server3']),
            'eu-west': LoadBalancer(['server4', 'server5', 'server6']),
        }

        # Layer 3: Service mesh (routes between microservices)
        self.service_mesh = ServiceMesh()

class GeoDNS:
    """
    DNS-based global load balancing
    Route 53, Cloudflare
    """

    def __init__(self):
        self.region_ips = {
            'us-east': '1.2.3.4',
            'us-west': '5.6.7.8',
            'eu-west': '9.10.11.12',
            'asia-se': '13.14.15.16',
        }

    def resolve(self, domain: str, client_ip: str) -> str:
        """
        Return IP of nearest region based on client location
        """
        client_location = self._geolocate_ip(client_ip)
        nearest_region = self._find_nearest_region(client_location)

        return self.region_ips[nearest_region]

class LoadBalancer:
    """
    Regional load balancer (NGINX, HAProxy, ELB)
    """

    def __init__(self, servers: list):
        self.servers = servers
        self.current_index = 0

        # Health check
        self.healthy_servers = set(servers)

    def get_server(self, algorithm: str = 'round_robin'):
        """
        Load balancing algorithms
        """
        if algorithm == 'round_robin':
            return self._round_robin()
        elif algorithm == 'least_connections':
            return self._least_connections()
        elif algorithm == 'ip_hash':
            return self._ip_hash()
        elif algorithm == 'weighted':
            return self._weighted()

    def _round_robin(self) -> str:
        """
        Simple round-robin
        """
        healthy = list(self.healthy_servers)
        if not healthy:
            raise Exception("No healthy servers!")

        server = healthy[self.current_index % len(healthy)]
        self.current_index += 1

        return server

    def _least_connections(self) -> str:
        """
        Route to server with fewest active connections
        """
        # Query connection count from each server
        connection_counts = {
            server: self._get_connection_count(server)
            for server in self.healthy_servers
        }

        return min(connection_counts, key=connection_counts.get)

    def _ip_hash(self, client_ip: str) -> str:
        """
        Consistent hashing by client IP
        (same client always goes to same server)
        """
        import hashlib

        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        healthy = list(self.healthy_servers)

        return healthy[hash_value % len(healthy)]

    def _weighted(self) -> str:
        """
        Weighted round-robin (for heterogeneous servers)
        """
        server_weights = {
            'server1': 5,  # Powerful server
            'server2': 3,  # Medium server
            'server3': 1,  # Weak server
        }

        # Weighted random selection
        import random

        total_weight = sum(server_weights.values())
        rand = random.randint(1, total_weight)

        cumulative = 0
        for server, weight in server_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return server

# Health Checks
class HealthChecker:
    """
    Continuous health checking of backend servers
    """

    def __init__(self, servers: list, check_interval: int = 5):
        self.servers = servers
        self.check_interval = check_interval
        self.healthy_servers = set(servers)

    def start_health_checks(self):
        """
        Continuously check server health
        """
        import time
        import threading

        def health_check_loop():
            while True:
                for server in self.servers:
                    if self._check_health(server):
                        self.healthy_servers.add(server)
                    else:
                        self.healthy_servers.discard(server)
                        print(f"⚠️ Server {server} is unhealthy!")

                time.sleep(self.check_interval)

        thread = threading.Thread(target=health_check_loop, daemon=True)
        thread.start()

    def _check_health(self, server: str) -> bool:
        """
        Check if server is healthy
        """
        try:
            response = requests.get(f"http://{server}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
```

---

## Caching Strategies

### Multi-Layer Caching

```python
# multi_layer_cache.py
class PlanetScaleCaching:
    """
    Multi-layer caching strategy
    L1: Browser cache
    L2: CDN cache
    L3: Application cache (Redis)
    L4: Database query cache
    """

    def __init__(self):
        # L2: CDN (Cloudflare, Fastly)
        self.cdn = CDN()

        # L3: Application cache (Redis cluster)
        self.app_cache = RedisCluster(nodes=[
            'redis1:6379',
            'redis2:6379',
            'redis3:6379',
        ])

        # L4: Database
        self.db = DatabaseConnection()

    def get_user_profile(self, user_id: int) -> dict:
        """
        Get user profile with multi-layer caching
        """
        cache_key = f"user:{user_id}:profile"

        # L3: Check application cache (Redis)
        cached = self.app_cache.get(cache_key)
        if cached:
            print("✅ Hit: Application cache")
            return json.loads(cached)

        # L4: Database
        print("❌ Miss: Querying database")
        user = self.db.query(
            "SELECT id, name, email, bio FROM users WHERE id = %s",
            (user_id,)
        )

        if user:
            # Cache for 1 hour
            self.app_cache.setex(cache_key, 3600, json.dumps(user))

        return user

# Cache Warming
class CacheWarming:
    """
    Pre-populate cache before traffic spike
    """

    def __init__(self, cache, db):
        self.cache = cache
        self.db = db

    def warm_popular_content(self):
        """
        Warm cache with popular content
        """
        # Get top 1000 most viewed posts
        popular_posts = self.db.query(
            "SELECT id FROM posts ORDER BY views DESC LIMIT 1000"
        )

        for post in popular_posts:
            # Load into cache
            post_data = self.db.query("SELECT * FROM posts WHERE id = %s", (post['id'],))
            self.cache.setex(f"post:{post['id']}", 3600, json.dumps(post_data))

        print(f"Warmed cache with {len(popular_posts)} popular posts")

# Cache Aside Pattern
class CacheAsidePattern:
    """
    Application manages cache
    """

    def get(self, key: str):
        # Try cache
        value = cache.get(key)

        if value:
            return value

        # Cache miss - load from DB
        value = db.query(key)

        # Populate cache
        cache.set(key, value)

        return value

    def update(self, key: str, value: any):
        # Update database
        db.update(key, value)

        # Invalidate cache
        cache.delete(key)

# Write-Through Cache
class WriteThroughCache:
    """
    Write to cache and database simultaneously
    """

    def update(self, key: str, value: any):
        # Write to cache
        cache.set(key, value)

        # Write to database
        db.update(key, value)

# Write-Behind Cache
class WriteBehindCache:
    """
    Write to cache first, async write to database
    """

    def update(self, key: str, value: any):
        # Write to cache (fast)
        cache.set(key, value)

        # Queue for async DB write
        write_queue.put((key, value))

# Redis Cluster for Planet-Scale
class RedisClusterSetup:
    """
    Redis cluster with sharding and replication
    """

    @staticmethod
    def setup_redis_cluster():
        """
        6-node Redis cluster (3 masters + 3 replicas)
        """
        # docker-compose.yml
        return """
        version: '3'
        services:
          redis-1:
            image: redis:7-alpine
            command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
            ports:
              - "7000:6379"

          redis-2:
            image: redis:7-alpine
            command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
            ports:
              - "7001:6379"

          redis-3:
            image: redis:7-alpine
            command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
            ports:
              - "7002:6379"

          redis-4:
            image: redis:7-alpine
            command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
            ports:
              - "7003:6379"

          redis-5:
            image: redis:7-alpine
            command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
            ports:
              - "7004:6379"

          redis-6:
            image: redis:7-alpine
            command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
            ports:
              - "7005:6379"
        """

    @staticmethod
    def create_cluster():
        """
        Initialize Redis cluster
        """
        # bash
        return """
        redis-cli --cluster create \
          127.0.0.1:7000 \
          127.0.0.1:7001 \
          127.0.0.1:7002 \
          127.0.0.1:7003 \
          127.0.0.1:7004 \
          127.0.0.1:7005 \
          --cluster-replicas 1
        """
```

---

## Message Queue at Scale

### Kafka at Planet-Scale

```python
# kafka_planet_scale.py
class PlanetScaleKafka:
    """
    Kafka setup for billions of messages per second
    """

    def __init__(self):
        self.kafka_clusters = {
            'us-east': ['kafka1:9092', 'kafka2:9092', 'kafka3:9092'],
            'eu-west': ['kafka4:9092', 'kafka5:9092', 'kafka6:9092'],
            'asia-se': ['kafka7:9092', 'kafka8:9092', 'kafka9:9092'],
        }

        self.producers = {}
        self.consumers = {}

    def get_producer(self, region: str):
        """
        Get Kafka producer for region
        """
        if region not in self.producers:
            from kafka import KafkaProducer

            self.producers[region] = KafkaProducer(
                bootstrap_servers=self.kafka_clusters[region],
                acks='all',  # Wait for all replicas
                compression_type='snappy',
                batch_size=16384,  # Batch messages
                linger_ms=10,  # Wait up to 10ms to batch
                buffer_memory=33554432,  # 32MB buffer
            )

        return self.producers[region]

    def produce_message(self, topic: str, message: dict, region: str):
        """
        Produce message to Kafka
        """
        producer = self.get_producer(region)

        # Async send
        future = producer.send(
            topic,
            value=json.dumps(message).encode('utf-8'),
            key=str(message.get('user_id')).encode('utf-8')  # Partition by user_id
        )

        # Optional: Wait for acknowledgment
        try:
            record_metadata = future.get(timeout=10)
            print(f"Message sent to partition {record_metadata.partition} at offset {record_metadata.offset}")
        except Exception as e:
            print(f"Failed to send message: {e}")

# Kafka Topic Configuration
class KafkaTopicConfig:
    """
    Kafka topic configuration for planet-scale
    """

    @staticmethod
    def create_high_throughput_topic():
        """
        Create topic optimized for high throughput
        """
        # kafka-topics command
        return """
        kafka-topics --create \
          --bootstrap-server localhost:9092 \
          --topic user-events \
          --partitions 100 \
          --replication-factor 3 \
          --config retention.ms=86400000 \
          --config compression.type=snappy \
          --config max.message.bytes=1048576 \
          --config segment.bytes=1073741824
        """

    @staticmethod
    def configure_for_low_latency():
        """
        Configure for low latency
        """
        return {
            'partitions': 50,
            'replication_factor': 2,  # Lower replication for speed
            'min_insync_replicas': 1,
            'acks': 1,  # Only wait for leader
            'compression': None,  # No compression for speed
        }

    @staticmethod
    def configure_for_durability():
        """
        Configure for maximum durability
        """
        return {
            'partitions': 20,
            'replication_factor': 3,
            'min_insync_replicas': 2,
            'acks': 'all',  # Wait for all replicas
            'retention_ms': 604800000,  # 7 days
        }

# Cross-Region Kafka Replication
class CrossRegionKafkaReplication:
    """
    Mirror Kafka topics across regions
    """

    @staticmethod
    def setup_mirrormaker():
        """
        Kafka MirrorMaker 2.0 setup
        """
        # Configuration
        return """
        # mm2.properties
        clusters = us-east, eu-west
        us-east.bootstrap.servers = kafka1:9092,kafka2:9092
        eu-west.bootstrap.servers = kafka4:9092,kafka5:9092

        # Replication flows
        us-east->eu-west.enabled = true
        us-east->eu-west.topics = user-events, orders

        eu-west->us-east.enabled = true
        eu-west->us-east.topics = user-events, orders

        # Replication factor
        replication.factor = 2
        """
```

---

## Real-World Architectures

### 1. Google Search Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Google Search (Simplified)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Query → GeoDNS → Nearest Data Center                 │
│                            ↓                                │
│                   Global Load Balancer (GFE)                │
│                            ↓                                │
│              ┌─────────────┴─────────────┐                  │
│              │                           │                  │
│         Web Servers              Index Servers              │
│              │                           │                  │
│              └─────────────┬─────────────┘                  │
│                            ↓                                │
│                      Distributed Index                      │
│                   (Sharded by keywords)                     │
│                   100,000+ servers                          │
│                            ↓                                │
│                      Ranking & ML                           │
│                            ↓                                │
│                    Results Aggregation                      │
│                            ↓                                │
│                    Response (< 200ms)                       │
└─────────────────────────────────────────────────────────────┘
```

**Key Techniques:**
- **Massive sharding** - Index distributed across 100,000+ servers
- **In-memory caching** - Hot queries cached in RAM
- **Predictive pre-fetching** - Predict what user will search
- **Colossus** - Distributed file system (successor to GFS)
- **Bigtable** - Distributed database for web index

### 2. Facebook/Meta Architecture

```python
# facebook_architecture.py
"""
Facebook Architecture (Simplified)

Scale:
- 3 billion daily active users
- 100+ billion messages per day
- 350+ million photos uploaded per day
"""

class FacebookNewsfeedArchitecture:
    """
    How Facebook generates newsfeeds
    """

    def __init__(self):
        # TAO: Facebook's distributed graph database
        self.tao = TAODatabase()

        # Memcached: Massive caching layer
        self.memcache = MemcachedCluster()

        # Leaf: Load balancing and routing
        self.leaf = LeafRouter()

    def get_newsfeed(self, user_id: int) -> list:
        """
        Generate newsfeed for user
        """
        # 1. Get user's friends (from TAO)
        cache_key = f"friends:{user_id}"
        friends = self.memcache.get(cache_key)

        if not friends:
            friends = self.tao.get_friends(user_id)
            self.memcache.set(cache_key, friends, ttl=3600)

        # 2. Get recent posts from friends (fan-out on read)
        posts = []
        for friend_id in friends[:5000]:  # Limit to 5000 friends
            friend_posts = self.get_user_posts(friend_id, limit=10)
            posts.extend(friend_posts)

        # 3. Rank posts (EdgeRank algorithm)
        ranked_posts = self.rank_posts(posts, user_id)

        return ranked_posts[:50]

    def rank_posts(self, posts: list, user_id: int) -> list:
        """
        Rank posts using ML model (EdgeRank)
        """
        scored_posts = []

        for post in posts:
            # Features
            affinity = self.calculate_affinity(user_id, post['author_id'])
            weight = self.get_edge_weight(post['type'])
            time_decay = self.calculate_time_decay(post['created_at'])

            # EdgeRank score
            score = affinity * weight * time_decay

            scored_posts.append((score, post))

        # Sort by score
        scored_posts.sort(reverse=True, key=lambda x: x[0])

        return [post for score, post in scored_posts]

class TAODatabase:
    """
    TAO: The Associations and Objects
    Facebook's distributed graph database
    """

    def __init__(self):
        # Sharded across thousands of MySQL servers
        self.shards = [MySQLConnection(f"shard_{i}") for i in range(1000)]

        # Heavy caching
        self.cache = MemcachedCluster()

    def get_friends(self, user_id: int) -> list:
        """
        Get user's friends
        """
        # Check cache
        friends = self.cache.get(f"friends:{user_id}")
        if friends:
            return friends

        # Query database
        shard = self.get_shard(user_id)
        friends = shard.query(
            "SELECT friend_id FROM friendships WHERE user_id = %s",
            (user_id,)
        )

        # Cache
        self.cache.set(f"friends:{user_id}", friends, ttl=3600)

        return friends

    def get_shard(self, user_id: int):
        """Get shard for user_id"""
        return self.shards[user_id % len(self.shards)]

# Haystack: Facebook's photo storage
class HaystackPhotoStorage:
    """
    Haystack: Store billions of photos efficiently
    """

    def __init__(self):
        # Physical needles (files containing many photos)
        self.haystack_store = "/mnt/haystack"

        # Index (maps photo_id → needle_id + offset)
        self.index = {}

    def store_photo(self, photo_id: int, photo_data: bytes):
        """
        Store photo in haystack
        Multiple photos per file to reduce metadata overhead
        """
        # Find needle (file) with space
        needle_id = self.find_needle_with_space()

        # Append photo to needle
        with open(f"{self.haystack_store}/needle_{needle_id}", 'ab') as f:
            offset = f.tell()
            f.write(photo_data)

        # Update index
        self.index[photo_id] = {
            'needle_id': needle_id,
            'offset': offset,
            'size': len(photo_data)
        }

    def get_photo(self, photo_id: int) -> bytes:
        """
        Retrieve photo from haystack
        """
        # Lookup index
        metadata = self.index[photo_id]

        # Read from needle
        with open(f"{self.haystack_store}/needle_{metadata['needle_id']}", 'rb') as f:
            f.seek(metadata['offset'])
            photo_data = f.read(metadata['size'])

        return photo_data
```

### 3. Netflix Architecture

```python
# netflix_architecture.py
"""
Netflix Architecture

Scale:
- 250+ million subscribers
- 1+ billion hours watched per week
- 15+ petabytes of content
"""

class NetflixArchitecture:
    """
    Netflix microservices architecture
    """

    def __init__(self):
        # 700+ microservices
        self.microservices = self._init_microservices()

        # Zuul: API Gateway
        self.api_gateway = ZuulGateway()

        # Eureka: Service discovery
        self.service_discovery = EurekaServer()

        # Hystrix: Circuit breaker
        self.circuit_breaker = HystrixCircuitBreaker()

    def _init_microservices(self):
        """
        Initialize Netflix microservices
        """
        return {
            'user-service': MicroService('user-service'),
            'content-service': MicroService('content-service'),
            'recommendation-service': MicroService('recommendation-service'),
            'playback-service': MicroService('playback-service'),
            'billing-service': MicroService('billing-service'),
            # ... 695 more services
        }

    def get_homepage(self, user_id: int):
        """
        Generate Netflix homepage
        Calls multiple microservices in parallel
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Parallel requests to microservices
            futures = {
                executor.submit(self.get_user_profile, user_id): 'profile',
                executor.submit(self.get_recommendations, user_id): 'recommendations',
                executor.submit(self.get_continue_watching, user_id): 'continue_watching',
                executor.submit(self.get_trending, user_id): 'trending',
                executor.submit(self.get_new_releases, user_id): 'new_releases',
            }

            results = {}
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    # Fallback if service fails
                    results[key] = self._get_fallback(key)

        return results

class NetflixCDN:
    """
    Netflix Open Connect CDN
    """

    def __init__(self):
        # 17,000+ servers in 1000+ locations
        self.edge_servers = self._init_edge_servers()

    def stream_video(self, video_id: str, user_location: tuple):
        """
        Stream video from nearest Open Connect server
        """
        # Find nearest edge server
        nearest_edge = self.find_nearest_edge(user_location)

        # Check if video is cached
        if nearest_edge.has_video(video_id):
            # Stream from edge (low latency)
            return nearest_edge.stream(video_id)
        else:
            # Fetch from S3 and cache
            video_data = s3.get_object(Bucket='netflix-videos', Key=video_id)
            nearest_edge.cache_video(video_id, video_data)
            return nearest_edge.stream(video_id)

class NetflixRecommendations:
    """
    Netflix recommendation engine
    """

    def __init__(self):
        self.collaborative_filtering = CollaborativeFiltering()
        self.content_based = ContentBasedFiltering()
        self.deep_learning = DeepLearningModel()

    def get_recommendations(self, user_id: int, limit: int = 50):
        """
        Hybrid recommendation system
        """
        # 1. Collaborative filtering (what similar users watched)
        collab_recs = self.collaborative_filtering.recommend(user_id, limit=100)

        # 2. Content-based (similar to what user watched)
        content_recs = self.content_based.recommend(user_id, limit=100)

        # 3. Deep learning model (neural network)
        dl_recs = self.deep_learning.recommend(user_id, limit=100)

        # 4. Ensemble (combine all models)
        final_recs = self.ensemble([collab_recs, content_recs, dl_recs], weights=[0.4, 0.3, 0.3])

        return final_recs[:limit]
```

### 4. Amazon E-commerce Architecture

```python
# amazon_architecture.py
"""
Amazon Architecture

Scale:
- 300+ million customers
- 12+ million products
- 4,000+ orders per second (peak)
"""

class AmazonArchitecture:
    """
    Amazon's service-oriented architecture
    """

    def __init__(self):
        # Thousands of microservices
        self.services = {
            'product-catalog': ProductCatalogService(),
            'inventory': InventoryService(),
            'pricing': PricingService(),
            'cart': ShoppingCartService(),
            'checkout': CheckoutService(),
            'payment': PaymentService(),
            'fulfillment': FulfillmentService(),
            'shipping': ShippingService(),
            'recommendations': RecommendationService(),
        }

        # DynamoDB: Primary database
        self.dynamodb = DynamoDB()

        # S3: Object storage
        self.s3 = S3()

    def place_order(self, user_id: int, items: list):
        """
        Place order (orchestrates multiple services)
        """
        # 1. Reserve inventory
        for item in items:
            inventory_reserved = self.services['inventory'].reserve(
                product_id=item['product_id'],
                quantity=item['quantity']
            )

            if not inventory_reserved:
                raise OutOfStockError(f"Product {item['product_id']} out of stock")

        # 2. Calculate total price
        total_price = sum(
            self.services['pricing'].get_price(item['product_id']) * item['quantity']
            for item in items
        )

        # 3. Process payment
        payment_result = self.services['payment'].charge(
            user_id=user_id,
            amount=total_price
        )

        if not payment_result['success']:
            # Rollback inventory reservation
            self.services['inventory'].release(items)
            raise PaymentError("Payment failed")

        # 4. Create order
        order_id = self.services['checkout'].create_order(
            user_id=user_id,
            items=items,
            total_price=total_price
        )

        # 5. Initiate fulfillment
        self.services['fulfillment'].fulfill_order(order_id)

        return order_id

class DynamoDBAtScale:
    """
    DynamoDB: Amazon's NoSQL database
    Powers Amazon.com shopping cart
    """

    def __init__(self):
        import boto3
        self.dynamodb = boto3.resource('dynamodb')

        # Shopping cart table
        self.cart_table = self.dynamodb.Table('shopping-carts')

    def add_to_cart(self, user_id: str, product_id: str, quantity: int):
        """
        Add item to shopping cart
        """
        self.cart_table.update_item(
            Key={'user_id': user_id},
            UpdateExpression='SET #items.#product_id = :quantity',
            ExpressionAttributeNames={
                '#items': 'items',
                '#product_id': product_id
            },
            ExpressionAttributeValues={
                ':quantity': quantity
            }
        )

    def get_cart(self, user_id: str):
        """
        Get user's shopping cart
        """
        response = self.cart_table.get_item(Key={'user_id': user_id})
        return response.get('Item', {})

# Amazon's recommendation engine
class AmazonRecommendations:
    """
    Item-to-item collaborative filtering
    """

    def __init__(self):
        # Pre-computed similarity matrix
        self.similarity_matrix = {}

    def recommend(self, user_id: str, limit: int = 20):
        """
        Recommend products based on:
        1. Items in cart
        2. Previously purchased items
        3. Browsing history
        """
        # Get user's cart
        cart_items = self.get_cart_items(user_id)

        # Get similar items
        recommendations = []
        for item in cart_items:
            similar_items = self.get_similar_items(item['product_id'])
            recommendations.extend(similar_items)

        # Deduplicate and rank
        ranked_recs = self.rank_recommendations(recommendations, user_id)

        return ranked_recs[:limit]

    def get_similar_items(self, product_id: str):
        """
        Get items similar to this product
        Uses pre-computed similarity scores
        """
        return self.similarity_matrix.get(product_id, [])
```

---

## Observability & Monitoring

### Metrics at Scale

```python
# observability.py
from prometheus_client import Counter, Histogram, Gauge
import time

class PlanetScaleObservability:
    """
    Observability for planet-scale systems
    """

    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status', 'region']
        )

        self.request_latency = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency',
            ['method', 'endpoint', 'region']
        )

        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['region']
        )

        self.error_rate = Counter(
            'errors_total',
            'Total errors',
            ['type', 'service', 'region']
        )

    def track_request(self, method: str, endpoint: str, region: str):
        """
        Track HTTP request
        """
        start_time = time.time()

        try:
            # Process request
            response = self.handle_request(method, endpoint)

            # Record success
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=200,
                region=region
            ).inc()

            return response

        except Exception as e:
            # Record error
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=500,
                region=region
            ).inc()

            self.error_rate.labels(
                type=type(e).__name__,
                service='api',
                region=region
            ).inc()

            raise

        finally:
            # Record latency
            duration = time.time() - start_time
            self.request_latency.labels(
                method=method,
                endpoint=endpoint,
                region=region
            ).observe(duration)

# Distributed Tracing (Jaeger, Zipkin)
class DistributedTracing:
    """
    Trace requests across microservices
    """

    def __init__(self):
        from jaeger_client import Config

        config = Config(
            config={
                'sampler': {'type': 'const', 'param': 1},
                'logging': True,
            },
            service_name='api-service',
        )

        self.tracer = config.initialize_tracer()

    def trace_request(self, request_id: str):
        """
        Trace request across services
        """
        with self.tracer.start_span('handle_request') as span:
            span.set_tag('request_id', request_id)

            # Call service 1
            with self.tracer.start_span('call_user_service', child_of=span) as child_span:
                user_data = self.call_user_service()
                child_span.set_tag('user_id', user_data['id'])

            # Call service 2
            with self.tracer.start_span('call_product_service', child_of=span) as child_span:
                products = self.call_product_service()
                child_span.set_tag('product_count', len(products))

            return {'user': user_data, 'products': products}

# Logging at Scale (Elasticsearch)
class CentralizedLogging:
    """
    Centralized logging with ELK stack
    """

    def __init__(self):
        from elasticsearch import Elasticsearch

        self.es = Elasticsearch(['es1:9200', 'es2:9200', 'es3:9200'])

    def log(self, level: str, message: str, context: dict = None):
        """
        Log to Elasticsearch
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'service': 'api-service',
            'region': os.getenv('REGION'),
            'host': os.getenv('HOSTNAME'),
            'context': context or {}
        }

        self.es.index(index='logs', document=log_entry)

# Alerting
class AlertingSystem:
    """
    Alert on critical issues
    """

    def __init__(self):
        self.slack_webhook = "https://hooks.slack.com/services/..."
        self.pagerduty_key = "xxx"

    def alert_high_error_rate(self, error_rate: float, region: str):
        """
        Alert if error rate exceeds threshold
        """
        if error_rate > 0.05:  # 5% error rate
            self.send_slack_alert(
                f"🚨 High error rate in {region}: {error_rate*100:.2f}%"
            )

            if error_rate > 0.10:  # 10% error rate
                self.page_oncall_engineer(
                    f"CRITICAL: Error rate in {region}: {error_rate*100:.2f}%"
                )

    def send_slack_alert(self, message: str):
        """Send alert to Slack"""
        requests.post(self.slack_webhook, json={'text': message})

    def page_oncall_engineer(self, message: str):
        """Page on-call engineer via PagerDuty"""
        requests.post(
            'https://events.pagerduty.com/v2/enqueue',
            headers={'Authorization': f'Token token={self.pagerduty_key}'},
            json={
                'event_action': 'trigger',
                'payload': {
                    'summary': message,
                    'severity': 'critical',
                    'source': 'monitoring-system'
                }
            }
        )
```

---

## Disaster Recovery

### Multi-Region Failover

```python
# disaster_recovery.py
class DisasterRecoveryPlan:
    """
    Disaster recovery for planet-scale systems
    RTO (Recovery Time Objective): < 5 minutes
    RPO (Recovery Point Objective): < 1 minute
    """

    def __init__(self):
        self.primary_region = 'us-east-1'
        self.failover_region = 'us-west-2'

        self.health_check_interval = 10  # seconds

    def continuous_health_check(self):
        """
        Continuously monitor primary region health
        """
        import time

        while True:
            if not self.is_region_healthy(self.primary_region):
                print(f"⚠️ Primary region {self.primary_region} is down!")
                self.initiate_failover()

            time.sleep(self.health_check_interval)

    def is_region_healthy(self, region: str) -> bool:
        """
        Check if region is healthy
        """
        try:
            # Check multiple health indicators
            checks = [
                self.check_api_health(region),
                self.check_database_health(region),
                self.check_cache_health(region),
            ]

            return all(checks)
        except:
            return False

    def initiate_failover(self):
        """
        Failover to secondary region
        """
        print(f"Initiating failover from {self.primary_region} to {self.failover_region}")

        # 1. Update global load balancer (Route 53)
        self.update_global_lb()

        # 2. Promote read replicas to masters
        self.promote_database_replicas()

        # 3. Update application configuration
        self.update_app_config()

        # 4. Notify team
        self.send_alert("Failover completed successfully")

        # Swap regions
        self.primary_region, self.failover_region = self.failover_region, self.primary_region

    def update_global_lb(self):
        """
        Update DNS to point to failover region
        """
        # AWS Route 53 health checks automatically failover
        pass

    def promote_database_replicas(self):
        """
        Promote read replicas to master
        """
        # PostgreSQL example
        db_replica = DatabaseConnection(f"{self.failover_region}-replica")
        db_replica.execute("SELECT pg_promote()")

# Backup Strategy
class BackupStrategy:
    """
    Continuous backups for disaster recovery
    """

    def __init__(self):
        self.s3 = boto3.client('s3')
        self.backup_bucket = 'database-backups'

    def continuous_backup(self):
        """
        Continuous database backups
        """
        # PostgreSQL Write-Ahead Log (WAL) archiving
        db.execute("ALTER SYSTEM SET archive_mode = on")
        db.execute("ALTER SYSTEM SET archive_command = 'aws s3 cp %p s3://wal-archive/%f'")

        # Point-in-time recovery (PITR)
        # Can restore to any point in last 30 days

    def restore_to_point_in_time(self, timestamp: datetime):
        """
        Restore database to specific point in time
        """
        # 1. Restore latest base backup
        self.restore_base_backup()

        # 2. Replay WAL logs up to timestamp
        self.replay_wal_logs(timestamp)

        print(f"Database restored to {timestamp}")
```

---

## Cost Optimization

```python
# cost_optimization.py
class CostOptimization:
    """
    Strategies to reduce infrastructure costs
    """

    @staticmethod
    def use_spot_instances():
        """
        Use AWS Spot Instances (up to 90% cheaper)
        Good for: Batch processing, CI/CD, stateless services
        """
        # Kubernetes with Spot instances
        return """
        apiVersion: v1
        kind: Node
        metadata:
          labels:
            node.kubernetes.io/lifecycle: spot
        """

    @staticmethod
    def auto_scaling():
        """
        Auto-scale based on demand
        Save costs during low traffic
        """
        return """
        # Kubernetes Horizontal Pod Autoscaler
        apiVersion: autoscaling/v2
        kind: HorizontalPodAutoscaler
        metadata:
          name: api-hpa
        spec:
          scaleTargetRef:
            apiVersion: apps/v1
            kind: Deployment
            name: api
          minReplicas: 10
          maxReplicas: 1000
          metrics:
          - type: Resource
            resource:
              name: cpu
              target:
                type: Utilization
                averageUtilization: 70
        """

    @staticmethod
    def reserved_capacity():
        """
        Reserve capacity for baseline (up to 75% discount)
        Use on-demand for spikes
        """
        baseline = 100  # servers
        peak = 500  # servers

        # Reserve baseline capacity (3-year commitment)
        reserved_savings = 0.75  # 75% discount

        # On-demand for (peak - baseline)
        on_demand = peak - baseline

        return {
            'reserved': baseline,
            'on_demand': on_demand,
            'estimated_savings': f"{reserved_savings * 100}%"
        }

    @staticmethod
    def tiered_storage():
        """
        Use different storage tiers
        """
        return {
            'hot_data': 'S3 Standard (frequently accessed)',
            'warm_data': 'S3 Infrequent Access (30-90 days old)',
            'cold_data': 'S3 Glacier (> 90 days old)',
            'archive': 'S3 Deep Archive (rarely accessed)'
        }
```

---

## Key Takeaways

### Building Planet-Scale Systems

1. **Start Regional** → **Go Multi-Regional** → **Go Global**
2. **Cache Everything** (CDN, application, database)
3. **Embrace Eventual Consistency** (where possible)
4. **Shard Data** (horizontally partition)
5. **Replicate** (for availability and performance)
6. **Monitor Everything** (you can't fix what you can't see)
7. **Plan for Failure** (chaos engineering)
8. **Optimize Costs** (or you'll go bankrupt!)

### Architecture Evolution

```
Stage 1: Single Server
    ↓
Stage 2: Load Balancer + Multiple Servers
    ↓
Stage 3: Add Database Replicas
    ↓
Stage 4: Add Caching (Redis)
    ↓
Stage 5: Microservices
    ↓
Stage 6: Sharding
    ↓
Stage 7: Multi-Region
    ↓
Stage 8: Global CDN
    ↓
Stage 9: Planet-Scale! 🌍
```

### Common Mistakes

❌ **Over-engineering early** - Don't build for 1B users when you have 1000
❌ **Ignoring monitoring** - You'll be blind when things fail
❌ **No disaster recovery plan** - Hope is not a strategy
❌ **Not testing at scale** - Load test before launch
❌ **Tight coupling** - Makes it impossible to scale independently

---

**Remember:** Planet-scale is not just about technology - it's about people, processes, and culture! 🚀

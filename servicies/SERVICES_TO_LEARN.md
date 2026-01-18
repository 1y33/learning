# Complete Services & Technologies to Learn

## Table of Contents
1. [Container Orchestration](#container-orchestration)
2. [Message Queues & Streaming](#message-queues--streaming)
3. [Monitoring & Observability](#monitoring--observability)
4. [Databases](#databases)
5. [Caching](#caching)
6. [CI/CD](#cicd)
7. [Infrastructure as Code](#infrastructure-as-code)
8. [Service Mesh & API Gateway](#service-mesh--api-gateway)
9. [Security & Secrets Management](#security--secrets-management)
10. [Search Engines](#search-engines)
11. [Load Balancing & Reverse Proxy](#load-balancing--reverse-proxy)
12. [Storage & File Systems](#storage--file-systems)
13. [Logging & Log Management](#logging--log-management)
14. [Tracing](#tracing)
15. [Service Discovery](#service-discovery)
16. [Testing Tools](#testing-tools)
17. [Cloud Platforms](#cloud-platforms)

---

## Container Orchestration

### ⭐ Kubernetes (K8s)
**What it is:** Container orchestration platform for automating deployment, scaling, and management of containerized applications.

**Why learn it:** Industry standard for container orchestration. Used by 88% of Fortune 100 companies.

**Key concepts:**
- Pods, Deployments, Services
- StatefulSets, DaemonSets
- ConfigMaps, Secrets
- Ingress, NetworkPolicies
- Helm charts
- Operators

**Resources:**
- [Official Docs](https://kubernetes.io/docs/)
- [Kubernetes the Hard Way](https://github.com/kelseyhightower/kubernetes-the-hard-way)

---

### Helm
**What it is:** Package manager for Kubernetes. Think "npm for Kubernetes".

**Why learn it:** Simplifies Kubernetes deployments and configuration management.

**Key concepts:**
- Charts (packages)
- Values files
- Templates
- Releases
- Repositories

**Resources:**
- [Helm Docs](https://helm.sh/docs/)

---

### Docker Swarm
**What it is:** Docker's native container orchestration tool.

**Why learn it:** Simpler alternative to Kubernetes for smaller deployments.

**Key concepts:**
- Services
- Stacks
- Swarm mode
- Overlay networks

---

### Nomad
**What it is:** HashiCorp's workload orchestrator (not just containers).

**Why learn it:** Simpler than K8s, supports more workload types.

**Key concepts:**
- Jobs
- Task groups
- Allocation
- Consul integration

---

## Message Queues & Streaming

### ⭐ RabbitMQ
**What it is:** Message broker using AMQP protocol.

**Why learn it:** Most popular message queue for asynchronous communication.

**Key concepts:**
- Exchanges (fanout, direct, topic, headers)
- Queues
- Bindings
- Virtual hosts
- Dead letter queues
- Message acknowledgments

**Use cases:**
- Task queues
- Microservices communication
- Event-driven architecture

**Resources:**
- [RabbitMQ Tutorials](https://www.rabbitmq.com/tutorials)

---

### ⭐ Apache Kafka
**What it is:** Distributed event streaming platform.

**Why learn it:** Industry standard for high-throughput data streaming.

**Key concepts:**
- Topics
- Partitions
- Producers, Consumers
- Consumer Groups
- Kafka Streams
- Kafka Connect
- Zookeeper (being replaced by KRaft)

**Use cases:**
- Event sourcing
- Log aggregation
- Real-time analytics
- Stream processing

**Resources:**
- [Kafka Docs](https://kafka.apache.org/documentation/)

---

### Redis Pub/Sub
**What it is:** Redis publish/subscribe messaging.

**Why learn it:** Simple, fast messaging for real-time features.

**Key concepts:**
- Publishers
- Subscribers
- Channels
- Pattern matching

---

### NATS
**What it is:** Cloud-native messaging system.

**Why learn it:** Lightweight, high-performance messaging.

**Key concepts:**
- Core NATS
- JetStream (persistence)
- Subjects
- Request-reply

---

### Amazon SQS
**What it is:** AWS managed message queue service.

**Why learn it:** Serverless, fully managed queue.

**Key concepts:**
- Standard queues
- FIFO queues
- Dead letter queues
- Visibility timeout

---

### Apache Pulsar
**What it is:** Cloud-native distributed messaging and streaming platform.

**Why learn it:** Alternative to Kafka with better multi-tenancy.

**Key concepts:**
- Topics
- Subscriptions
- Tenants & Namespaces
- Pulsar Functions

---

## Monitoring & Observability

### ⭐ Prometheus
**What it is:** Open-source monitoring and alerting toolkit.

**Why learn it:** Industry standard for metrics collection and monitoring.

**Key concepts:**
- Time-series database
- PromQL (query language)
- Exporters
- Service discovery
- Alertmanager
- Recording rules

**Use cases:**
- Infrastructure monitoring
- Application metrics
- Alerting

**Resources:**
- [Prometheus Docs](https://prometheus.io/docs/)

---

### ⭐ Grafana
**What it is:** Analytics and monitoring platform with beautiful dashboards.

**Why learn it:** Best visualization tool for metrics.

**Key concepts:**
- Dashboards
- Data sources (Prometheus, InfluxDB, etc.)
- Panels
- Variables
- Alerting
- Grafana Loki (logs)
- Grafana Tempo (tracing)

**Resources:**
- [Grafana Tutorials](https://grafana.com/tutorials/)

---

### Datadog
**What it is:** SaaS monitoring and analytics platform.

**Why learn it:** All-in-one monitoring solution (paid).

**Key concepts:**
- Metrics, logs, traces
- APM (Application Performance Monitoring)
- Infrastructure monitoring
- Dashboards

---

### New Relic
**What it is:** Application performance monitoring (APM) platform.

**Why learn it:** Popular APM tool for production monitoring.

---

### Nagios
**What it is:** Classic infrastructure monitoring system.

**Why learn it:** Still used in many enterprises.

---

### Zabbix
**What it is:** Enterprise-level monitoring solution.

**Why learn it:** Comprehensive monitoring with alerting.

---

### Elastic APM
**What it is:** Application performance monitoring from Elastic Stack.

**Why learn it:** Integrated with Elasticsearch ecosystem.

---

## Databases

### SQL Databases

#### ⭐ PostgreSQL
**What it is:** Advanced open-source relational database.

**Why learn it:** Most powerful open-source RDBMS.

**Key concepts:**
- ACID transactions
- JSON/JSONB support
- Full-text search
- Extensions (PostGIS, TimescaleDB)
- Replication
- Partitioning

---

#### MySQL / MariaDB
**What it is:** Popular open-source relational database.

**Why learn it:** Widely used, especially in web applications.

---

#### Microsoft SQL Server
**What it is:** Enterprise relational database from Microsoft.

**Why learn it:** Dominant in enterprise Windows environments.

---

#### Oracle Database
**What it is:** Enterprise-grade database from Oracle.

**Why learn it:** Used in large enterprises.

---

### NoSQL Databases

#### ⭐ MongoDB
**What it is:** Document-oriented NoSQL database.

**Why learn it:** Most popular NoSQL database.

**Key concepts:**
- Documents (JSON-like)
- Collections
- Aggregation pipeline
- Indexes
- Replication (replica sets)
- Sharding

---

#### ⭐ Redis
**What it is:** In-memory data structure store.

**Why learn it:** Super fast, great for caching and sessions.

**Key concepts:**
- Data structures (strings, hashes, lists, sets, sorted sets)
- Pub/Sub
- Persistence (RDB, AOF)
- Clustering
- Sentinel (high availability)
- Redis Streams

---

#### Cassandra
**What it is:** Wide-column distributed NoSQL database.

**Why learn it:** Handles massive amounts of data across many servers.

**Key concepts:**
- Column families
- Consistent hashing
- Eventual consistency
- CQL (Cassandra Query Language)

---

#### DynamoDB
**What it is:** AWS fully managed NoSQL database.

**Why learn it:** Serverless, highly scalable.

---

#### CouchDB
**What it is:** Document-oriented database with HTTP API.

**Why learn it:** Offline-first capabilities.

---

#### Neo4j
**What it is:** Graph database.

**Why learn it:** Best for relationship-heavy data.

**Key concepts:**
- Nodes
- Relationships
- Properties
- Cypher query language

---

#### InfluxDB
**What it is:** Time-series database.

**Why learn it:** Optimized for time-series data (metrics, events).

**Key concepts:**
- Measurements
- Tags and fields
- Retention policies
- Continuous queries

---

#### TimescaleDB
**What it is:** Time-series database built on PostgreSQL.

**Why learn it:** SQL + time-series optimization.

---

## Caching

### ⭐ Redis
(See above in NoSQL Databases)

**Caching strategies:**
- Cache-aside
- Write-through
- Write-behind
- Refresh-ahead

---

### Memcached
**What it is:** Distributed memory caching system.

**Why learn it:** Simple, fast in-memory caching.

**Key concepts:**
- Key-value storage
- LRU eviction
- Distributed caching

---

### Varnish
**What it is:** HTTP reverse proxy cache.

**Why learn it:** Excellent for caching HTTP responses.

---

### Hazelcast
**What it is:** In-memory data grid.

**Why learn it:** Distributed caching with computation.

---

## CI/CD

### ⭐ Jenkins
**What it is:** Open-source automation server.

**Why learn it:** Most widely used CI/CD tool.

**Key concepts:**
- Pipelines (Declarative, Scripted)
- Jobs
- Agents/Slaves
- Plugins
- Jenkinsfile

---

### ⭐ GitLab CI/CD
**What it is:** Built-in CI/CD in GitLab.

**Why learn it:** Integrated with Git, easy to use.

**Key concepts:**
- .gitlab-ci.yml
- Pipelines
- Jobs, Stages
- Runners
- Artifacts

---

### GitHub Actions
**What it is:** CI/CD platform built into GitHub.

**Why learn it:** Growing rapidly, great for open-source.

**Key concepts:**
- Workflows
- Actions
- Runners
- Secrets
- Matrix builds

---

### CircleCI
**What it is:** Cloud-based CI/CD platform.

**Why learn it:** Fast, easy to configure.

---

### Travis CI
**What it is:** CI service for GitHub projects.

**Why learn it:** Popular for open-source.

---

### ArgoCD
**What it is:** GitOps continuous delivery tool for Kubernetes.

**Why learn it:** Declarative Kubernetes deployments.

**Key concepts:**
- GitOps
- Applications
- Sync policies
- Rollbacks

---

### Flux
**What it is:** GitOps tool for Kubernetes.

**Why learn it:** Alternative to ArgoCD.

---

### Spinnaker
**What it is:** Multi-cloud continuous delivery platform.

**Why learn it:** Advanced deployment strategies.

---

### Tekton
**What it is:** Kubernetes-native CI/CD framework.

**Why learn it:** Cloud-native pipelines.

---

## Infrastructure as Code

### ⭐ Terraform
**What it is:** Infrastructure as Code tool.

**Why learn it:** Industry standard for IaC.

**Key concepts:**
- HCL (HashiCorp Configuration Language)
- Providers
- Resources
- State
- Modules
- Workspaces

**Resources:**
- [Terraform Docs](https://www.terraform.io/docs)

---

### Ansible
**What it is:** Configuration management and automation tool.

**Why learn it:** Agentless, simple YAML syntax.

**Key concepts:**
- Playbooks
- Roles
- Inventory
- Modules
- Tasks

---

### Pulumi
**What it is:** IaC using real programming languages.

**Why learn it:** Write infrastructure in Python, TypeScript, Go, etc.

---

### CloudFormation
**What it is:** AWS native IaC tool.

**Why learn it:** Tight AWS integration.

---

### Chef
**What it is:** Configuration management tool.

**Why learn it:** Enterprise configuration management.

---

### Puppet
**What it is:** Configuration management tool.

**Why learn it:** Large-scale infrastructure automation.

---

### SaltStack
**What it is:** Configuration management and orchestration.

**Why learn it:** Fast, scalable automation.

---

## Service Mesh & API Gateway

### ⭐ Istio
**What it is:** Service mesh for Kubernetes.

**Why learn it:** Advanced traffic management, security, observability.

**Key concepts:**
- Envoy sidecar proxy
- Virtual services
- Destination rules
- Gateways
- mTLS
- Traffic splitting

---

### Linkerd
**What it is:** Lightweight service mesh.

**Why learn it:** Simpler alternative to Istio.

---

### Consul
**What it is:** Service mesh and service discovery by HashiCorp.

**Why learn it:** Service discovery + mesh capabilities.

**Key concepts:**
- Service registry
- Health checks
- KV store
- Connect (service mesh)

---

### ⭐ Kong
**What it is:** API Gateway and microservices management.

**Why learn it:** Popular open-source API gateway.

**Key concepts:**
- Routes
- Services
- Plugins
- Load balancing
- Rate limiting
- Authentication

---

### Nginx Plus
**What it is:** Commercial API gateway based on Nginx.

**Why learn it:** High-performance API gateway.

---

### Traefik
**What it is:** Modern HTTP reverse proxy and load balancer.

**Why learn it:** Cloud-native, automatic service discovery.

**Key concepts:**
- EntryPoints
- Routers
- Middlewares
- Services

---

### Ambassador
**What it is:** Kubernetes-native API gateway.

**Why learn it:** Built for Kubernetes.

---

### Envoy
**What it is:** High-performance proxy.

**Why learn it:** Foundation of many service meshes.

---

## Security & Secrets Management

### ⭐ HashiCorp Vault
**What it is:** Secrets management and data protection.

**Why learn it:** Industry standard for secrets management.

**Key concepts:**
- Secret engines
- Authentication methods
- Policies
- Dynamic secrets
- Encryption as a service

---

### AWS Secrets Manager
**What it is:** AWS managed secrets service.

**Why learn it:** Native AWS integration.

---

### Azure Key Vault
**What it is:** Azure secrets management.

**Why learn it:** Native Azure integration.

---

### Google Secret Manager
**What it is:** GCP secrets management.

**Why learn it:** Native GCP integration.

---

### Sealed Secrets
**What it is:** Kubernetes secrets encryption.

**Why learn it:** GitOps-friendly encrypted secrets.

---

### cert-manager
**What it is:** Kubernetes certificate management.

**Why learn it:** Automatic TLS certificate management.

---

### Keycloak
**What it is:** Open-source identity and access management.

**Why learn it:** SSO, OAuth2, OpenID Connect.

---

### OAuth2 Proxy
**What it is:** Reverse proxy for OAuth2 authentication.

**Why learn it:** Add auth to any app.

---

## Search Engines

### ⭐ Elasticsearch
**What it is:** Distributed search and analytics engine.

**Why learn it:** Industry standard for search and log analytics.

**Key concepts:**
- Indexes
- Documents
- Shards and replicas
- Mappings
- Query DSL
- Aggregations
- ELK Stack (Elasticsearch, Logstash, Kibana)

---

### Apache Solr
**What it is:** Enterprise search platform.

**Why learn it:** Alternative to Elasticsearch.

---

### Algolia
**What it is:** Hosted search API.

**Why learn it:** Fast, easy-to-use search-as-a-service.

---

### Meilisearch
**What it is:** Fast, typo-tolerant search engine.

**Why learn it:** Lightweight, easy to deploy.

---

### Typesense
**What it is:** Fast, typo-tolerant search engine.

**Why learn it:** Alternative to Algolia (self-hosted).

---

## Load Balancing & Reverse Proxy

### ⭐ Nginx
**What it is:** High-performance web server and reverse proxy.

**Why learn it:** Most popular web server/reverse proxy.

**Key concepts:**
- Server blocks
- Location blocks
- Upstream servers
- Load balancing algorithms
- SSL/TLS termination
- Caching

---

### HAProxy
**What it is:** High-performance TCP/HTTP load balancer.

**Why learn it:** Industry-standard load balancer.

**Key concepts:**
- Frontend/Backend
- ACLs
- Health checks
- Sticky sessions

---

### Traefik
(See Service Mesh section)

---

### Envoy
(See Service Mesh section)

---

### Amazon ELB/ALB/NLB
**What it is:** AWS load balancing services.

**Why learn it:** Native AWS load balancing.

---

## Storage & File Systems

### ⭐ MinIO
**What it is:** High-performance S3-compatible object storage.

**Why learn it:** Self-hosted alternative to AWS S3.

**Key concepts:**
- Buckets
- Objects
- S3 API compatibility
- Distributed mode

---

### Ceph
**What it is:** Distributed storage system.

**Why learn it:** Unified storage (object, block, file).

---

### GlusterFS
**What it is:** Scalable network filesystem.

**Why learn it:** Distributed file system.

---

### Amazon S3
**What it is:** AWS object storage service.

**Why learn it:** Industry standard for object storage.

---

### Azure Blob Storage
**What it is:** Azure object storage.

**Why learn it:** Azure cloud storage.

---

### Google Cloud Storage
**What it is:** GCP object storage.

**Why learn it:** GCP cloud storage.

---

### NFS (Network File System)
**What it is:** Distributed file system protocol.

**Why learn it:** Classic shared storage.

---

## Logging & Log Management

### ⭐ ELK Stack (Elasticsearch, Logstash, Kibana)
**What it is:** Complete log management solution.

**Why learn it:** Industry standard for centralized logging.

**Components:**
- **Elasticsearch:** Store and search logs
- **Logstash:** Collect, parse, transform logs
- **Kibana:** Visualize logs

**Key concepts:**
- Log ingestion
- Parsing (grok patterns)
- Index management
- Dashboards
- Alerts

---

### ⭐ Grafana Loki
**What it is:** Log aggregation system inspired by Prometheus.

**Why learn it:** Lightweight, cost-effective logging.

**Key concepts:**
- Labels (not full-text indexing)
- LogQL
- Promtail (agent)
- Grafana integration

---

### Fluentd
**What it is:** Unified logging layer.

**Why learn it:** CNCF project, plugin-based architecture.

**Key concepts:**
- Input, filter, output plugins
- Buffering
- Routing

---

### Fluent Bit
**What it is:** Lightweight log processor.

**Why learn it:** Lower memory footprint than Fluentd.

---

### Splunk
**What it is:** Enterprise log management platform.

**Why learn it:** Popular in large enterprises (paid).

---

### Graylog
**What it is:** Open-source log management.

**Why learn it:** Alternative to ELK Stack.

---

### Papertrail
**What it is:** Cloud-hosted log management.

**Why learn it:** Simple, hosted logging.

---

## Tracing

### ⭐ Jaeger
**What it is:** Distributed tracing system.

**Why learn it:** CNCF project, industry standard for tracing.

**Key concepts:**
- Traces
- Spans
- Context propagation
- Sampling
- OpenTracing/OpenTelemetry

---

### Zipkin
**What it is:** Distributed tracing system.

**Why learn it:** Original open-source tracing system.

---

### OpenTelemetry
**What it is:** Observability framework (metrics, logs, traces).

**Why learn it:** Vendor-neutral observability standard.

**Key concepts:**
- Instrumentation
- Exporters
- Collectors
- SDK

---

### AWS X-Ray
**What it is:** AWS distributed tracing service.

**Why learn it:** Native AWS tracing.

---

### Grafana Tempo
**What it is:** High-scale distributed tracing backend.

**Why learn it:** Cost-effective tracing with Grafana.

---

## Service Discovery

### Consul
(See Service Mesh section)

---

### etcd
**What it is:** Distributed key-value store.

**Why learn it:** Used by Kubernetes for service discovery.

**Key concepts:**
- Raft consensus
- Watch mechanism
- Distributed configuration

---

### ZooKeeper
**What it is:** Centralized service for configuration and coordination.

**Why learn it:** Used by Kafka, Hadoop ecosystem.

---

### Eureka
**What it is:** Netflix service discovery.

**Why learn it:** Spring Cloud ecosystem.

---

## Testing Tools

### ⭐ Postman
**What it is:** API testing platform.

**Why learn it:** Most popular API testing tool.

**Key concepts:**
- Collections
- Environments
- Tests (JavaScript)
- Mock servers

---

### Insomnia
**What it is:** API client and testing tool.

**Why learn it:** Cleaner alternative to Postman.

---

### JMeter
**What it is:** Load testing tool.

**Why learn it:** Performance and load testing.

---

### Gatling
**What it is:** Load testing tool.

**Why learn it:** High-performance load testing.

---

### k6
**What it is:** Modern load testing tool.

**Why learn it:** Developer-friendly load testing.

---

### Selenium
**What it is:** Web browser automation.

**Why learn it:** E2E testing for web applications.

---

### Cypress
**What it is:** Modern E2E testing framework.

**Why learn it:** Better developer experience than Selenium.

---

### Playwright
**What it is:** Browser automation framework.

**Why learn it:** Cross-browser testing.

---

### Locust
**What it is:** Python-based load testing tool.

**Why learn it:** Write tests in Python.

---

## Cloud Platforms

### ⭐ AWS (Amazon Web Services)
**Why learn it:** Market leader (32% market share).

**Key services to know:**
- **Compute:** EC2, Lambda, ECS, EKS
- **Storage:** S3, EBS, EFS
- **Database:** RDS, DynamoDB, Aurora
- **Network:** VPC, Route 53, CloudFront
- **Messaging:** SQS, SNS, Kinesis
- **IAM:** Identity and Access Management
- **CloudWatch:** Monitoring
- **CloudFormation:** IaC

---

### ⭐ Google Cloud Platform (GCP)
**Why learn it:** Strong in data/ML (10% market share).

**Key services:**
- **Compute:** Compute Engine, Cloud Functions, GKE
- **Storage:** Cloud Storage
- **Database:** Cloud SQL, Firestore, Bigtable
- **Network:** VPC, Cloud Load Balancing
- **Messaging:** Pub/Sub
- **BigQuery:** Data warehouse

---

### ⭐ Microsoft Azure
**Why learn it:** Enterprise favorite (23% market share).

**Key services:**
- **Compute:** Virtual Machines, Azure Functions, AKS
- **Storage:** Blob Storage
- **Database:** Azure SQL, Cosmos DB
- **Network:** Virtual Network, Azure Front Door
- **Messaging:** Service Bus, Event Hubs
- **Azure Active Directory**

---

### DigitalOcean
**Why learn it:** Simple, developer-friendly cloud.

---

### Linode
**Why learn it:** Affordable VPS hosting.

---

### Heroku
**Why learn it:** Easy PaaS for rapid deployment.

---

### Vercel
**Why learn it:** Best for Next.js/frontend deployments.

---

### Netlify
**Why learn it:** JAMstack deployments.

---

## Priority Learning Path

### 🔴 **Must Learn First (Foundation)**
1. **Docker** - Containerization basics
2. **Kubernetes** - Container orchestration
3. **PostgreSQL** - Relational database
4. **Redis** - Caching and in-memory store
5. **Nginx** - Web server and reverse proxy
6. **Git/GitHub** - Version control

### 🟠 **High Priority (Core Skills)**
7. **Prometheus + Grafana** - Monitoring
8. **ELK Stack** - Logging
9. **Terraform** - Infrastructure as Code
10. **Jenkins/GitLab CI** - CI/CD
11. **RabbitMQ** - Message queue
12. **MongoDB** - NoSQL database

### 🟡 **Medium Priority (Advanced)**
13. **Kafka** - Event streaming
14. **Istio/Linkerd** - Service mesh
15. **Helm** - Kubernetes package manager
16. **Vault** - Secrets management
17. **Jaeger** - Distributed tracing
18. **Elasticsearch** - Search engine

### 🟢 **Nice to Have (Specialized)**
19. **Cassandra** - Wide-column database
20. **Consul** - Service discovery
21. **ArgoCD** - GitOps
22. **Ansible** - Configuration management

---

## Learning Resources

### Books
- "Kubernetes Up & Running" - Kelsey Hightower
- "Designing Data-Intensive Applications" - Martin Kleppmann
- "Site Reliability Engineering" - Google
- "Terraform: Up & Running" - Yevgeniy Brikman

### Platforms
- [Kubernetes.io](https://kubernetes.io/)
- [CNCF Landscape](https://landscape.cncf.io/)
- [AWS Training](https://aws.amazon.com/training/)
- [Linux Academy / A Cloud Guru](https://acloudguru.com/)
- [KodeKloud](https://kodekloud.com/)

### Practice
- [KillerCoda](https://killercoda.com/) - Interactive scenarios
- [Play with Docker](https://labs.play-with-docker.com/)
- [Play with Kubernetes](https://labs.play-with-k8s.com/)
- [Katacoda](https://www.katacoda.com/)

---

## Certifications Worth Getting

### Kubernetes
- **CKA** - Certified Kubernetes Administrator
- **CKAD** - Certified Kubernetes Application Developer
- **CKS** - Certified Kubernetes Security Specialist

### Cloud
- **AWS Solutions Architect Associate**
- **AWS DevOps Engineer Professional**
- **GCP Professional Cloud Architect**
- **Azure Administrator Associate**

### HashiCorp
- **Terraform Associate**
- **Vault Associate**

### Linux
- **RHCSA** - Red Hat Certified System Administrator
- **LFCS** - Linux Foundation Certified System Administrator

---

## Quick Reference Comparison

### Message Queues
| Service | Best For | Throughput | Persistence | Learning Curve |
|---------|----------|------------|-------------|----------------|
| RabbitMQ | Traditional queues | Medium | Yes | Easy |
| Kafka | Event streaming | Very High | Yes | Hard |
| Redis Pub/Sub | Real-time | Very High | No | Easy |
| SQS | AWS serverless | Medium | Yes | Easy |

### Databases
| Database | Type | Best For | Scalability | Learning Curve |
|----------|------|----------|-------------|----------------|
| PostgreSQL | SQL | General purpose | Vertical | Medium |
| MySQL | SQL | Web apps | Vertical | Easy |
| MongoDB | NoSQL | Document storage | Horizontal | Easy |
| Cassandra | NoSQL | Time-series, IoT | Very High | Hard |
| Redis | In-memory | Caching, sessions | High | Easy |

### Container Orchestration
| Tool | Complexity | Best For | Market Share |
|------|------------|----------|--------------|
| Kubernetes | High | Enterprise, cloud-native | 88% |
| Docker Swarm | Low | Small deployments | 5% |
| Nomad | Medium | Multi-workload | 3% |

---

**Remember:** Don't try to learn everything at once! Focus on the **Must Learn First** section, then gradually expand. 🚀

**Pro Tip:** Build projects using these services - that's the best way to learn!

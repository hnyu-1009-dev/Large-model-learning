# PostgreSQL 学习笔记（超详细中文版）

> 版本：2026-03-10  
> 定位：从 0 到能独立负责 PostgreSQL 开发与基础运维  
> 使用方式：边看边敲。每章都包含“概念 + 命令 + SQL + 易错点 + 练习建议”。

---

## 目录

1. PostgreSQL 总览与学习路线  
2. 环境准备与安装（Windows/Linux/macOS/Docker）  
3. pgAdmin 全流程使用  
4. `psql` 命令行高频操作  
5. SQL 执行顺序与语法框架  
6. 数据类型与字段设计  
7. DDL（建库建表改表删表）  
8. DML（增删改）与基础查询  
9. 多表 JOIN、子查询、CTE、递归查询  
10. 聚合、分组、窗口函数  
11. 约束、范式与建模规范  
12. 索引体系（B-Tree/GIN/GiST/BRIN）  
13. `EXPLAIN ANALYZE` 执行计划详解  
14. 事务、隔离级别、并发问题  
15. 锁机制、死锁分析与解决  
16. 视图、物化视图、函数、触发器  
17. JSONB、数组、全文检索  
18. 分区表与大表治理  
19. 用户、角色、权限与安全  
20. 备份恢复（逻辑备份、恢复演练、PITR 概念）  
21. 监控、维护与性能调优  
22. 常见报错与排障手册  
23. 项目实战：电商订单系统完整案例  
24. 学习计划（4 周 / 8 周）  
25. 练习题与参考答案思路  
26. 面试常问问题速记

---

## 1. PostgreSQL 总览与学习路线

### 1.1 PostgreSQL 是什么

PostgreSQL（简称 PG）是开源关系型数据库系统（RDBMS），核心特性：

- 完整事务支持（ACID）
- 支持高级 SQL（窗口函数、CTE、递归查询）
- 强扩展（JSONB、全文检索、地理空间扩展 PostGIS）
- 数据一致性和可靠性强

### 1.2 适用场景

- 业务主库（订单、用户、支付）
- 数据分析中台（中小到中大型）
- 需要结构化 + 半结构化混合存储的业务

### 1.3 建议学习路径

1. 先跑通环境和连接  
2. 再掌握“建表 + CRUD + JOIN + 聚合”  
3. 再学事务并发和索引优化  
4. 最后补权限、备份恢复和运维排障

---

## 2. 环境准备与安装

### 2.1 Docker（推荐学习）

```bash
# 示例目的：使用 PostgreSQL 16 镜像启动数据库容器。

# 启动 PostgreSQL 容器并映射 5432 端口
docker run --name pg16 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=demo \
  -p 5432:5432 \
  -d postgres:16
```

代码讲解：
- 使用 PostgreSQL 16 镜像启动数据库容器。
- 通过环境变量初始化密码和默认数据库 demo。


进入：

```bash
# 示例目的：进入容器内执行 psql 客户端。

# 进入容器并连接 psql 客户端
docker exec -it pg16 psql -U postgres -d demo
```

代码讲解：
- 进入容器内执行 psql 客户端。
- 适合确认数据库是否已正常启动。


### 2.2 Linux（Ubuntu/Debian）

```bash
# 示例目的：在 Ubuntu/Debian 安装 PostgreSQL 及扩展包。

# 刷新 APT 软件包索引
sudo apt update
# 安装 PostgreSQL 与常用扩展组件
sudo apt install postgresql postgresql-contrib
# 设置服务开机自启
sudo systemctl enable postgresql
# 立即启动 PostgreSQL 服务
sudo systemctl start postgresql
```

代码讲解：
- 在 Ubuntu/Debian 安装 PostgreSQL 及扩展包。
- 将服务设置为开机自启并立即启动。


### 2.3 macOS

```bash
# 示例目的：在 macOS 上通过 Homebrew 安装数据库。

# 通过 Homebrew 安装 PostgreSQL
brew install postgresql
# 启动 PostgreSQL 后台服务
brew services start postgresql
```

代码讲解：
- 在 macOS 上通过 Homebrew 安装数据库。
- 将 PostgreSQL 注册为系统后台服务。


### 2.4 Windows

- 安装官方安装包（EnterpriseDB）
- 安装时记录超级用户 `postgres` 密码
- 使用 SQL Shell (`psql`) 或 pgAdmin 连接

### 2.5 首次连接排错

- `password authentication failed`
  - 用户名/密码错误，或 `pg_hba.conf` 认证方式不匹配
- `connection refused`
  - 服务未启动、端口未监听、防火墙拦截
- `database does not exist`
  - 目标库名拼写错误

---

## 3. pgAdmin 全流程使用

### 3.1 注册服务器

1. 打开 pgAdmin  
2. `Servers` 右键 -> `Register` -> `Server`  
3. `General` 填名称：`local-pg`  
4. `Connection` 填：
   - Host: `127.0.0.1`
   - Port: `5432`
   - Maintenance database: `postgres`
   - Username: `postgres`
   - Password: 你的密码

### 3.2 常见操作入口

- 建库：`Databases` 右键 -> `Create` -> `Database`
- 建表：目标库 -> `Schemas` -> `Tables` -> `Create`
- 执行 SQL：工具栏 `Query Tool`
- 备份：数据库右键 -> `Backup...`
- 恢复：数据库右键 -> `Restore...`

### 3.3 pgAdmin 使用建议

- 结构管理用 pgAdmin，复杂 SQL 优先写在 `Query Tool`
- 生产环境操作前先在测试库演练
- 不在图形界面直接做大批量删改，先写 SQL 验证

---

## 4. `psql` 命令行高频操作

```sql
-- 示例目的：这些是 psql 元命令，用于查看对象和会话控制。

-- psql 元命令：用于查看对象或控制会话
\l                      -- 列出数据库
-- psql 元命令：用于查看对象或控制会话
\c appdb                -- 切换数据库
-- psql 元命令：用于查看对象或控制会话
\dn                     -- 列出 schema
-- psql 元命令：用于查看对象或控制会话
\dt                     -- 当前 schema 的表
-- psql 元命令：用于查看对象或控制会话
\dt biz.*               -- biz schema 下的表
-- psql 元命令：用于查看对象或控制会话
\d biz.users            -- 查看表结构
-- psql 元命令：用于查看对象或控制会话
\du                     -- 查看角色
-- psql 元命令：用于查看对象或控制会话
\x                      -- 扩展显示
-- psql 元命令：用于查看对象或控制会话
\timing                 -- 显示 SQL 耗时
-- psql 元命令：用于查看对象或控制会话
\i ./init.sql           -- 执行 SQL 文件
-- psql 元命令：用于查看对象或控制会话
\q                      -- 退出
```

代码讲解：
- 这些是 psql 元命令，用于查看对象和会话控制。
- 建议先熟悉 `\\l`、`\\dt`、`\\d`、`\\timing`。


连接命令模板：

```bash
# 示例目的：连接到指定主机、端口、用户和数据库。

# 通过命令行连接 PostgreSQL
psql -h 127.0.0.1 -p 5432 -U postgres -d appdb
```

代码讲解：
- 连接到指定主机、端口、用户和数据库。
- 这是命令行连接模板，后续所有脚本都基于它。


---

## 5. SQL 执行顺序与语法框架

很多人 SQL 写不对，是因为没理解执行顺序。

逻辑顺序（简化）：

1. `FROM` / `JOIN`
2. `WHERE`
3. `GROUP BY`
4. `HAVING`
5. `SELECT`
6. `ORDER BY`
7. `LIMIT`

典型模板：

```sql
-- 示例目的：这是标准 SQL 查询模板。

-- 查询数据
SELECT col1, agg(col2)
FROM table_a a
JOIN table_b b ON ...
WHERE ...
GROUP BY col1
HAVING ...
ORDER BY ...
LIMIT 20;
```

代码讲解：
- 这是标准 SQL 查询模板。
- 按模板能快速写出可维护的多表查询。


---

## 6. 数据类型与字段设计

### 6.1 常用类型推荐

- 主键：`BIGSERIAL` 或 `GENERATED ... AS IDENTITY`
- 金额：`NUMERIC(12,2)`（避免 `FLOAT`）
- 时间：`TIMESTAMPTZ`（带时区）
- 文本：`TEXT` / `VARCHAR(n)`
- 布尔：`BOOLEAN`
- 唯一标识：`UUID`
- 半结构化：`JSONB`

### 6.2 设计原则

- 必填字段加 `NOT NULL`
- 业务唯一值加 `UNIQUE`
- 枚举状态加 `CHECK`
- 时间字段统一 `created_at/updated_at`
- 删除策略明确：物理删 vs 逻辑删

---

## 7. DDL：建库建表改表删表

### 7.1 创建数据库和 Schema

```sql
-- 示例目的：创建数据库并切换，再创建业务 schema。

-- 创建数据库
CREATE DATABASE appdb;
-- psql 元命令：用于查看对象或控制会话
\c appdb
-- 创建 schema（命名空间）
CREATE SCHEMA biz;
```

代码讲解：
- 创建数据库并切换，再创建业务 schema。
- schema 可以把业务对象按模块隔离管理。


### 7.2 建表示例

```sql
-- 示例目的：创建用户表并设置主键、唯一约束、检查约束。

-- 创建数据表并定义字段与约束
CREATE TABLE biz.users (
  id BIGSERIAL PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  email TEXT NOT NULL UNIQUE,
  status VARCHAR(20) NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (status IN ('active', 'disabled'))
);
```

代码讲解：
- 创建用户表并设置主键、唯一约束、检查约束。
- `updated_at` 为后续审计和变更追踪做准备。


### 7.3 修改表结构

```sql
-- 示例目的：演示在线变更表结构：加列、约束、改名。

-- 修改表结构或约束
ALTER TABLE biz.users ADD COLUMN phone VARCHAR(30);
-- 修改表结构或约束
ALTER TABLE biz.users ALTER COLUMN phone SET NOT NULL;
-- 修改表结构或约束
ALTER TABLE biz.users RENAME COLUMN phone TO mobile;
```

代码讲解：
- 演示在线变更表结构：加列、约束、改名。
- 生产环境改表前建议先评估锁影响。


### 7.4 删除对象

```sql
-- 示例目的：演示删除表、schema、数据库。

-- 删除数据库对象（高风险操作）
DROP TABLE IF EXISTS biz.users;
-- 删除数据库对象（高风险操作）
DROP SCHEMA IF EXISTS biz CASCADE;
-- 删除数据库对象（高风险操作）
DROP DATABASE IF EXISTS appdb;
```

代码讲解：
- 演示删除表、schema、数据库。
- `CASCADE` 会级联删除依赖对象，谨慎操作。


注意：`CASCADE` 会级联删除依赖对象，谨慎使用。

---

## 8. DML 与基础查询

### 8.1 增删改查

```sql
-- 示例目的：一组完整 CRUD：插入、查询、更新、删除。

-- 插入新数据
INSERT INTO biz.users (username, email) VALUES ('alice', 'alice@example.com');

-- 查询数据
SELECT id, username, email
FROM biz.users
WHERE username = 'alice';

-- 更新已有数据
UPDATE biz.users
SET email = 'alice_new@example.com', updated_at = now()
WHERE id = 1;

-- 删除数据
DELETE FROM biz.users
WHERE id = 1;
```

代码讲解：
- 一组完整 CRUD：插入、查询、更新、删除。
- 实际开发建议“先查后改”，防止误更新。


### 8.2 条件、排序、分页

```sql
-- 示例目的：按时间过滤并分页返回最近数据。

-- 查询数据
SELECT *
FROM biz.users
WHERE created_at >= now() - interval '7 days'
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;
```

代码讲解：
- 按时间过滤并分页返回最近数据。
- `ORDER BY + LIMIT` 是列表页的常见组合。


警告：深分页（大 `OFFSET`）性能差。

---

## 9. JOIN、子查询、CTE、递归查询

### 9.1 业务表准备

```sql
-- 示例目的：创建商品、订单、订单明细三张核心业务表。

-- 创建数据表并定义字段与约束
CREATE TABLE biz.products (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  price NUMERIC(10,2) NOT NULL CHECK (price >= 0),
  stock INT NOT NULL CHECK (stock >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 创建数据表并定义字段与约束
CREATE TABLE biz.orders (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES biz.users(id),
  order_no TEXT NOT NULL UNIQUE,
  status VARCHAR(20) NOT NULL DEFAULT 'created',
  total_amount NUMERIC(10,2) NOT NULL DEFAULT 0 CHECK (total_amount >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (status IN ('created', 'paid', 'canceled'))
);

-- 创建数据表并定义字段与约束
CREATE TABLE biz.order_items (
  id BIGSERIAL PRIMARY KEY,
  order_id BIGINT NOT NULL REFERENCES biz.orders(id) ON DELETE CASCADE,
  product_id BIGINT NOT NULL REFERENCES biz.products(id),
  quantity INT NOT NULL CHECK (quantity > 0),
  unit_price NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0)
);
```

代码讲解：
- 创建商品、订单、订单明细三张核心业务表。
- 外键约束保证订单数据引用完整。


### 9.2 JOIN 示例

```sql
-- 示例目的：通过 INNER JOIN 关联订单和用户信息。

-- 查询数据
SELECT
  o.order_no,
  u.username,
  o.total_amount,
  o.status
FROM biz.orders o
JOIN biz.users u ON u.id = o.user_id
WHERE o.status = 'paid';
```

代码讲解：
- 通过 INNER JOIN 关联订单和用户信息。
- 用于查询已支付订单的基础报表。


### 9.3 LEFT JOIN 查“未下单用户”

```sql
-- 示例目的：LEFT JOIN + `IS NULL` 查找未匹配记录。

-- 查询数据
SELECT u.id, u.username
FROM biz.users u
LEFT JOIN biz.orders o ON o.user_id = u.id
WHERE o.id IS NULL;
```

代码讲解：
- LEFT JOIN + `IS NULL` 查找未匹配记录。
- 这是“找未下单用户”的经典写法。


### 9.4 子查询

```sql
-- 示例目的：子查询先选用户集合，再回查用户详情。

-- 查询数据
SELECT *
FROM biz.users
WHERE id IN (
-- 查询数据
  SELECT user_id
  FROM biz.orders
  WHERE status = 'paid'
);
```

代码讲解：
- 子查询先选用户集合，再回查用户详情。
- 适合把复杂条件拆成两步表达。


### 9.5 CTE

```sql
-- 示例目的：CTE 先定义中间结果，再做聚合。

-- 定义 CTE（公共表表达式）
WITH paid_orders AS (
-- 查询数据
  SELECT user_id, total_amount
  FROM biz.orders
  WHERE status = 'paid'
)
-- 查询数据
SELECT user_id, SUM(total_amount) AS paid_total
FROM paid_orders
GROUP BY user_id;
```

代码讲解：
- CTE 先定义中间结果，再做聚合。
- 能提升复杂 SQL 的可读性和可维护性。


### 9.6 递归 CTE（组织树/分类树）

```sql
-- 示例目的：递归 CTE 适合组织树/分类树遍历。

-- 定义递归 CTE 处理树形结构
WITH RECURSIVE t AS (
-- 查询数据
  SELECT id, parent_id, name, 1 AS level
  FROM biz.categories
  WHERE parent_id IS NULL
  UNION ALL
-- 查询数据
  SELECT c.id, c.parent_id, c.name, t.level + 1
  FROM biz.categories c
  JOIN t ON c.parent_id = t.id
)
-- 查询数据
SELECT * FROM t;
```

代码讲解：
- 递归 CTE 适合组织树/分类树遍历。
- 起始层通过 `parent_id IS NULL` 锚定根节点。


---

## 10. 聚合、分组、窗口函数

### 10.1 聚合统计

```sql
-- 示例目的：按天统计订单量和成交额。

-- 查询数据
SELECT
  date_trunc('day', created_at) AS day,
  COUNT(*) AS order_cnt,
  SUM(total_amount) AS total_gmv
FROM biz.orders
WHERE status = 'paid'
GROUP BY 1
ORDER BY 1;
```

代码讲解：
- 按天统计订单量和成交额。
- `date_trunc` 是时间分桶常用函数。


### 10.2 HAVING

```sql
-- 示例目的：`HAVING` 用于过滤聚合后的结果。

-- 查询数据
SELECT user_id, SUM(total_amount) AS spend
FROM biz.orders
WHERE status = 'paid'
GROUP BY user_id
HAVING SUM(total_amount) >= 1000;
```

代码讲解：
- `HAVING` 用于过滤聚合后的结果。
- 与 `WHERE` 的区别是执行阶段不同。


### 10.3 窗口函数

```sql
-- 示例目的：窗口函数可做组内排名与累计统计。

-- 查询数据
SELECT
  user_id,
  order_no,
  total_amount,
  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn,
  SUM(total_amount) OVER (PARTITION BY user_id) AS user_total
FROM biz.orders
WHERE status = 'paid';
```

代码讲解：
- 窗口函数可做组内排名与累计统计。
- 不会折叠行，适合明细+分析同时输出。


常用窗口函数：

- `ROW_NUMBER()`
- `RANK()`
- `DENSE_RANK()`
- `SUM() OVER`
- `LAG()/LEAD()`

---

## 11. 约束、范式与建模规范

### 11.1 约束清单

- 主键约束：唯一标识行
- 唯一约束：防业务重复
- 非空约束：防缺失关键数据
- 外键约束：保证引用完整性
- 检查约束：保证取值范围正确

### 11.2 三范式（简化）

1. 字段不可再分  
2. 非主键字段完全依赖主键  
3. 非主键字段不传递依赖主键

实战中建议：

- OLTP 业务表优先范式化
- 分析查询可适度冗余换性能

---

## 12. 索引体系

### 12.1 B-Tree（默认）

适用于：

- 等值查询：`=`
- 范围查询：`> < BETWEEN`
- 排序：`ORDER BY`

### 12.2 GIN

适用于：

- `JSONB` 包含查询
- 数组查询
- 全文检索

### 12.3 GiST / BRIN

- GiST：空间与范围类型
- BRIN：超大表且物理顺序相关

### 12.4 索引设计实践

```sql
-- 示例目的：为高频过滤和关联字段创建基础索引。

-- 创建索引提升查询性能
CREATE INDEX idx_orders_user_id ON biz.orders(user_id);
-- 创建索引提升查询性能
CREATE INDEX idx_orders_status_created_at ON biz.orders(status, created_at);
-- 创建索引提升查询性能
CREATE INDEX idx_order_items_order_id ON biz.order_items(order_id);
```

代码讲解：
- 为高频过滤和关联字段创建基础索引。
- 索引提升读性能，但会增加写入维护成本。


复合索引规则：

- `(a, b)` 适配 `a` 或 `(a,b)`
- 只按 `b` 过滤通常无法高效使用该索引

### 12.5 高级索引

- 部分索引：

```sql
-- 示例目的：部分索引只覆盖 `status = paid` 的数据。

-- 创建索引提升查询性能
CREATE INDEX idx_orders_paid_created_at
ON biz.orders(created_at)
WHERE status = 'paid';
```

代码讲解：
- 部分索引只覆盖 `status = paid` 的数据。
- 可减少索引体积并提升特定查询性能。


- 表达式索引：

```sql
-- 示例目的：表达式索引适用于函数查询场景。

-- 创建索引提升查询性能
CREATE INDEX idx_users_lower_email ON biz.users ((lower(email)));
```

代码讲解：
- 表达式索引适用于函数查询场景。
- 例如按 `lower(email)` 做大小写不敏感查询。


- 在线建索引（减少锁表影响）：

```sql
-- 示例目的：并发创建索引，降低对业务写入的阻塞。

-- 创建索引提升查询性能
CREATE INDEX CONCURRENTLY idx_orders_order_no ON biz.orders(order_no);
```

代码讲解：
- 并发创建索引，降低对业务写入的阻塞。
- 适合在线环境补索引。


---

## 13. 执行计划详解（`EXPLAIN ANALYZE`）

### 13.1 基础用法

```sql
-- 示例目的：用 `EXPLAIN ANALYZE` 查看真实执行计划。

-- 输出查询执行计划与真实耗时
EXPLAIN ANALYZE
-- 查询数据
SELECT *
FROM biz.orders
WHERE status = 'paid'
  AND created_at >= now() - interval '30 days'
ORDER BY created_at DESC
LIMIT 20;
```

代码讲解：
- 用 `EXPLAIN ANALYZE` 查看真实执行计划。
- 重点关注扫描方式、行数估算和耗时。


### 13.2 重点字段

- `cost`: 优化器估算成本
- `rows`: 估算/实际行数
- `actual time`: 实际耗时
- `loops`: 节点循环次数

### 13.3 常见节点

- `Seq Scan`：顺序扫描（可能慢）
- `Index Scan`：索引扫描
- `Bitmap Index Scan` + `Bitmap Heap Scan`
- `Hash Join` / `Merge Join` / `Nested Loop`
- `Sort` / `Aggregate`

### 13.4 优化思路

1. 看过滤条件是否命中索引  
2. 看返回列是否过多（减少 `SELECT *`）  
3. 看连接顺序与行数估算误差  
4. 必要时更新统计信息：`ANALYZE`

---

## 14. 事务与隔离级别

### 14.1 事务四大特性（ACID）

- A 原子性
- C 一致性
- I 隔离性
- D 持久性

### 14.2 事务语法

```sql
-- 示例目的：事务中同时修改库存和订单状态。

-- 开启事务
BEGIN;
-- 更新已有数据
UPDATE biz.products SET stock = stock - 1 WHERE id = 1 AND stock > 0;
-- 更新已有数据
UPDATE biz.orders SET status = 'paid' WHERE id = 1001;
-- 提交事务
COMMIT;
-- 出错用 ROLLBACK;
```

代码讲解：
- 事务中同时修改库存和订单状态。
- 通过 `COMMIT/ROLLBACK` 保证原子一致。


### 14.3 隔离级别

```sql
-- 示例目的：查看当前会话隔离级别。

-- 查看当前配置或会话参数
SHOW transaction_isolation;
```

代码讲解：
- 查看当前会话隔离级别。
- 可用于定位并发读写行为差异。


- `READ COMMITTED`（默认）
- `REPEATABLE READ`
- `SERIALIZABLE`

### 14.4 并发现象

- 脏读（PG 默认不会）
- 不可重复读
- 幻读（在高隔离级别可防）

---

## 15. 锁机制与死锁

### 15.1 常见锁

- 表锁（DDL 时常见）
- 行锁（`FOR UPDATE`）
- 共享锁、排他锁（内部机制）

### 15.2 行锁防超卖

```sql
-- 示例目的：用 `FOR UPDATE` 锁住库存行。

-- 开启事务
BEGIN;
-- 查询数据
SELECT stock FROM biz.products WHERE id = 1 FOR UPDATE;
-- 更新已有数据
UPDATE biz.products SET stock = stock - 1 WHERE id = 1 AND stock > 0;
-- 提交事务
COMMIT;
```

代码讲解：
- 用 `FOR UPDATE` 锁住库存行。
- 防止并发扣减导致超卖。


### 15.3 死锁成因

事务 A：先锁行 1 后锁行 2  
事务 B：先锁行 2 后锁行 1  
彼此等待 -> 死锁。

### 15.4 死锁治理

- 统一加锁顺序
- 缩短事务时间
- 应用层捕获死锁异常并重试

---

## 16. 视图、物化视图、函数、触发器

### 16.1 视图

```sql
-- 示例目的：创建只读视图封装常用查询。

-- 创建视图封装查询逻辑
CREATE VIEW biz.v_paid_orders AS
-- 查询数据
SELECT o.order_no, u.username, o.total_amount, o.created_at
FROM biz.orders o
JOIN biz.users u ON u.id = o.user_id
WHERE o.status = 'paid';
```

代码讲解：
- 创建只读视图封装常用查询。
- 对上层应用暴露稳定查询接口。


### 16.2 物化视图

```sql
-- 示例目的：创建物化视图缓存聚合结果。

-- 创建物化视图缓存计算结果
CREATE MATERIALIZED VIEW biz.mv_daily_gmv AS
-- 查询数据
SELECT date_trunc('day', created_at) AS day, SUM(total_amount) AS gmv
FROM biz.orders
WHERE status = 'paid'
GROUP BY 1;

-- 刷新物化视图数据
REFRESH MATERIALIZED VIEW biz.mv_daily_gmv;
```

代码讲解：
- 创建物化视图缓存聚合结果。
- 通过 `REFRESH` 手动刷新数据。


### 16.3 函数（PL/pgSQL）

```sql
-- 示例目的：定义一个简单 PL/pgSQL 函数。

-- 定义或更新数据库函数
CREATE OR REPLACE FUNCTION biz.add_one(i int)
RETURNS int
LANGUAGE plpgsql
AS $$
-- 开启事务
BEGIN
  RETURN i + 1;
END;
$$;
```

代码讲解：
- 定义一个简单 PL/pgSQL 函数。
- 用于理解函数签名、返回值和函数体结构。


### 16.4 触发器（自动更新时间）

```sql
-- 示例目的：触发器函数在更新前自动写入 `updated_at`。

-- 定义或更新数据库函数
CREATE OR REPLACE FUNCTION biz.set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
-- 开启事务
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

-- 创建触发器绑定表事件
CREATE TRIGGER trg_users_updated_at
BEFORE UPDATE ON biz.users
FOR EACH ROW
EXECUTE FUNCTION biz.set_updated_at();
```

代码讲解：
- 触发器函数在更新前自动写入 `updated_at`。
- 减少应用层重复更新时间字段的代码。


---

## 17. JSONB、数组、全文检索

### 17.1 JSONB 基础

```sql
-- 示例目的：创建事件表并写入 JSONB 数据。

-- 创建数据表并定义字段与约束
CREATE TABLE biz.events (
  id BIGSERIAL PRIMARY KEY,
  event_type TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 插入新数据
INSERT INTO biz.events (event_type, payload) VALUES
('order_created', '{"order_no":"ORD1001","channel":"app","amount":399}'),
('order_paid', '{"order_no":"ORD1001","channel":"app","pay_method":"wechat"}');
```

代码讲解：
- 创建事件表并写入 JSONB 数据。
- 适合埋点、日志、扩展属性存储。


查询：

```sql
-- 示例目的：从 JSONB 中提取字段并按包含条件过滤。

-- 查询数据
SELECT payload->>'order_no' AS order_no
FROM biz.events
WHERE payload @> '{"channel":"app"}';
```

代码讲解：
- 从 JSONB 中提取字段并按包含条件过滤。
- `->>` 返回文本，`@>` 做包含匹配。


索引：

```sql
-- 示例目的：为 JSONB 建立 GIN 索引。

-- 创建索引提升查询性能
CREATE INDEX idx_events_payload_gin ON biz.events USING GIN (payload);
```

代码讲解：
- 为 JSONB 建立 GIN 索引。
- 可显著加速包含类查询。


### 17.2 数组示例

```sql
-- 示例目的：演示数组字段建模与包含查询。

-- 创建数据表并定义字段与约束
CREATE TABLE biz.tags_demo (
  id BIGSERIAL PRIMARY KEY,
  tags TEXT[]
);

-- 查询数据
SELECT * FROM biz.tags_demo WHERE tags @> ARRAY['pg'];
```

代码讲解：
- 演示数组字段建模与包含查询。
- 数组适合中小规模标签集合。


### 17.3 全文检索（基础）

```sql
-- 示例目的：展示全文检索的向量和查询表达式。

-- 查询数据
SELECT to_tsvector('simple', 'postgresql full text search');
-- 查询数据
SELECT to_tsquery('simple', 'postgresql & search');
```

代码讲解：
- 展示全文检索的向量和查询表达式。
- 后续可结合索引实现高性能搜索。


---

## 18. 分区表与大表治理

### 18.1 何时考虑分区

- 单表数据量非常大（千万级以上）
- 按时间、租户等维度查询明显
- 需要冷热数据分层管理

### 18.2 范围分区示例

```sql
-- 示例目的：创建按时间范围分区的订单表。

-- 创建数据表并定义字段与约束
CREATE TABLE biz.orders_p (
  id BIGSERIAL,
  user_id BIGINT NOT NULL,
  order_no TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 创建数据表并定义字段与约束
CREATE TABLE biz.orders_p_2026_03 PARTITION OF biz.orders_p
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
```

代码讲解：
- 创建按时间范围分区的订单表。
- 分区有助于大表查询与归档治理。


---

## 19. 用户、角色、权限、安全

### 19.1 角色模型建议

- `app_rw`: 应用读写
- `app_ro`: 报表只读
- `dba`: 管理员

### 19.2 授权示例

```sql
-- 示例目的：创建只读角色并授予库、schema、表权限。

-- 创建数据库角色（用户）
CREATE ROLE app_ro LOGIN PASSWORD 'replace_me';
-- 授予权限
GRANT CONNECT ON DATABASE appdb TO app_ro;
-- 授予权限
GRANT USAGE ON SCHEMA biz TO app_ro;
-- 授予权限
GRANT SELECT ON ALL TABLES IN SCHEMA biz TO app_ro;

-- 设置默认权限继承规则
ALTER DEFAULT PRIVILEGES IN SCHEMA biz
-- 授予权限
GRANT SELECT ON TABLES TO app_ro;
```

代码讲解：
- 创建只读角色并授予库、schema、表权限。
- 默认权限确保新表自动继承只读授权。


### 19.3 安全建议

- 应用账号禁止超级用户
- 生产环境强密码和定期轮换
- 限制公网暴露，优先内网访问
- 审计关键操作（DDL、权限变更）

---

## 20. 备份恢复

PostgreSQL 备份大体分为三类：

- 逻辑备份（`pg_dump` / `pg_dumpall`）
- 物理备份（`pg_basebackup` 等）
- 持续归档 + 时间点恢复（PITR）

日常开发和教学中，最常用的是逻辑备份；生产环境通常还需要结合 WAL 归档实现 PITR。

### 20.1 逻辑备份（Custom 格式）

逻辑备份通过 `pg_dump` 导出数据库对象和数据。常见格式：

- plain SQL：文本文件，可直接查看编辑，用 `psql` 恢复
- archive：如 custom、directory，支持选择性恢复、并行恢复，用 `pg_restore` 恢复

下面示例使用 custom 格式导出单个数据库：

```bash
# 示例目的：使用自定义格式导出逻辑备份，便于选择性恢复和并行恢复。

# 执行逻辑备份导出
pg_dump -h 127.0.0.1 -U postgres -d appdb -F c -f appdb_2026-03-10.dump
```

代码讲解：
- `-h 127.0.0.1`：指定数据库主机。
- `-U postgres`：指定连接用户。
- `-d appdb`：指定要备份的数据库。
- `-F c`：导出为 custom 归档格式。
- `-f ...dump`：指定输出文件路径和文件名。

适用场景：

- 备份单个数据库
- 迁移测试环境
- 按表、按 schema、按对象恢复
- 并行恢复提升恢复效率

补充说明：

- `pg_dump` 只备份单个数据库，不包含实例里的所有数据库。
- `pg_dump` 不会自动包含 cluster 级全局对象（角色、表空间等）。

### 20.2 恢复（Custom 格式）

`pg_restore` 用于恢复 `pg_dump` 生成的非纯文本备份（custom、directory 等），支持选择性恢复和并行恢复。

```bash
# 示例目的：先创建目标库，再执行 pg_restore 恢复。

# 创建目标数据库用于恢复
createdb -h 127.0.0.1 -U postgres appdb_restore
# 将逻辑备份恢复到目标数据库
pg_restore -h 127.0.0.1 -U postgres -d appdb_restore appdb_2026-03-10.dump
```

代码讲解：
- 第一条命令创建空数据库，作为恢复目标。
- 第二条命令将归档备份恢复到目标库。

常用补充命令：

```bash
# 示例目的：查看备份对象、覆盖恢复和并行恢复。

# 查看归档备份中包含哪些对象
pg_restore -l appdb_2026-03-10.dump
# 恢复前先清理目标库中的已有对象
pg_restore -c -h 127.0.0.1 -U postgres -d appdb_restore appdb_2026-03-10.dump
# 并行恢复（适合较大库）
pg_restore -j 4 -h 127.0.0.1 -U postgres -d appdb_restore appdb_2026-03-10.dump
```

代码讲解：
- `-l`：列出归档备份里的对象清单。
- `-c`：恢复前先清理目标库已有对象。
- `-j`：并行恢复（仅 archive 格式有效，不适用于纯 SQL 文本）。

注意事项：

- 恢复前确认目标库是否允许覆盖。
- 如果备份依赖角色、扩展、表空间，目标环境需提前准备。

### 20.3 SQL 文本备份

如果希望备份文件可直接打开审阅、做版本对比，可以导出为 SQL 文本格式：

```bash
# 示例目的：导出 SQL 文本并通过 psql 回放。

# 导出 SQL 文本备份
pg_dump -h 127.0.0.1 -U postgres -d appdb -f appdb.sql
# 将 SQL 脚本回放到目标数据库
psql -h 127.0.0.1 -U postgres -d appdb_restore -f appdb.sql
```

代码讲解：
- 第一条命令将数据库导出为 SQL 脚本。
- 第二条命令把脚本回放到目标数据库。

优点：

- 文件可读，便于学习和审查
- 适合代码评审、结构比对
- 恢复方式直观

缺点：

- 不支持 `pg_restore` 的选择性恢复能力
- 大库恢复通常不如 archive 格式灵活
- 不支持 `pg_restore -j` 并行恢复

### 20.4 整个实例备份与 `pg_dumpall`

当需要备份整个实例（所有数据库 + 全局对象）时，使用 `pg_dumpall`。

```bash
# 示例目的：导出整个 PostgreSQL 实例中的所有数据库和全局对象。

# 导出实例级 SQL 备份
pg_dumpall -h 127.0.0.1 -U postgres -f all_databases.sql
# 恢复实例级 SQL 备份
psql -h 127.0.0.1 -U postgres -f all_databases.sql
```

代码讲解：
- `pg_dumpall` 会导出整个 cluster，包含角色、表空间等全局对象。
- 恢复本质是回放一份实例级 SQL 脚本。

适用场景：

- 整机迁移
- 实例级全量备份
- 大版本升级前逻辑导出

### 20.5 PITR（时间点恢复）概念

PITR（Point-in-Time Recovery）核心思想：

`基础备份（Base Backup） + 持续归档 WAL = 恢复到任意目标时刻`

典型用途：

- 恢复到误删前几分钟
- 恢复到错误脚本执行前
- 构建高可用恢复能力

生产常见策略：

- 每天或每周做一次基础备份
- 持续归档 WAL
- 定期做恢复演练验证可用性

PITR 的组成：

- 基础备份：通常由 `pg_basebackup` 生成（针对整个 cluster）
- WAL 归档：保存备份后的 WAL 文件
- 恢复目标：按时间点、事务 ID 或恢复标记

### 20.6 文件系统级备份补充

除了逻辑备份，PostgreSQL 也支持物理基础备份。需要注意：

- 不能简单复制单表文件来恢复单表或单库
- 物理文件依赖 WAL 和内部一致性状态
- 物理备份应使用 `pg_basebackup` 或等效一致性方案

可这样理解：

- 选择性备份：`pg_dump`
- 整个 cluster 物理备份：`pg_basebackup`
- 时间点恢复：基础备份 + WAL 归档

### 20.7 备份验证建议

备份文件生成成功，不等于备份真正可用。更稳妥的做法是定期恢复演练。

建议至少验证以下内容：

- 备份文件是否成功生成
- 备份文件是否可读取
- 是否能恢复到测试库
- 恢复后的表、索引、约束、数据量是否正确
- 应用是否能正常连接恢复后的数据库

## 21. 监控、维护、调优

### 21.1 关键系统视图

```sql
-- 示例目的：查询活动会话、锁和表统计信息。

-- 查询数据
SELECT * FROM pg_stat_activity;
-- 查询数据
SELECT * FROM pg_locks;
-- 查询数据
SELECT * FROM pg_stat_user_tables;
```

代码讲解：
- 查询活动会话、锁和表统计信息。
- 这是排障慢查询和锁等待的入口。


### 21.2 维护命令

```sql
-- 示例目的：更新统计信息、回收空间、重建索引。

-- 更新统计信息，帮助优化器估算
ANALYZE;
-- 回收死元组并维护存储
VACUUM;
-- 回收死元组并维护存储
VACUUM ANALYZE biz.orders;
-- 重建索引，修复膨胀或损坏风险
REINDEX TABLE biz.orders;
```

代码讲解：
- 更新统计信息、回收空间、重建索引。
- 定期维护可避免性能衰退。


### 21.3 常见参数（入门理解）

- `shared_buffers`
- `work_mem`
- `maintenance_work_mem`
- `max_connections`
- `effective_cache_size`

调优原则：

- 先定位瓶颈（SQL/IO/CPU/锁）
- 再改参数，不要盲目调

---

## 22. 常见报错与排障手册

`duplicate key value violates unique constraint`

- 原因：唯一键冲突
- 处理：检查业务唯一字段；必要时 `INSERT ... ON CONFLICT`

`deadlock detected`

- 原因：锁顺序不一致
- 处理：统一顺序 + 重试机制

`canceling statement due to lock timeout`

- 原因：长事务持锁导致阻塞
- 处理：定位阻塞会话并优化事务

`permission denied for table ...`

- 原因：缺权限
- 处理：`GRANT SELECT/INSERT/UPDATE/DELETE ...`

查询突然变慢：

1. 看执行计划  
2. 看统计信息是否过期  
3. 看索引是否失效  
4. 看锁等待和资源使用

---

## 23. 项目实战：电商订单系统

### 23.1 业务目标

- 用户下单
- 订单明细
- 支付更新状态
- 库存扣减防超卖
- 日报统计

### 23.2 初始化数据（示例）

```sql
-- 示例目的：批量插入用户和商品测试数据。

-- 插入新数据
INSERT INTO biz.users (username, email) VALUES
('alice', 'alice@example.com'),
('bob', 'bob@example.com'),
('carol', 'carol@example.com');

-- 插入新数据
INSERT INTO biz.products (name, price, stock) VALUES
('机械键盘', 399.00, 100),
('电竞鼠标', 199.00, 200),
('显示器', 1299.00, 50);
```

代码讲解：
- 批量插入用户和商品测试数据。
- 用于后续报表和性能练习。


### 23.3 下单事务（模板）

```sql
-- 示例目的：完整事务模板：锁库存、建订单、写明细、扣库存、改状态。

-- 开启事务
BEGIN;

-- 1) 锁库存行
-- 查询数据
SELECT stock FROM biz.products WHERE id = 1 FOR UPDATE;

-- 2) 创建订单
-- 插入新数据
INSERT INTO biz.orders (user_id, order_no, status, total_amount)
VALUES (1, 'ORD202603100001', 'created', 399.00);

-- 3) 写入订单明细（假设订单 id = currval）
-- 插入新数据
INSERT INTO biz.order_items (order_id, product_id, quantity, unit_price)
VALUES (currval('biz.orders_id_seq'), 1, 1, 399.00);

-- 4) 扣减库存
-- 更新已有数据
UPDATE biz.products
SET stock = stock - 1
WHERE id = 1 AND stock > 0;

-- 5) 标记支付成功（示例）
-- 更新已有数据
UPDATE biz.orders
SET status = 'paid'
WHERE order_no = 'ORD202603100001';

-- 提交事务
COMMIT;
```

代码讲解：
- 完整事务模板：锁库存、建订单、写明细、扣库存、改状态。
- 可直接作为业务代码的数据库事务蓝本。


### 23.4 报表 SQL 示例

近 30 天用户消费排行：

```sql
-- 示例目的：统计近 30 天用户消费排行榜。

-- 查询数据
SELECT
  u.username,
  SUM(o.total_amount) AS total_spend
FROM biz.orders o
JOIN biz.users u ON u.id = o.user_id
WHERE o.status = 'paid'
  AND o.created_at >= now() - interval '30 days'
GROUP BY u.username
ORDER BY total_spend DESC
LIMIT 10;
```

代码讲解：
- 统计近 30 天用户消费排行榜。
- 典型运营报表 SQL。


---

## 24. 学习计划（4 周 / 8 周）

### 24.1 4 周快训

第 1 周：

- 跑通安装、连接、建库建表
- 熟练 CRUD、过滤、排序、分页

第 2 周：

- 掌握 JOIN、聚合、CTE、窗口函数
- 完成 4 张关联表建模

第 3 周：

- 学执行计划与索引设计
- 完成 2 条慢查询优化

第 4 周：

- 事务并发、锁死锁
- 权限配置与备份恢复演练

### 24.2 8 周进阶

- 第 5~6 周：JSONB、函数触发器、物化视图
- 第 7 周：分区、大表治理、监控指标
- 第 8 周：完成一份项目复盘（设计、优化、故障处理）

---

## 25. 练习题与答案思路

### 25.1 练习题

1. 设计 `users/products/orders/order_items` 四表并加完整约束  
2. 查“近 7 天每天订单数与 GMV”  
3. 查“从未下单用户”  
4. 查“每个用户最近 1 笔订单”  
5. 针对“按状态+时间查订单”设计索引并验证  
6. 写一个事务实现下单扣库存并发安全  
7. 创建只读角色并验证无法写入  
8. 备份恢复到新库并核对数据一致

### 25.2 答案思路

- 题 2 用 `date_trunc + group by`
- 题 3 用 `left join ... is null`
- 题 4 用 `row_number() over(partition by user_id order by created_at desc)`
- 题 5 用复合索引 `(status, created_at)` 并 `EXPLAIN ANALYZE`
- 题 6 用 `FOR UPDATE` 与事务控制

---

## 26. 面试常问问题速记

1. PG 和 MySQL 有什么区别？  
2. 事务隔离级别有哪些？默认是什么？  
3. 索引失效常见原因？  
4. `EXPLAIN ANALYZE` 怎么看？  
5. 什么是 MVCC？  
6. 如何避免死锁？  
7. 备份恢复怎么做？  
8. JSONB 为什么快？  
9. 分区适合什么场景？  
10. 如何做权限最小化？

简短参考：

- 默认隔离级别：`READ COMMITTED`
- 索引失效：函数包裹字段、类型不匹配、低选择性等
- 死锁避免：统一锁顺序 + 缩短事务 + 重试

---

## 附录 A：高频 SQL 速查

```sql
-- 示例目的：汇总高频命令用于快速排障。

-- 看库/表
-- psql 元命令：用于查看对象或控制会话
\l
-- psql 元命令：用于查看对象或控制会话
\dt biz.*

-- 看执行计划
-- 输出查询执行计划与真实耗时
EXPLAIN ANALYZE SELECT ...;

-- 事务
-- 开启事务
BEGIN;
...;
-- 提交事务
COMMIT;

-- 统计信息
ANALYZE biz.orders;

-- 创建索引
-- 创建索引提升查询性能
CREATE INDEX idx_xxx ON biz.orders(status, created_at);
```

代码讲解：
- 汇总高频命令用于快速排障。
- 建议按“查看 -> 诊断 -> 维护”顺序使用。


---

## 附录 B：建议收藏资料

- PostgreSQL 官方文档：https://www.postgresql.org/docs/  
- PostgreSQL 中文社区：https://postgresql.org.cn/  
- 菜鸟教程 PostgreSQL 目录：https://www.runoob.com/postgresql/postgresql-tutorial.html  
- 菜鸟教程 pgAdmin 页面：https://www.runoob.com/postgresql/postgresql-pgadmin.html

---

## 结束语

判断“学会 PostgreSQL”的标准不是记住语法，而是能独立完成以下闭环：

1. 设计表结构并保证数据正确  
2. 写出可维护、可优化的 SQL  
3. 处理事务并发与锁问题  
4. 做权限隔离和备份恢复  
5. 在慢查询和故障时有排障路径

把这份笔记完整实操一遍，你就能达到“可独立承担中小项目数据库工作”的水平。


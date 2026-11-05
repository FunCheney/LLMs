-- pgvector 初始化：只在数据卷为空时由 postgres 镜像执行一次。
--
-- 为什么不交给应用层的 ensure_schema：`CREATE EXTENSION` 需要超级用户权限，
-- 而应用连库用的是普通用户。装扩展必须发生在容器首次启动、以 postgres 身份
-- 执行的时候，否则运行时会撞上"permission denied to create extension"——
-- 这个错误跟向量检索看起来毫无关系，很容易被误判成代码问题。
--
-- 维度 1024 对应 bge-m3。换 embedding 模型必须重建表（`ALTER COLUMN ... TYPE`
-- 在有数据时会失败），这是 pgvector 的硬约束，不是可以绕过的细节。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id       text PRIMARY KEY,
    document_id    text NOT NULL,
    text           text NOT NULL,
    metadata       jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding      vector(1024),
    -- content_hash：W6 增量索引靠它判断"这段文本变没变"，变了才重算向量
    content_hash   text,
    -- model_snapshot：记录向量由哪个模型产生，换模型后能筛出需要重算的行
    model_snapshot text,
    updated_at     timestamptz NOT NULL DEFAULT now()
);

-- 按文档删除/统计（增量索引的 delete 传播走这条索引）
CREATE INDEX IF NOT EXISTS rag_chunks_doc_idx ON rag_chunks (document_id);

-- HNSW 比 IVFFlat 更适合这里的访问模式：不需要预先聚类训练，
-- 增量写入后不用重建，代价是建索引慢、内存占用高（万级 chunk 上可忽略）。
-- vector_cosine_ops 必须与查询侧的 `<=>` 一致，否则索引不会被使用，
-- 表现为"加了索引还是全表扫"，但 EXPLAIN 之外看不出来。
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);

-- 治理字段（tenant / doc_type / valid_until）的粗过滤走 jsonb 包含查询
CREATE INDEX IF NOT EXISTS rag_chunks_metadata_idx ON rag_chunks USING gin (metadata);

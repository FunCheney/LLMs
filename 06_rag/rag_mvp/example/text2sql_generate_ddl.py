from pathlib import Path

import yaml
import pymysql

db_name = "sakila"

conn = pymysql.connect(
    host="127.0.0.1",
    port=9999,
    user="root",
    password="123456",
    database="mysql",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.Cursor,
)


ddl_map = {}
try:
    with conn.cursor() as cursor:
        # 获取所有表名
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s;", (db_name,)
        )
        tables = [row[0] for row in cursor.fetchall()]

        # 遍历表列表，执行 SHOW CREATE TABLE
        for tbl in tables:
            cursor.execute(f"SHOW CREATE TABLE `{db_name}`.`{tbl}`;")
            result = cursor.fetchone()
            # result[0]=表名, result[1]=完整 DDL
            ddl_map[tbl] = result[1]

finally:
    conn.close()

# 5. 写入 YAML 文件
with open(Path("data/sakila/ddl_statements.yaml"), "w") as f:
    yaml.safe_dump(ddl_map, f, sort_keys=True, allow_unicode=True)
print("✅ ddl_statements.yaml 已生成，共包含表：", list(ddl_map.keys()))
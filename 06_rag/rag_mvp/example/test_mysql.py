import pymysql
"""
# 启动服务
docker run -d \
  --name mysql-local \
  -p 9999:3306 \
  -e MYSQL_ROOT_PASSWORD=123456 \
  -v mysql-data:/var/lib/mysql \
  mysql:8.0.45
# 本地连接
mysql -h 127.0.0.1 -P 9999 -uroot -p
# docker stop mysql:8.0.45
# docker start mysql:8.0.45
"""
conn = pymysql.connect(
    host="127.0.0.1",
    port=9999,
    user="root",
    password="123456",
    database="mysql",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
)

with conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT VERSION() AS version")
        row = cursor.fetchone()
        print(f"MySQL 版本：{row['version']}")

print("Python 已成功连接 Docker 中的 MySQL")

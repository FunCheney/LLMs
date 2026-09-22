# import pymysql
#
# conn = pymysql.connect(
#     host="127.0.0.1",
#     port=9999,
#     user="root",
#     password="123456",
#     database="mysql",
#     charset="utf8mb4",
#     cursorclass=pymysql.cursors.DictCursor,
# )
#
# with conn:
#     with conn.cursor() as cursor:
#         cursor.execute("SELECT VERSION() AS version")
#         row = cursor.fetchone()
#         print(f"MySQL 版本：{row['version']}")
#
# print("Python 已成功连接 Docker 中的 MySQL")

from huggingface_hub import scan_cache_dir

# 1. 扫描缓存并获取报告
cache_info = scan_cache_dir()

# 2. 现在你可以使用正确的属性名了
print(f"可释放空间: {cache_info.size_on_disk}")

# 3. 查看所有已缓存的模型列表，确认要删除的目标
for repo in cache_info.repos:
    print(f"- {repo.repo_id}")

# 4. 执行删除操作 (示例：删除一个指定模型)
# to_delete = []
# for repo in cache_info.repos:
#     if "BAAI/bge-small-zh-v1.5" in repo.repo_id:
#         for revision in repo.revisions:
#             to_delete.extend(revision.snapshots)
# if to_delete:
#     delete_strategy = cache_info.delete_revisions(*to_delete)
#     print(f"计划释放空间: {delete_strategy.expected_freed_size_str}")
#     # delete_strategy.execute()  # 确认无误后去掉注释执行
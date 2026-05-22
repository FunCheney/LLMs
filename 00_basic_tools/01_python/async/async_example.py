import asyncio
import time

# 同步版本：请求会一个一个排队执行
def sync_fetch(url):
    print(f"同步请求开始: {url}")
    time.sleep(2)  # 模拟耗时2秒的网络请求
    print(f"同步请求结束: {url}")
    return url

# 异步版本：请求会并发执行
async def async_fetch(url):
    print(f"异步请求开始: {url}")
    await asyncio.sleep(2)  # 模拟异步的网络请求
    print(f"异步请求结束: {url}")
    return url

async def main():
    urls = ['url1', 'url2', 'url3']

    # 同步执行
    print("同步执行开始")
    start = time.time()
    for url in urls:
        sync_fetch(url)
    print(f"同步执行耗时: {time.time() - start:.2f}秒\n")

    # 异步执行
    print("异步执行开始")
    start = time.time()
    # 创建三个任务，让事件循环并发调度
    tasks = [async_fetch(url) for url in urls]
    await asyncio.gather(*tasks)  # 等待所有任务完成
    print(f"异步执行耗时: {time.time() - start:.2f}秒")

# 运行异步主函数
asyncio.run(main())
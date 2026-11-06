import asyncio

lock = asyncio.Lock()
shared_data = 0


async def safe_increment():
    global shared_data
    async with lock: #获取锁
        # 下面这行代码在同一时刻只会被一个协程执行
        current_data = shared_data
        await asyncio.sleep(1) # 模拟一些异步操作
        shared_data = current_data + 1

    # 离开 with 块，锁自动释放

async def main():
    await asyncio.gather(*[safe_increment() for _ in range(5)])
    print(shared_data)

asyncio.run(main())

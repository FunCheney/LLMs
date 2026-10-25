import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from agent_framework.activities import greet
from agent_framework.config import TASK_QUEUE, TEMPORAL_ADDRESS
from agent_framework.workflows import SayHelloWorkflow


async def main() -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SayHelloWorkflow],
        activities=[greet],
    )
    await worker.run()


def run_worker() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run_worker()

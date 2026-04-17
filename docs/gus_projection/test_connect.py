import asyncio
from viam_client import connect_robot

async def main():
    robot = await connect_robot()
    print("Connected to Viam robot successfully")
    await robot.close()

if __name__ == "__main__":
    asyncio.run(main())
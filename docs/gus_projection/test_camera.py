import asyncio
import cv2
from viam_client import connect_robot, get_camera_handle
from utils import viam_image_to_cv2

async def main():
    robot = await connect_robot()
    camera = await get_camera_handle(robot)

    try:
        while True:
            images, _ = await camera.get_images()
            if not images:
                print("No image received")
                await asyncio.sleep(0.1)
                continue

            frame = viam_image_to_cv2(images[0])
            cv2.imshow("Viam Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.05)
    finally:
        cv2.destroyAllWindows()
        await robot.close()

if __name__ == "__main__":
    asyncio.run(main())
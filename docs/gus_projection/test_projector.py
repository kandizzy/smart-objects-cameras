import asyncio
import cv2

from viam_client import connect_robot, get_camera_handle
from utils import viam_image_to_cv2
from projector import setup_window, show_frame

async def main():
    print("1. Connecting to robot...")
    robot = await connect_robot()
    print("2. Connected.")

    print("3. Getting camera handle...")
    camera = await get_camera_handle(robot)
    print("4. Camera handle ready.")

    setup_window()
    print("5. Window opened.")

    try:
        while True:
            print("6. Requesting image...")
            images, _ = await camera.get_images()

            if not images:
                print("No image received.")
                await asyncio.sleep(0.1)
                continue

            print("7. Image received.")
            frame = viam_image_to_cv2(images[0])
            print("8. Converted to cv2 frame.")

            show_frame(frame)
            print("9. Frame displayed.")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("10. Quit pressed.")
                break

            await asyncio.sleep(0.05)

    finally:
        print("11. Cleaning up...")
        cv2.destroyAllWindows()
        await robot.close()
        print("12. Done.")

if __name__ == "__main__":
    asyncio.run(main())
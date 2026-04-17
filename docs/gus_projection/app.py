import asyncio
import cv2

from viam_client import connect_robot, get_handles
from detector import pick_target_detections
from projector import setup_fullscreen, show_frame, show_blank
from utils import viam_image_to_cv2, draw_boxes
from config import CAMERA_NAME, SHOW_BOXES

async def main():
    robot = await connect_robot()
    camera, detector = await get_handles(robot)

    setup_fullscreen()

    try:
        while True:
            # 1) get next detections from camera via Viam vision service
            detections = await detector.get_detections_from_camera(CAMERA_NAME)

            # 2) get next image frame from Viam camera
            images, _ = await camera.get_images()
            if not images:
                show_blank()
                await asyncio.sleep(0.1)
                continue

            frame = viam_image_to_cv2(images[0])

            gus_detections = pick_target_detections(detections)

            if gus_detections:
                if SHOW_BOXES:
                    frame = draw_boxes(frame, gus_detections)
                show_frame(frame)
            else:
                show_blank(frame.shape[1], frame.shape[0])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.05)

    finally:
        cv2.destroyAllWindows()
        await robot.close()

if __name__ == "__main__":
    asyncio.run(main())
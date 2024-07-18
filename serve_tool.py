import os
import correction_server
import asyncio
import fire

from logging import getLogger
logger = getLogger(__name__)
logger.setLevel("INFO")

def serve_tool(base_path = "/Volumes/ImageAnalysisHub/malbert/nhawkins/HAWKINS_highmag_sorted/dataset2_highmag_liposomesandproteinsATP/Good", server_url = "https://ai.imjoy.io"):

    logger.setLevel("DEBUG")

    loop = asyncio.get_event_loop()

    path2images = os.path.join(base_path, "membrainseg")
    path2labels = os.path.join(base_path, "skeletons")
    save_path = os.path.join(base_path, "corrections")

    loop.create_task(correction_server.start_server(server_url, path2images, path2labels, save_path))

    loop.run_forever()

    # 
    # server = await correction_server.start_server(server_url, path2images, path2labels, save_path)


if __name__ == '__main__':
  fire.Fire(serve_tool)




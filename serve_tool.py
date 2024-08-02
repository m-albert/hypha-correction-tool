import os
import correction_server
import asyncio
import fire

from logging import getLogger
logger = getLogger(__name__)
logger.setLevel("INFO")

def serve_tool(path2images = "/pasteur/zeus/projets/p02/ImageAnalysisHub/malbert/nhawkins/HAWKINS_highmag_sorted/training0", server_url = "https://ai.imjoy.io"):

    logger.setLevel("DEBUG")

    loop = asyncio.get_event_loop()

    loop.create_task(correction_server.start_server(server_url, path2images))

    loop.run_forever()

    # 
    # server = await correction_server.start_server(server_url, path2images, path2labels, save_path)


if __name__ == '__main__':
  fire.Fire(serve_tool)




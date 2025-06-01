from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, Request

import uvicorn
import tempfile
import os
import yt_dlp
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/download_youtube")
async def download_youtube(request: Request, background_tasks: BackgroundTasks):
    try:
        query_params = dict(request.query_params)
        url = query_params.get("url")
        session_id = query_params.get("session_id")

        if not url:
            return {"error": "Missing URL parameter"}

        # (Optional) log session_id for debugging
        if session_id:
            print(f"Download request received for session: {session_id}")

        # Create a temporary file path
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        #     output_path = tmp_file.name
        output_path = os.path.expanduser(f'~/Desktop/video_{session_id}.mp4')

        import subprocess
        cmd = [
            "yt-dlp",
            #"--cookies-from-browser", "chrome",
            "--cookies=/Users/luojiaxuan/PycharmProjects/updateCookies/cookies.txt",
            #"-f", "best[ext=mp4]/best",
            "-f", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "-o", output_path,
            url
        ]

        print("Current working dir:", os.getcwd())
        #print("cookies.txt", os.path.abspath("cookies.txt"))

        print("Running command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("yt-dlp failed:", result.stderr)
            raise Exception(f"yt-dlp error: {result.stderr}")
        else:
            print("yt-dlp success:", result.stdout)

        print("Download completed.")
        print(f"Checking file after download: {output_path}")
        if os.path.exists(output_path):
            print(f"File exists. Size: {os.path.getsize(output_path)} bytes")
        else:
            print("Download failed: File not found.")

        #background_tasks.add_task(os.remove, output_path)
        return FileResponse(output_path, media_type='video/mp4', filename=f'video_{session_id}.mp4')
    except Exception as e:
        return {"error": str(e)}
    # finally:
    #     if 'output_path' in locals() and os.path.exists(output_path):
    #         try:
    #             os.remove(output_path)
    #         except Exception:
    #             pass


if __name__ == "__main__":
    uvicorn.run("api-test:app", host="127.0.0.1", port=8001, reload=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import uuid
from typing import List, Dict
import json
from pdf2image import convert_from_path
from docx2pdf import convert
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ocr_service import process_image

app = FastAPI()

# Thread pool for CPU-bound OCR tasks
executor = ThreadPoolExecutor(max_workers=3)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def run_ocr_task(img_path, mode, prompt, iou_threshold, vis_path):
    return process_image(img_path, mode, prompt, iou_threshold, vis_path)

async def process_file_async(file_path, filename, file_id, mode, prompt, iou_threshold):
    try:
        ext = os.path.splitext(filename)[1].lower()
        temp_images = []
        
        # Notify: File received
        await manager.broadcast({
            "type": "log", 
            "file_id": file_id, 
            "message": f"Start processing {filename}..."
        })

        # 1. Convert to images
        if ext == ".pdf":
            await manager.broadcast({"type": "log", "file_id": file_id, "message": "Converting PDF to images..."})
            
            # Run conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(executor, convert_from_path, file_path)
            
            for i, img in enumerate(images):
                img_path = os.path.join(UPLOAD_DIR, f"{file_id}_page_{i}.jpg")
                img.save(img_path, "JPEG")
                temp_images.append((i+1, img_path))
                
        elif ext in [".docx", ".doc"]:
            await manager.broadcast({"type": "log", "file_id": file_id, "message": "Converting Word to PDF..."})
            
            pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
            subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", UPLOAD_DIR, file_path], check=True)
            
            generated_pdf = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
            if os.path.exists(generated_pdf):
                loop = asyncio.get_event_loop()
                images = await loop.run_in_executor(executor, convert_from_path, generated_pdf)
                for i, img in enumerate(images):
                    img_path = os.path.join(UPLOAD_DIR, f"{file_id}_page_{i}.jpg")
                    img.save(img_path, "JPEG")
                    temp_images.append((i+1, img_path))
            else:
                 raise Exception("Word conversion failed to generate PDF")
                 
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            temp_images.append((1, file_path))
            
        total_pages = len(temp_images)
        ocr_data = []
        file_result = {
            "filename": filename,
            "file_id": file_id,
            "pages": [],
            "json_url": ""
        }

        # 2. Process each page
        for idx, (page_num, img_path) in enumerate(temp_images):
            await manager.broadcast({
                "type": "progress", 
                "file_id": file_id, 
                "current": idx + 1, 
                "total": total_pages,
                "message": f"Processing page {page_num}/{total_pages}..."
            })
            
            print(f"[{file_id}] Processing page {page_num}/{total_pages} - {img_path}")
            
            vis_filename = f"{file_id}_page_{page_num}_vis.jpg"
            vis_path = os.path.join(RESULT_DIR, vis_filename)
            
            # Run OCR in thread pool
            loop = asyncio.get_event_loop()
            try:
                page_ocr = await loop.run_in_executor(executor, run_ocr_task, img_path, mode, prompt, iou_threshold, vis_path)
                print(f"[{file_id}] Page {page_num} completed.")
            except Exception as e:
                print(f"[{file_id}] Error processing page {page_num}: {e}")
                raise e
            
            ocr_data.append({
                "page": page_num,
                "results": page_ocr
            })
            
            file_result["pages"].append({
                "page_num": page_num,
                "vis_url": f"/results/{vis_filename}"
            })

        # 3. Save JSON
        json_filename = f"{file_id}.json"
        json_path = os.path.join(RESULT_DIR, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)
            
        file_result["json_url"] = f"/results/{json_filename}"
        
        # Notify: Complete
        await manager.broadcast({
            "type": "complete", 
            "file_id": file_id, 
            "result": file_result
        })
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        await manager.broadcast({
            "type": "error", 
            "file_id": file_id, 
            "message": str(e)
        })

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    mode: str = Form("Table"),
    prompt: str = Form("检测并识别图片中的文字，输出每段文本的坐标。特别注意表格内容，如果同一个单元格内的文字分多行显示，请务必将其合并为单行文本输出，不要分开。例如单元格内第一行是'ABC'，第二行是'D'，应直接输出'ABCD'。以JSON数组形式返回，每个元素包含text与bbox，bbox为[x1,y1,x2,y2]，坐标单位为像素，禁止返回非JSON内容。"),
    iou_threshold: float = Form(0.8)
):
    import time
    
    tasks = []
    # We need to broadcast to all connected clients? 
    # Or assume the client who uploaded is listening. 
    # Ideally, we should pass a session ID, but for simplicity, we broadcast or pick the first active connection.
    # The frontend should establish WS connection first.
    
    # Simple strategy: Broadcast to all connected clients (assuming single user or small team)
    # If multiple users, we need session_id. Let's assume simple use case.
    # target_ws = manager.active_connections[0] if manager.active_connections else None
    # Updated: Broadcast to all clients as target_ws logic was flaky
    
    file_info_list = []
    
    for file in files:
        timestamp = int(time.time() * 1000)
        filename = file.filename
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        file_id = f"{timestamp}_{name}"
        
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_info_list.append({
            "file_id": file_id,
            "filename": filename
        })

        # Create async task
        task = asyncio.create_task(
            process_file_async(file_path, filename, file_id, mode, prompt, iou_threshold)
        )
        tasks.append(task)
        
    # Return immediately with file IDs so frontend can track
    return JSONResponse(content={"status": "processing_started", "files": file_info_list})


@app.delete("/delete/{file_id}")
async def delete_file(file_id: str):
    # Try to clean up uploads and results with this ID
    # This is a basic cleanup, matching patterns
    try:
        for root, dirs, files in os.walk(UPLOAD_DIR):
            for file in files:
                if file.startswith(file_id):
                    os.remove(os.path.join(root, file))
        for root, dirs, files in os.walk(RESULT_DIR):
            for file in files:
                if file.startswith(file_id):
                    os.remove(os.path.join(root, file))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_all():
    try:
        for folder in [UPLOAD_DIR, RESULT_DIR]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

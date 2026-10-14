import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .models.database import init_db, SessionLocal, Slot
from .routers import auth, parking, admin, websocket
from .services.vision.worker import VisionWorker
from .services.vision.cache import occupancy_cache
from .services.reservation_manager import reservation_manager
from .routers.websocket import ws_manager
from .services.allocation import set_vision_worker as alloc_set_vision_worker

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "campus_config.json")

vision_worker = VisionWorker(CONFIG_PATH)
admin.set_vision_worker(vision_worker)
alloc_set_vision_worker(vision_worker)   # ← gives allocation service live camera access

# Connect CV loop events to reservation manager
def on_vision_update(slot_id: str, state: str, bikes: int, has_car: bool):
    reservation_manager.handle_cv_detection_event(slot_id, state, bikes, has_car)

vision_worker.register_callback(on_vision_update)

async def periodic_ws_broadcast():
    """Periodically pushes slot state updates to all active WebSocket clients."""
    while True:
        try:
            if ws_manager.active_connections:
                db = SessionLocal()
                slots = db.query(Slot).all()
                cache_all = occupancy_cache.get_all()
                data = {
                    "type": "SLOT_UPDATE",
                    "timestamp": asyncio.get_event_loop().time(),
                    "slots": [
                        {
                            "id": s.id,
                            "area_id": s.area_id,
                            "state": s.state,
                            "current_bikes": s.current_bikes,
                            "has_car": s.has_car,
                            "detected_state": cache_all.get(s.id, {}).get("detected_state", "FREE")
                        }
                        for s in slots
                    ]
                }
                db.close()
                await ws_manager.broadcast_json(data)
        except Exception as e:
            print(f"[WS-Broadcast] Error: {e}")
        await asyncio.sleep(1.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[FastAPI] Initializing Smart Campus Parking Database & Services...")
    init_db(CONFIG_PATH)
    reservation_manager.start()
    vision_worker.start()
    ws_task = asyncio.create_task(periodic_ws_broadcast())
    yield
    # Shutdown
    print("[FastAPI] Shutting down services...")
    ws_task.cancel()
    vision_worker.stop()
    reservation_manager.stop()

app = FastAPI(
    title="Smart Campus Parking System API",
    description="Next-generation intelligent parking allocation with computer vision ROI tracking, Dijkstra shortest path routing, and atomic concurrency controls.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(parking.router)
app.include_router(admin.router)
app.include_router(websocket.router)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "Smart Campus Parking Backend",
        "vision_worker_running": vision_worker.running,
        "reservation_manager_running": reservation_manager.running
    }

# Mount frontend web application at root (must be after all API routes)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

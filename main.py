
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from collections import deque
from utils.fight import fight_detector
from utils.weapon import weapon_detector
from utils.email_alert import send_email_alert
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Buffer for sequential fight detection
frame_buffer = deque(maxlen=16)

# Cooldown timers to prevent email spam
last_weapon_alert_time = 0
last_fight_alert_time = 0
ALERT_COOLDOWN = 60  # seconds

@app.websocket("/ws")
async def video_feed(websocket: WebSocket):
    global last_weapon_alert_time, last_fight_alert_time
    await websocket.accept()
    while True:
        try:
            # Receive and decode frame
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Append to fight buffer
            frame_buffer.append(frame)

            # Run models
            weapon = weapon_detector(frame)
            fight = (fight_detector(frame_buffer)
                     if len(frame_buffer) == frame_buffer.maxlen
                     else "Loading...")

            # Email alert logic with cooldown
            now = time.time()
            if weapon == "Weapon" and (now - last_weapon_alert_time > ALERT_COOLDOWN):
                send_email_alert(
                    "ShastraEye Alert: Weapon Detected 🚨",
                    "A weapon was detected by ShastraEye surveillance."
                )
                last_weapon_alert_time = now

            if fight == "Fights" and (now - last_fight_alert_time > ALERT_COOLDOWN):
                send_email_alert(
                    "ShastraEye Alert: Fight Detected ⚠️",
                    "A physical altercation was detected by ShastraEye surveillance."
                )
                last_fight_alert_time = now

            # Send predictions back
            await websocket.send_json({"fight": fight, "weapon": weapon})

        except Exception as exc:
            print(f"Connection closed or error: {exc}")
            break

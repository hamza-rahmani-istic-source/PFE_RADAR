#!/bin/bash
set -e

sudo cp /home/pi/ai_camera/systemd/ai-camera-final.service /etc/systemd/system/
sudo cp /home/pi/ai_camera/systemd/ai-camera-sender.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable ai-camera-final.service
sudo systemctl enable ai-camera-sender.service
sudo systemctl restart ai-camera-final.service
sudo systemctl restart ai-camera-sender.service

sudo systemctl status ai-camera-final.service --no-pager -l
sudo systemctl status ai-camera-sender.service --no-pager -l

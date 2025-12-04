# Fraud Approval UI - Quick Setup

## 1. Backend Service (Flask API)

Create `/etc/systemd/system/fraud-api.service`:

```ini
[Unit]
Description=Fraud Approval API
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/fraud-approval-ui
ExecStart=/usr/bin/python3 backend.py
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

## 2. Frontend Service (React UI)

Create `/etc/systemd/system/fraud-ui.service`:

```ini
[Unit]
Description=Fraud Approval React UI
After=network.target fraud-api.service

[Service]
Type=simple
WorkingDirectory=/root/fraud-approval-ui
ExecStart=/usr/bin/npm run preview
Restart=always

[Install]
WantedBy=multi-user.target
```

## 3. Commands

```bash
# Install
cd /root/fraud-approval-ui
pip install flask flask-cors mysql-connector-python requests
npm install
npm run build

# Enable services
systemctl daemon-reload
systemctl enable fraud-api fraud-ui
systemctl start fraud-api fraud-ui

# Check status
systemctl status fraud-api fraud-ui
```

## 4. Access

- **React UI:** http://SERVER_IP:2224
- **Flask API:** http://SERVER_IP:2225/api/
- **Login:** aris / Admin123

## Ports
- 2222 = Scoring Backend (existing)
- 2224 = React UI (new)
- 2225 = Flask API (new)

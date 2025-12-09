# ClaimFlow - Deployment Guide for Rocky Linux 8

## Prerequisites

- Rocky Linux 8 server with root access
- Domain name (optional, for production)
- MySQL database server (existing)

---

## Step 1: System Preparation

```bash
# Update system
sudo dnf update -y

# Install EPEL repository
sudo dnf install -y epel-release

# Install essential tools
sudo dnf install -y git curl wget nano vim
```

---

## Step 2: Install Node.js 20 LTS

```bash
# Install Node.js 20 via NodeSource
curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
sudo dnf install -y nodejs

# Verify installation
node --version   # Should show v20.x.x
npm --version
```

---

## Step 3: Install Python 3.11

```bash
# Install Python 3.11
sudo dnf install -y python3.11 python3.11-pip python3.11-devel

# Set as default (optional)
sudo alternatives --set python3 /usr/bin/python3.11

# Verify
python3.11 --version
```

---

## Step 4: Install Nginx

```bash
# Install Nginx
sudo dnf install -y nginx

# Enable and start
sudo systemctl enable nginx
sudo systemctl start nginx

# Configure firewall
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

---

## Step 5: Create Application User

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash claimflow
sudo usermod -aG wheel claimflow

# Create app directory
sudo mkdir -p /var/www/claimflow
sudo chown -R claimflow:claimflow /var/www/claimflow
```

---

## Step 6: Deploy Frontend

```bash
# Switch to app user
sudo su - claimflow
cd /var/www/claimflow

# Clone or copy your project files here
# If using git:
# git clone <your-repo-url> frontend

# Or create directory and copy files
mkdir -p frontend
cd frontend

# Copy all frontend files from claim-ui/ folder here

# Install dependencies
npm install

# Build for production
npm run build

# The built files will be in /var/www/claimflow/frontend/dist
```

---

## Step 7: Deploy Backend

```bash
cd /var/www/claimflow

# Create backend directory
mkdir -p backend
cd backend

# Copy backend files (app.py, requirements.txt)

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cat > .env << 'EOF'
SECRET_KEY=your-very-secure-secret-key-here
DB_HOST=cdpmsi.tomodachis.org
DB_USER=cloudera
DB_PASSWORD=T1ku$H1t4m
DB_NAME=claimdb
ADMIN_USER=aris
ADMIN_PASS=Admin123
EOF

# Test the app
python app.py

# Exit with Ctrl+C after testing
deactivate
```

---

## Step 8: Create Systemd Service for Backend

```bash
# Create service file
sudo nano /etc/systemd/system/claimflow-api.service
```

Add the following content:

```ini
[Unit]
Description=ClaimFlow API Server
After=network.target

[Service]
User=claimflow
Group=claimflow
WorkingDirectory=/var/www/claimflow/backend
Environment="PATH=/var/www/claimflow/backend/venv/bin"
EnvironmentFile=/var/www/claimflow/backend/.env
ExecStart=/var/www/claimflow/backend/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5000 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable claimflow-api
sudo systemctl start claimflow-api

# Check status
sudo systemctl status claimflow-api
```

---

## Step 9: Configure Nginx

```bash
sudo nano /etc/nginx/conf.d/claimflow.conf
```

Add the following configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Or your server IP

    # Frontend (React build)
    root /var/www/claimflow/frontend/dist;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # API proxy
    location /api {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }

    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # SPA routing - serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

```bash
# Test nginx config
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

---

## Step 10: SELinux Configuration (Important for Rocky Linux)

```bash
# Allow nginx to connect to network
sudo setsebool -P httpd_can_network_connect 1

# Allow nginx to serve files
sudo chcon -Rt httpd_sys_content_t /var/www/claimflow/frontend/dist

# If you still have issues
sudo setenforce 0  # Temporarily disable (for testing)

# To permanently allow, create custom policy:
sudo dnf install -y policycoreutils-python-utils
sudo audit2allow -a -M claimflow_policy
sudo semodule -i claimflow_policy.pp
```

---

## Step 11: SSL/HTTPS Setup (Optional but Recommended)

```bash
# Install certbot
sudo dnf install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
sudo systemctl status certbot-renew.timer
```

---

## Quick Commands Reference

```bash
# View backend logs
sudo journalctl -u claimflow-api -f

# Restart backend
sudo systemctl restart claimflow-api

# Restart nginx
sudo systemctl restart nginx

# Rebuild frontend after changes
cd /var/www/claimflow/frontend
npm run build
```

---

## Troubleshooting

### Check if services are running:
```bash
sudo systemctl status claimflow-api
sudo systemctl status nginx
```

### Check logs:
```bash
# Backend logs
sudo journalctl -u claimflow-api --since "1 hour ago"

# Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

### Test API directly:
```bash
curl http://localhost:5000/api/health
```

### Check port bindings:
```bash
sudo ss -tlnp | grep -E '(5000|80|443)'
```

---

## Directory Structure After Deployment

```
/var/www/claimflow/
├── frontend/
│   ├── dist/          # Built React app
│   ├── node_modules/
│   ├── src/
│   ├── package.json
│   └── ...
└── backend/
    ├── venv/          # Python virtual environment
    ├── app.py
    ├── requirements.txt
    └── .env           # Environment variables
```

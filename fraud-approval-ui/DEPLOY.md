# Fraud Approval UI - Deployment Guide for Rocky Linux 8

## Prerequisites

- Rocky Linux 8 server with root/sudo access
- Minimum 2GB RAM, 20GB disk
- Open ports: 80, 443, 3000 (dev), 2223 (backend)

---

## Step 1: System Update & Basic Tools

```bash
# Update system
sudo dnf update -y

# Install essential tools
sudo dnf install -y epel-release
sudo dnf install -y git curl wget vim firewalld
```

---

## Step 2: Install Node.js 20 LTS

```bash
# Install Node.js repository
curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -

# Install Node.js
sudo dnf install -y nodejs

# Verify installation
node --version  # Should show v20.x.x
npm --version
```

---

## Step 3: Install Nginx (Reverse Proxy)

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

## Step 4: Clone & Build the Project

```bash
# Create app directory
sudo mkdir -p /var/www/fraud-approval-ui
sudo chown $USER:$USER /var/www/fraud-approval-ui

# Clone or copy project files
cd /var/www/fraud-approval-ui

# If using git:
# git clone <your-repo-url> .

# Or copy files manually, then:
npm install

# Build for production
npm run build
```

---

## Step 5: Configure Nginx

Create nginx config:

```bash
sudo vim /etc/nginx/conf.d/fraud-approval.conf
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Or use server IP

    # Frontend (React build)
    root /var/www/fraud-approval-ui/dist;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

    # Frontend routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy to Flask backend
    location /api/ {
        proxy_pass http://127.0.0.1:2223/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Static assets caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Test and reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## Step 6: Setup Flask Backend as Service

Create systemd service:

```bash
sudo vim /etc/systemd/system/fraud-backend.service
```

Add:

```ini
[Unit]
Description=Fraud Approval Flask Backend
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/your/flask/app
ExecStart=/usr/bin/python3 app.py
Restart=always
RestartSec=10
Environment=FLASK_ENV=production

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable fraud-backend
sudo systemctl start fraud-backend
sudo systemctl status fraud-backend
```

---

## Step 7: SSL with Let's Encrypt (Optional but Recommended)

```bash
# Install certbot
sudo dnf install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
# Test with:
sudo certbot renew --dry-run
```

---

## Step 8: SELinux Configuration (if enabled)

```bash
# Allow Nginx to connect to upstream
sudo setsebool -P httpd_can_network_connect 1

# If serving from custom directory
sudo semanage fcontext -a -t httpd_sys_content_t "/var/www/fraud-approval-ui(/.*)?"
sudo restorecon -Rv /var/www/fraud-approval-ui
```

---

## Step 9: PM2 for Node Development Server (Alternative)

If you want to run dev server instead:

```bash
# Install PM2 globally
sudo npm install -g pm2

# Start app with PM2
cd /var/www/fraud-approval-ui
pm2 start npm --name "fraud-ui" -- run dev -- --host 0.0.0.0

# Save PM2 process list
pm2 save

# Setup PM2 to start on boot
pm2 startup systemd
```

---

## Quick Commands Reference

```bash
# Check services
sudo systemctl status nginx
sudo systemctl status fraud-backend

# View logs
sudo journalctl -u fraud-backend -f
sudo tail -f /var/log/nginx/error.log

# Rebuild after changes
cd /var/www/fraud-approval-ui
npm run build
sudo systemctl reload nginx

# Restart everything
sudo systemctl restart nginx fraud-backend
```

---

## Troubleshooting

### Port already in use
```bash
sudo lsof -i :3000
sudo kill -9 <PID>
```

### Permission denied
```bash
sudo chown -R nginx:nginx /var/www/fraud-approval-ui/dist
sudo chmod -R 755 /var/www/fraud-approval-ui/dist
```

### API not connecting
```bash
# Check if backend is running
curl http://127.0.0.1:2223/api/get_claim
```

---

## Security Recommendations

1. Keep system updated: `sudo dnf update -y`
2. Use strong passwords in Flask app
3. Enable firewall rules for specific IPs if needed
4. Regular backups of database
5. Monitor logs for suspicious activity

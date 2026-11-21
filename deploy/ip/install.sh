#!/usr/bin/env bash
# 在服务器上运行：校验并解压上传的分卷，安装到 /srv/stm，重启服务。
#   cd 分卷所在目录 && sudo bash install.sh
set -euo pipefail
cd "$(dirname "$0")"
echo "== 校验分卷"
sha256sum -c stm-release.tar.zst.sha256
echo "== 解压到 /srv/stm"
systemctl stop stm 2>/dev/null || true
mkdir -p /srv/stm
cat stm-release.tar.zst.part-* | zstd -dc | tar -x -C /srv/stm
chown -R stm:stm /srv/stm
cp /srv/stm/deploy/ip/Caddyfile /etc/caddy/Caddyfile
cp /srv/stm/deploy/ip/stm.env /etc/stm.env
cp /srv/stm/deploy/ip/stm.service /etc/systemd/system/stm.service
systemctl daemon-reload
systemctl enable --now stm
systemctl restart stm
systemctl reload caddy || systemctl restart caddy
echo "== 等待应用启动"
for i in $(seq 1 60); do curl -fsS localhost:8787/api/health && break; sleep 2; done
echo
curl -fsS -o /dev/null -w "首页 HTTP %{http_code}\n" localhost/
echo "部署完成：http://$(curl -fsS --max-time 3 ifconfig.me 2>/dev/null || hostname -I | cut -d' ' -f1)/"

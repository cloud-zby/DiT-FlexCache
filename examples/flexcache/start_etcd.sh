rm -rf default.etcd
# etcd --debug --listen-client-urls=http://0.0.0.0:2379 --advertise-client-urls=http://0.0.0.0:2380

etcd \
  --name s1 \
  --data-dir /tmp/etcd-data \
  --listen-client-urls "http://0.0.0.0:2379" \
  --advertise-client-urls "http://127.0.0.1:2379" \
  --listen-peer-urls "http://0.0.0.0:2380" \
  --initial-advertise-peer-urls "http://127.0.0.1:2380" \
  --initial-cluster "s1=http://127.0.0.1:2380" \
  --initial-cluster-state new \
  # --log-level debug

#   --listen-client-urls：容器里 etcd 接收客户端请求（绑定 0.0.0.0:2379）。
# --advertise-client-urls：告诉客户端该用哪个地址访问。这里用 127.0.0.1:2379（如果你要从容器外连，就改成容器可达的 IP）。
# --listen-peer-urls / --initial-advertise-peer-urls：集群内部通信用 2380（即使单机也要写）。
# --log-level debug：替代旧版本的 --debug。
# /tmp/etcd-data：数据目录（可以换别的路径）。
